# Handoff: build an Android demo app that uses the llama_cpp_dart AAR

You are taking over from a previous Claude instance that built and
shipped `llama_cpp_dart 0.9.0-dev.1`. The Dart binding works on
macOS (35/35 tests green). The Android AAR builds cleanly in CI but
**has not been exercised by a real app on a real device yet**. Your
job is to build that app.

This document is the entire context you need. You do not have to read
the prior conversation.

## Mission

Build a small **Flutter Android app** that:

1. Loads a small GGUF model bundled as an app asset.
2. Has a chat UI with a text field and a scrolling token stream.
3. Streams tokens off the UI thread via `LlamaEngine`.
4. Demonstrates that cancellation works (stop button).
5. (Stretch) Image input — let the user pick a photo, attach it to a
   chat turn, get a description back.
6. (Stretch) Saves chat state to local storage and reloads it on
   restart.

Acceptance: the app installs on an Android emulator or phone, model
loads, prompt → streamed reply works, cancel button mid-stream stops
generation cleanly. The reply text is whatever the model produces; we
are validating the **plumbing**, not the model quality.

The binding is at this repo (`/Users/adel/Workspace/llama_cpp_dart`,
also at https://github.com/netdur/llama_cpp_dart). It does **not**
need to be installed from pub.dev; reference it as a `path:`
dependency from the demo app.

## What you have to work with

### Native artifacts

The Android AAR is built by `tool/build_android_aar.sh` and CI. Two
flavors exist; **start with the CPU-only one**:

| AAR | Path in this repo | Size |
|---|---|---|
| `llama-cpp-dart.aar` (CPU + mtmd, arm64-v8a) | `build/android/llama-cpp-dart.aar` | ~2 MB |
| `llama-cpp-dart-hexagon.aar` (CPU + OpenCL + Hexagon NPU + mtmd) | `build/android-hexagon/llama-cpp-dart-hexagon.aar` | ~3.7 MB |

Both also live as GitHub Release assets:
https://github.com/netdur/llama_cpp_dart/releases/tag/v0.9.0-dev.1.

**Use the CPU AAR for the first end-to-end test.** It works on every
Android device including emulators. Once that's working, switch to
the Hexagon AAR to validate NPU acceleration on a real Snapdragon
phone (8 Gen 2 or newer).

The AAR contains, under `jni/arm64-v8a/`:

- CPU AAR: `libllama.so`, `libggml.so`, `libggml-base.so`,
  `libggml-cpu.so`, `libmtmd.so`
- Hexagon AAR: same plus `libggml-opencl.so`, `libggml-hexagon.so`,
  and six DSP-side `libggml-htp-v{68,69,73,75,79,81}.so` files
  (Hexagon ELF, runtime picks the right variant per device).

### Dart binding

`llama_cpp_dart` exports the user-facing API. The relevant pieces:

```dart
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

// 1. Spawn an engine. On Android the libraryPath is a basename;
//    Android's dynamic linker resolves it from the app's lib path.
final engine = await LlamaEngine.spawn(
  libraryPath: 'libllama.so',
  modelParams: ModelParams(path: '/data/.../model.gguf', gpuLayers: 0),
  contextParams: const ContextParams(nCtx: 2048, nBatch: 512, nUbatch: 512),
);

// 2. Create a chat (uses the model's embedded Jinja template).
final chat = await engine.createChat();
chat.addSystem('You are concise.');
chat.addUser('Hello');

// 3. Stream tokens.
await for (final event in chat.generate(maxTokens: 256)) {
  switch (event) {
    case TokenEvent():
      // event.text is the streaming UTF-8 chunk.
      onToken(event.text);
    case ShiftEvent():
      // KV was shifted; usually no UI action needed.
    case DoneEvent():
      // event.reason is StopEog | StopMaxTokens | StopUserAbort.
      onDone(event.reason);
  }
}

// 4. Clean up.
await chat.dispose();
await engine.dispose();
```

Cancellation: `StreamSubscription.cancel()` (or breaking out of
`await for`) propagates a `CancelCommand` to the worker isolate.
For UI use:

```dart
StreamSubscription? sub;
sub = chat.generate(maxTokens: 256).listen(handleEvent,
  onDone: () { sub = null; });

// Stop button:
sub?.cancel();
```

For multimodal (vision):

```dart
final engine = await LlamaEngine.spawn(
  libraryPath: 'libllama.so',
  modelParams: ModelParams(path: llmPath, gpuLayers: 0),
  contextParams: const ContextParams(nCtx: 4096, nBatch: 512, nUbatch: 512),
  multimodalParams: MultimodalParams(mmprojPath: mmprojPath),
);
final chat = await engine.createChat();
chat.addUser(
  'Describe this image.',
  media: [LlamaMedia.imageFile('/data/.../picture.jpg')],
);
// ... same generate loop
```

For persistence:

```dart
await session.saveState('/data/.../chat.lcdc');
// ... later
await otherSession.loadState('/data/.../chat.lcdc');
```

Other public types you'll likely touch: `SamplerParams`,
`KnownChatTemplates`, `LlamaStateException`. Full list in
`lib/llama_cpp_dart.dart`.

### Existing Dart probes

`example/probes/` has runnable scripts that demonstrate each subsystem
on macOS. Read them to understand the API in motion:

- `engine_stream.dart` — basic streaming via `LlamaEngine`.
- `chat_two_turn.dart` — multi-turn chat with `EngineChat`.
- `vision_describe.dart` — image input.
- `save_resume.dart` — state save/load across engine restarts.

Adapt their patterns into Flutter widgets in the demo app.

## Recommended app shape

A single-screen app is enough:

```
+----------------------------------------+
| llama demo                              |
+----------------------------------------+
|                                         |
|  [chat history scrollview]              |
|  ...                                    |
|                                         |
|  user: hi                               |
|  assistant: hello there|                |  <- streaming cursor
|                                         |
+----------------------------------------+
| [text input              ] [send]      |
+----------------------------------------+
| status: 12 tok/s · 23 of 256            |
+----------------------------------------+
```

A second pass adds a 📷 button next to the input that opens
`image_picker`, attaches the image to the next user turn, and shows
the picture inline in the chat history.

State machine: `idle → loadingModel → ready → generating → ready`.
Send button is disabled outside `ready`; turns into a stop button
during `generating`.

## Project skeleton

```bash
# from anywhere
cd /Users/adel/Workspace
flutter create --org dev.netdur --platforms android llama_demo
cd llama_demo
```

In `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  llama_cpp_dart:
    path: ../llama_cpp_dart
  path_provider: ^2.1.5   # to find the docs dir for model files
  image_picker: ^1.1.2    # only if you do the vision stretch
```

Drop the AAR:

```bash
mkdir -p android/app/libs
cp ../llama_cpp_dart/build/android/llama-cpp-dart.aar android/app/libs/
```

In `android/app/build.gradle` (or `build.gradle.kts`), add the AAR to
the dependencies block:

```gradle
dependencies {
    implementation files('libs/llama-cpp-dart.aar')
}
```

Make sure `android/app/build.gradle` has `arm64-v8a` in `ndk.abiFilters`
(skip `armeabi-v7a` and `x86*` — the AAR doesn't ship those):

```gradle
android {
    defaultConfig {
        ndk {
            abiFilters 'arm64-v8a'
        }
    }
}
```

`minSdkVersion` should be 26 or higher (matches the AAR's manifest).

## Bundling a model

Don't ship multi-gigabyte models in assets. Pick a small one:

- **SmolLM2-360M-Instruct** — Q4_K_M is ~270 MB. Has a chat template.
- **Qwen2.5-0.5B-Instruct** — Q4_K_M is ~400 MB. Slightly bigger but
  better instruction following.
- **Gemma-3-270m-it** — small Gemma if you can find a recent quant.

Put the GGUF in `assets/models/` and add to `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/smollm2-360m-q4.gguf
```

On first run, copy from the asset bundle to the documents directory
(llama.cpp needs a real filesystem path, not an asset URI):

```dart
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'dart:io';

Future<String> ensureModelOnDisk() async {
  final dir = await getApplicationDocumentsDirectory();
  final dest = '${dir.path}/smollm2-360m-q4.gguf';
  if (!File(dest).existsSync()) {
    final bytes = await rootBundle.load('assets/models/smollm2-360m-q4.gguf');
    await File(dest).writeAsBytes(bytes.buffer.asUint8List(), flush: true);
  }
  return dest;
}
```

## Things that will trip you up

- **Don't pass an absolute path on Android.** `LlamaEngine.spawn(libraryPath: 'libllama.so', ...)` — basename. Android's linker resolves from the app's `lib/arm64-v8a/` dir.
- **`gpuLayers: 0`** on Android (the CPU AAR has no GPU backend). With the Hexagon AAR, leave `gpuLayers: 0` too — the Hexagon backend kicks in automatically via ggml-backend device selection, not via gpuLayers.
- **App size**: the AAR adds ~2 MB; the model is the elephant in the room. Use Q4_K_M, not F16.
- **First-run model copy is slow.** Do it once on a splash/loading screen, cache the path.
- **Custom Jinja chat templates** can fail. SmolLM2 and Qwen2.5 work via `EngineChat` directly. If you pick something exotic and `EngineChat` returns no tokens, fall back to `EngineSession.generate(prompt:)` with a manually-formatted prompt. There are worked examples in `example/probes/gemma_chat.dart`.
- **Permissions**: image_picker needs `READ_MEDIA_IMAGES` on Android 13+.
- **Worker isolate**: `LlamaEngine.spawn` already runs the model in a worker isolate. Don't wrap it in another `compute()` or `Isolate.spawn`.
- **Disposal order**: `chat.dispose()` → `engine.dispose()`. The engine kills the worker isolate; do it after every chat is done.

## Testing matrix

In order of effort:

1. **Android emulator (x86_64 host arch).** ❌ The AAR is arm64-v8a only. Use a Pixel-arm64 emulator image, not the default x86_64.
2. **Pixel-arm64 emulator.** Slow inference but proves CPU pipeline works.
3. **Physical Android phone (any).** Real-world inference speed.
4. **Snapdragon phone (8 Gen 2 / 8 Gen 3 / 8 Elite).** Swap to the Hexagon AAR; verify NPU activates. The runtime should pick the matching `libggml-htp-v*.so` automatically. Look for log lines mentioning `HTP` or `hexagon`.

For step 4, you can use Hugind itself as a reference — they target the same NPU stack. See `/Users/adel/Workspace/hugind/src/server/llm/runtime/`.

## When you're done

Stage the demo app at `/Users/adel/Workspace/llama_demo/` (or wherever),
commit it as a separate repo. The llama_cpp_dart binding stays at its
current commit; if you find a Dart-side bug, file it against
`netdur/llama_cpp_dart` and we'll patch + bump to `0.9.0-dev.2`.

Update `plan.md` row M10 from `⏳ next` to `✅ shipped` once the
emulator path is verified, and add a note for what tested devices
hit Hexagon. If anything surprised you, capture it under "Decisions
that flipped during the rewrite" in `plan.md`.

## References

- `README.md` — install + quick start + public API surface.
- `BUILD.md` — how the AAR is built (in case you need to rebuild).
- `MIGRATION.md` — only relevant if you've used the 0.2.x line.
- `plan.md` — full milestone tracker; M10 is the gate you're working on.
- `example/probes/*.dart` — runnable Dart scripts that demonstrate
  each subsystem; these are your best reference for API patterns.
- `lib/llama_cpp_dart.dart` — the public exports list.

Good luck.
