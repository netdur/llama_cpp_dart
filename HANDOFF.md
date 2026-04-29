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

`minSdkVersion` requirements differ per AAR:
- **CPU AAR** (`llama-cpp-dart.aar`): `minSdkVersion 26` (Android 8.0).
- **Hexagon AAR** (`llama-cpp-dart-hexagon.aar`): `minSdkVersion 31` (Android 12). The AAR's manifest declares 31 because OpenCL on Android requires the vendor-lib opt-in introduced in API 31, and Gradle's manifest merger refuses to lower it.

### Hexagon AAR — five mandatory ingredients

Validated end-to-end on Galaxy S23 Ultra (SD 8 Gen 2) and Galaxy Fold7 (SD 8 Elite). Miss **any** of these and the HTP backend silently disappears from `engine.devices`.

**1. minSdk 31 + arm64-v8a-only Gradle.** Already covered above for the Hexagon AAR.

**2. Two `<uses-native-library>` opt-ins** in your app's `AndroidManifest.xml` inside `<application>`:

```xml
<application ... android:extractNativeLibs="true">
    <uses-native-library android:name="libOpenCL.so"  android:required="false" />
    <uses-native-library android:name="libcdsprpc.so" android:required="false" />
    ...
</application>
```

- `libOpenCL.so` — the Hexagon-built `libllama.so` has a hard `DT_NEEDED` on `libggml-opencl.so`, which transitively needs vendor `libOpenCL.so`. Without it `dlopen` fails on first load.
- `libcdsprpc.so` — `ggml-hexagon` does `dlopen("libcdsprpc.so")` by basename. Without this opt-in, Android's per-app linker namespace blocks access and the HTP backend silently fails to register (the failure log goes to stderr, which is invisible on Android — see `LlamaLog.captureToFile` below).

`required="false"` lets the app still install on devices without OpenCL/cDSP — runtime falls back to CPU.

**3. `android:extractNativeLibs="true"`** (shown above). Modern AGP defaults to `false`, which leaves `.so` files inside the APK and mmaps them. That works for the app's own linker but **not** for FastRPC: the cDSP-side skeleton libs (`libggml-htp-v73.so` etc.) are loaded by `cdsprpcd` (a separate system process) via `fopen()`, which can't read inside an APK. With `extractNativeLibs="true"`, libs are extracted to `applicationInfo.nativeLibraryDir` at install time where `cdsprpcd` can read them.

Without it you'll see (only via stderr capture):
```
W cdsprpcd: apps_std_fopen_with_env failed with 0x2 for libggml-htp-v73.so
E ggml-hex: remote_handle_open_domain: dynamic loading failed
```

**4. `ADSP_LIBRARY_PATH`** is now auto-set by the binding. As of `0.9.0-dev.2`, `LlamaLibrary.load()` reads `/proc/self/maps` to find the resolved `libllama.so` path on Android and calls `setenv("ADSP_LIBRARY_PATH", dir, 1)` before `ggml_backend_load_all`. **No app-side wiring required.** If you were carrying a `MethodChannel` + manual `setenv` hack from earlier dev builds, you can drop it.

**5. llama.cpp pin must include the URI-fallback fix** (commit `63d2fc46` or later — already in `0.9.0-dev.1`). On SD 8 Gen 2 specifically, `remote_session_control(FASTRPC_GET_URI, ...)` returns error `0x14` and ggml-hex aborts HTP init without it.

You may also see harmless warnings at install time:

```
Failed to punch uncompressed elf file: libggml-htp-v*.so
```

These are the Hexagon DSP-target ELFs (one per HTP variant). The Android linker doesn't recognize the Hexagon machine type so it skips ELF page-alignment optimization. Cosmetic only — the DSP libraries are loaded by the Hexagon backend at runtime via FastRPC, not by the Android linker.

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

## Verifying which backends actually loaded

**No AAR rebuild needed for this** — it's a pure-Dart addition. Just
`git pull` the binding and `flutter pub get` in the demo app.

After spawning the engine:

```dart
final engine = await LlamaEngine.spawn(...);
debugPrint('=== backends ===');
for (final d in engine.devices) {
  debugPrint('$d');
}
debugPrint('hasAccelerator     = ${engine.hasAccelerator}');
debugPrint('primaryAccelerator = ${engine.primaryAcceleratorName}');
```

Expected output by AAR (observed on Galaxy S23 Ultra / SD 8 Gen 2 + Fold7 / SD 8 Elite):

| AAR | `engine.devices` should include | `primaryAcceleratorName` |
|---|---|---|
| CPU AAR | `CPU (cpu, CPU)` only | `null` |
| Hexagon AAR | `GPUOpenCL (gpu, OpenCL): QUALCOMM Adreno(TM)`, `HTP0 (gpu, HTP): Hexagon`, `CPU (cpu, CPU)` | `HTP0` |

Notes:
- HTP reports `type=gpu, registryName=HTP` (not `accel/Hexagon`) — that's how ggml-backend classifies it.
- `primaryAcceleratorName` returns `HTP0` because the binding's picker promotes registry name `HTP` above `OpenCL`. (This was fixed in `0.9.0-dev.2`; in `0.9.0-dev.1` the picker followed registration order and returned `GPUOpenCL`.)
- Apple platforms see Metal + (on iOS) Accelerate BLAS + CPU. Example macOS/M1 Max: `MTL0 (gpu, Metal): Apple M1 Max`, `CPU`.

If the Hexagon AAR is in `android/app/libs/` and the manifest has the
`<uses-native-library>` line for `libOpenCL.so` but **no `HTP0` shows
up** in `engine.devices`, the DSP libraries failed to load silently.
Most likely causes:
- Missing one of the six `libggml-htp-v*.so` files from the AAR.
- Device's HTP firmware is too old for the variants we ship (v68
  through v81 covers Snapdragon 865 → 8 Elite; older devices fall back
  to CPU).
- FastRPC service unreachable (Android security policy or selinux).

Pre-engine variant (no model load required) is also available for
quicker probes:

```dart
LlamaLibrary.load(path: 'libllama.so');
final devices = LlamaBackends.list();
```

Both paths surface the same data; `engine.devices` is captured once
inside the worker isolate at engine spawn and survives across
generate calls.

## Closed binding gaps in `0.9.0-dev.2`

Surfaced during M10 device validation; all four shipped:

- ✅ **`typeK` / `typeV` in `ContextParams`.** Use `KvCacheType.q8_0` (or
  any other supported ggml type) to quantize the KV cache. Common
  memory-saver on 8 GB Android devices with longer contexts. Set both
  `typeK` and `typeV` to the same value when FlashAttention is on —
  most backends require it.
- ✅ **Backend inspection API.** `engine.devices` (`List<BackendDevice>`),
  `engine.hasAccelerator`, `engine.primaryAcceleratorName`, or
  `LlamaBackends.list()` for the pre-engine view. Doesn't tell you
  which device ran an individual op (ggml-backend distributes ops
  across devices) but answers "is Hexagon loaded? is OpenCL loaded?"
  cleanly. `primaryAcceleratorName` now uses a registry-priority list
  so HTP wins over OpenCL on Snapdragon (was registration-order
  before).
- ✅ **Stderr capture** for llama.cpp / ggml log lines. Toggleable —
  off by default. Use:
  ```dart
  LlamaLog.captureToFile('${docsDir.path}/llama.log');
  // ... later, to revert:
  LlamaLog.restoreStderr();
  ```
  On Android, tail the file and forward lines to `debugPrint` since
  stderr is not connected to logcat. This was the single biggest
  debugging blocker during M10.
- ✅ **Auto-`setenv ADSP_LIBRARY_PATH`** on Android. `LlamaLibrary.load()`
  now reads `/proc/self/maps` to find where the linker resolved
  `libllama.so` and sets the env var so FastRPC can locate
  `libggml-htp-v*.so`. No `MethodChannel` plumbing needed in the app.

## Things that won't change before `1.0`

- **HTP only engages Q4_0 / Q8_0 quants.** K-quants (`Q4_K_*`,
  `Q5_K_*`) and I-quants (`IQ*`) have no HTP kernels in upstream
  ggml-hexagon. Pick `Q4_0` if you want NPU acceleration; `Q4_K_M`
  will run on OpenCL+CPU only on Snapdragon devices.
- **HTP REPACK budget is ~2 GB per session.** Models larger than that
  get partial-offload (some layers HTP, rest OpenCL). For `≥7B`-class
  models you need the `NDEV=2` multi-session pattern, which the
  binding does not expose yet.
- **Engine spawn time on Hexagon AAR is ~11–13s** (vs ~6s for CPU
  AAR). Most of the extra time is OpenCL kernel pre-compilation +
  HTP REPACK buffer mmap. Plan a splash screen on first launch.

## Apple integration (iOS / macOS)

If you're building the same app for iPhone / iPad / Mac as well, there are **two distinct paths**. Pick the wrong one and you'll get `dlsym(RTLD_DEFAULT, llama_backend_init): symbol not found` at runtime.

### Path A — Dev/test on macOS (loose dylib)

For `dart test`, CLI probes, and Flutter macOS apps where you control where the dylib lives:

```dart
final engine = await LlamaEngine.spawn(
  libraryPath: '/abs/path/to/libllama.dylib',
  modelParams: ModelParams(path: modelPath, gpuLayers: 99),
  contextParams: const ContextParams(nCtx: 2048),
);
```

`LlamaLibrary.load(path: ...)` resolves siblings (`libggml-metal.dylib`, `libmtmd.dylib`, ...) from the same directory automatically. Metal lights up on first matmul.

For Flutter macOS specifically, **disable App Sandbox in `DebugProfile.entitlements`** if the dylib or model files live outside the app bundle — sandboxed apps can't read arbitrary `/Users/...` paths. Keep `Release.entitlements` sandboxed for distribution; that means Release builds need Path B.

### Path B — Shippable iOS / macOS app (xcframework via CocoaPods)

Required for any iOS app and any sandboxed Release macOS app. Uses `build/apple/llama.xcframework` from this repo.

**The xcframework is static**, not dynamic — `file build/apple/llama.xcframework/ios-arm64/llama.framework/llama` reports `current ar archive`. This means:

1. **"Embed & Sign" is wrong.** Use **"Do Not Embed"**.
2. **`-force_load` is mandatory** — without it, the linker drops every llama symbol because nothing in `AppDelegate.swift` references them, and you get `dlsym not found` at runtime.

A working podspec ships at `llama_cpp.podspec` in this repo. From a Flutter app's `ios/Podfile` (and `macos/Podfile`):

```ruby
platform :ios, '14.0'   # xcframework's MinimumOSVersion is 14.0
target 'Runner' do
  use_frameworks!
  flutter_install_all_ios_pods File.dirname(File.realpath(__FILE__))
  pod 'llama_cpp', :path => '../../llama_cpp_dart'
end
```

The podspec wires `OTHER_LDFLAGS = -force_load $(PODS_XCFRAMEWORKS_BUILD_DIR)/llama_cpp/llama.framework/llama` for both `pod_target_xcconfig` and `user_target_xcconfig`.

After `pod install`, verify with:

```bash
nm -gU build/ios/iphoneos/Runner.app/Runner.debug.dylib | grep llama_backend_init
```

— if `_llama_backend_init` shows up, link succeeded.

In Dart, **use `spawnFromProcess` not `spawn`**:

```dart
final engine = await LlamaEngine.spawnFromProcess(
  modelParams: ModelParams(path: modelPath, gpuLayers: 99),
  contextParams: const ContextParams(nCtx: 2048),
);
```

There's nothing to `dlopen` by path on iOS — symbols are already in the running process.

### iOS gotchas (validated on iPad M1, iOS 26.3)

- **iOS apps cannot `dlopen` arbitrary dylib paths.** Path A is fundamentally not available on iOS. Always Path B.
- **Model files must live in the app's Documents directory** (`getApplicationDocumentsDirectory()`). Push large models with `xcrun devicectl device copy to --domain-type appDataContainer --domain-identifier <bundle-id> --source <local> --destination Documents/<file>`. Set `UIFileSharingEnabled = YES` and `LSSupportsOpeningDocumentsInPlace = YES` in `Info.plist`.
- **First-run Metal shader compile is ~18s on iPad M1**, ~9.5s on M1 Max — show a splash screen. After the binary cache is built, subsequent spawns are sub-second.
- **Free Apple Developer profile caps you at 3 installed apps from that team.** A fourth install fails silently (`flutter run` hangs at "Dart VM Service was not discovered after 60 seconds"). Delete one to install another.
- **Flutter's Profile xcconfig is missing by default.** `flutter create` only generates `Debug.xcconfig` and `Release.xcconfig` but Xcode has three configs. Profile inherits Release, so CocoaPods' Profile pods xcconfig is never included. Fix once per project: create `ios/Flutter/Profile.xcconfig` with `#include? "Pods/Target Support Files/Pods-Runner/Pods-Runner.profile.xcconfig"` + `#include "Generated.xcconfig"`, then set Runner's Profile config base to it. Otherwise Profile builds (e.g. `xcodebuild archive`) fail with `Framework 'Pods_Runner' not found`.
- **`engine.devices` on iOS reports 3 backends**: `MTL0` (Metal), `BLAS` (Accelerate), `CPU`. On macOS only Metal + CPU. Both are correct.

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
