# llama_cpp_dart

Dart FFI binding for [llama.cpp](https://github.com/ggml-org/llama.cpp), targeting **iOS, Android, and macOS** for Flutter mobile apps.

> **Status:** 0.9.x — clean rewrite of the 0.2 binding. Public API will likely have one more breaking pass before 1.0.

## Project direction

I originally intended this to be a Dart-only binding that also worked from Flutter — same package serving CLI, server, desktop, and mobile use cases.

In practice that scope ran into hard limits: continuous batching, multi-process agent runtimes, OpenAI-compatible HTTP, and tool-use orchestration are all easier to express in a language with proper threads, fewer FFI quirks, and a richer ecosystem. So I started a separate project — [netdur/hugind](https://github.com/netdur/hugind) — that takes the server / agent / desktop role in Rust.

This repo (`llama_cpp_dart`) now focuses on one thing: **llama.cpp inside a Flutter mobile app**. The 0.9.x rewrite reflects that scope: the public API is single-active-session, off-thread, multimodal-aware, and packaged for iOS / Android only. macOS sticks around as a development target because that's what Flutter devs build on.

I also build and ship the native binaries from this repo's CI — Apple xcframework, macOS dylib, Android CPU AAR, and Android Hexagon AAR — so consumers don't need to touch CMake, NDK, or the Snapdragon Docker toolchain.

## Highlights

- Streaming token output via `Stream<GenerationEvent>`.
- Off-thread inference via `LlamaEngine` worker isolate (UI never blocks).
- **Multimodal**: vision and audio through llama.cpp's `mtmd` (image and audio bitmaps go in, the model emits text).
- **Chat**: uses the model's embedded Jinja chat template via `llama_chat_apply_template`. Falls back to manual prompt rendering for models with custom Jinja the C API can't parse.
- **Persistence**: KV-cache + token history + chat messages save/restore to a single self-describing file with metadata-validated reload.
- **Context shift**: `llama-server`-style auto-shift when the context fills (off by default, opt in per request, blocked on caches that can't shift).
- Apple **Metal** + Android **CPU** + Snapdragon **Hexagon NPU** + **OpenCL** acceleration (Hexagon AAR pending physical-device validation).

## Install

The Dart package contains no native binaries — they're shipped per platform from GitHub Releases.

```yaml
dependencies:
  llama_cpp_dart: ^0.9.0
```

Then download the platform binary for your project:

| Platform | Artifact | Where to put it |
|---|---|---|
| macOS (dev/test) | `libllama.dylib` + sibling `libggml*.dylib`, `libmtmd.dylib` | anywhere on disk; pass path to `LlamaEngine.spawn` |
| iOS / macOS app | `llama.xcframework` (3 slices: `ios-arm64`, `ios-arm64-simulator`, `macos-arm64`) | drag into Xcode → "Embed & Sign" → call `LlamaEngine.spawnFromProcess` |
| Android | `llama-cpp-dart.aar` (CPU + mtmd, arm64-v8a) **or** `llama-cpp-dart-hexagon.aar` (CPU + OpenCL + Hexagon NPU + mtmd, arm64-v8a, Snapdragon) | `android/app/libs/` and `implementation files('libs/llama-cpp-dart.aar')` in Gradle |

Build artifacts yourself with:

```bash
tool/build_native.sh --platform macos --with-mtmd
tool/build_apple_xcframework.sh
tool/build_android_aar.sh                 # CPU AAR
tool/build_android_hexagon_aar.sh         # Hexagon NPU + OpenCL AAR (Snapdragon)
```

## Example app

A working Flutter chat app built on this binding lives at [netdur/imaged-sdk-examples — `aichat`](https://github.com/netdur/imaged-sdk-examples/tree/main/aichat). Useful as a reference for wiring `LlamaEngine`, streaming events, and chat templates into a real UI.

## Quick start

### Streaming generation

```dart
final engine = await LlamaEngine.spawn(
  libraryPath: '/path/to/libllama.dylib',
  modelParams: ModelParams(path: '/path/to/model.gguf', gpuLayers: 99),
  contextParams: const ContextParams(nCtx: 4096),
);

final session = await engine.createSession();
await for (final event in session.generate(
  prompt: 'Once upon a time',
  addSpecial: true,
  sampler: const SamplerParams(temperature: 0.7, topP: 0.9),
  maxTokens: 128,
)) {
  switch (event) {
    case TokenEvent():
      stdout.write(event.text);
    case ShiftEvent():
      // KV was shifted to make room. Bookkeeping; usually ignored.
    case DoneEvent():
      stdout.writeln('\n[${event.reason}, ${event.generatedCount} tokens]');
  }
}

await session.dispose();
await engine.dispose();
```

### Chat with a model whose template `llama_chat_apply_template` recognizes

```dart
final chat = await engine.createChat();
chat.addSystem('You are concise.');
chat.addUser('What is 2+2?');

await for (final event in chat.generate(maxTokens: 64)) {
  if (event is TokenEvent) stdout.write(event.text);
}
// chat.messages now holds [system, user, assistant]
```

For models that ship custom Jinja the matcher can't parse (some Unsloth quants), pass a sentinel string:

```dart
chat.generate(templateOverride: KnownChatTemplates.gemma);
```

If even that fails, format the prompt yourself and use `EngineSession.generate(prompt:)` directly. See `example/probes/gemma_chat.dart` for a worked example.

### Multimodal (vision + audio)

```dart
final engine = await LlamaEngine.spawn(
  libraryPath: '/path/to/libllama.dylib',
  modelParams: ModelParams(path: '/path/to/llm.gguf', gpuLayers: 99),
  contextParams: const ContextParams(nCtx: 4096),
  multimodalParams: const MultimodalParams(mmprojPath: '/path/to/mmproj.gguf'),
);

print('vision=${engine.supportsVision} audio=${engine.supportsAudio} '
      'rate=${engine.audioSampleRate}');

final chat = await engine.createChat();
chat.addUser(
  'Describe this image.',
  media: [LlamaMedia.imageFile('cat.jpg')],
);

await for (final event in chat.generate(maxTokens: 128)) {
  if (event is TokenEvent) stdout.write(event.text);
}
```

`LlamaMedia` accepts images (jpg/png/bmp/gif via stb_image) and audio (wav/mp3/flac via miniaudio) — both decoded inside libmtmd. Use `imageFile`/`imageBytes`/`audioFile`/`audioBytes` constructors.

### Persistence

```dart
await session.saveState('/tmp/conversation.lcdc');

// later, possibly after engine restart:
await otherSession.loadState('/tmp/conversation.lcdc');
```

The file format includes a metadata header (model identity, context params, mmproj identity, token checksum) so loading into an incompatible engine throws `LlamaStateException` with a discriminator (`modelMismatch`, `contextTooSmall`, `multimodalMismatch`, ...) instead of corrupting state.

### Context shift

```dart
session.generate(
  prompt: longPrompt,
  shiftPolicy: ContextShiftPolicy.auto,
  shift: const ContextShift(nKeep: -1),  // preserve the original prompt
);
```

When the next decode would push past `nCtx`, the engine drops the oldest non-keep tokens and slides the rest left, exactly like llama-server's `--context-shift`. Check `engine.canShift` first — recurrent and iSWA caches (Qwen3 SWA, Gemma 3 4B) report false and the policy throws.

## Public API surface

```
LlamaEngine        // worker isolate handle
EngineSession      // raw token-stream session
EngineChat         // chat-style session with message history
LlamaMedia         // image or audio attachment

ModelParams
ContextParams
SamplerParams
MultimodalParams
ContextShiftPolicy / ContextShift

GenerationEvent (sealed): TokenEvent | ShiftEvent | DoneEvent
StopReason (sealed):      StopEog | StopMaxTokens | StopUserAbort
ChatMessage / KnownChatTemplates
StateMetadata / LlamaStateException
LlamaLibrary       // load native lib

LlamaModel / LlamaContext / LlamaSession / LlamaBatch / Tokenizer / Sampler
                   // synchronous API for advanced use; LlamaEngine is the
                   // recommended entry point for app code
```

## Loading the native library

| Where | How |
|---|---|
| `dart test`, CLI, macOS dev | `LlamaEngine.spawn(libraryPath: '/path/to/libllama.dylib', ...)` |
| iOS / macOS app with xcframework | `LlamaEngine.spawnFromProcess(...)` (Xcode static-links the framework into the app binary) |
| Android with AAR / jniLibs | `LlamaEngine.spawn(libraryPath: 'libllama.so', ...)` (basename — Android resolves) |

`mtmd` resolution mirrors the same logic — opened by basename if `libllama` was a basename, by sibling path otherwise.

## Layout

```
lib/
  llama_cpp_dart.dart          // public exports
  src/
    ffi/         bindings.dart, library_loader.dart, log.dart
    model/       LlamaModel + vocab + ModelParams
    context/     LlamaContext + ContextParams
    batch/       LlamaBatch
    sampling/    Sampler + SamplerFactory + SamplerParams
    tokenizer/   Tokenizer + Utf8Accumulator
    generation/  Generator + Request + GenerationEvent + ShiftPolicy
    chat/        ChatMessage + ChatTemplate + KnownChatTemplates
    multimodal/  MultimodalContext + LlamaMedia + MultimodalParams
    session/     LlamaSession + StateCodec
    isolate/     LlamaEngine + EngineSession + EngineChat + worker
    types/       exception hierarchy

tool/                          // build scripts (macOS dylib, Apple xcframework, Android AAR)
example/probes/                // runnable Dart scripts demonstrating each subsystem
test/                          // pure-Dart and integration tests
plan.md                        // milestone-by-milestone roadmap
```

## Versioning

`0.9.x` is the rewrite line. The Dart API is mostly stable but **may break once more** before 1.0 — most likely around: real Jinja support, on-device validation findings, and final naming for chat-template/policy knobs. Pin to a minor when you ship.

llama.cpp is pinned per release in `src/llama.cpp` (git submodule). Bumps are tested against the full suite before tagging.

## License

MIT.
