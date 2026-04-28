# Changelog

## 0.9.0-dev.0 — full rewrite

This is effectively a new package. The 0.2.x line was a single-class FFI
binding glued to a multi-target build system (server, CLI, MCP, multiple
backends). 0.9.0 throws all of that away and rebuilds around three things:
**Flutter mobile**, **modular FFI**, and **off-thread inference**.

If you were on 0.2.x, see [MIGRATION.md](MIGRATION.md). The public API does
not preserve names from 0.2.

### What's new

- `LlamaEngine` worker isolate — primary public API. Streaming token output
  via `Stream<GenerationEvent>` (sealed: `TokenEvent` | `ShiftEvent` |
  `DoneEvent`). Cancellation via stream subscription cancel.
- `EngineSession` (raw prompt) and `EngineChat` (message-history with chat
  template) on top of the engine isolate.
- Multimodal (vision + audio) via llama.cpp's `mtmd` — image and audio
  bytes go to the model unmodified; libmtmd handles decoding.
- Persistence: `EngineSession.saveState/loadState` and
  `EngineChat.saveState/loadState` with metadata-validated reload
  (`StateMetadata`, `LlamaStateException` with discriminator).
- `llama-server`-style context shift (`ContextShiftPolicy.auto`) gated on
  `engine.canShift` — falls back gracefully on iSWA / recurrent caches.
- Three platform artifacts:
  - macOS dylib for `dart test` (`tool/build_native.sh`)
  - Apple xcframework with `ios-arm64` + `ios-arm64-simulator` +
    `macos-arm64` slices (`tool/build_apple_xcframework.sh`)
  - Android AAR for `arm64-v8a`, CPU + mtmd (`tool/build_android_aar.sh`)
- `LlamaLibrary.load(path:)` for dylib loading and
  `LlamaLibrary.loadFromProcess()` for static-linked iOS/macOS apps.

### What's gone (vs 0.2.x)

- The `Llama` god-class.
- `LlamaParent` / `LlamaChild` / `IsolateScope` (replaced by
  `LlamaEngine`).
- `LlamaService` — the multi-session scheduler. Mobile apps do one
  conversation at a time; multi-session can be added back as a higher
  layer if needed.
- The MCP client / server / agent surface.
- `TextChunker` (RAG helper) — application-layer concern.
- The `lib/src/prompt/` chat-format classes (ChatML, Alpaca, Gemma,
  Harmony, ChatML-thinking). Modern llama.cpp embeds the Jinja chat
  template in the GGUF; we use it via `llama_chat_apply_template`. For
  models with custom Jinja the matcher can't parse, see
  `KnownChatTemplates` and the manual-prompt workaround in
  `example/probes/gemma_chat.dart`.
- All non-mobile platform code: Linux, Windows, CUDA, OpenCL Linux,
  Vulkan desktop. macOS is kept as a dev/test target only.
- The bundled binary distribution path inside the Dart package. Native
  artifacts ship from GitHub Releases instead.

### Known limitations

- Custom Jinja chat templates (some Unsloth quants) require manual prompt
  rendering. Real Jinja support is post-1.0.
- Hexagon NPU is not built. Needs the Hexagon SDK + Snapdragon toolchain
  Docker image. Tracked as M8.5 in `plan.md`.
- Multimodal generation does not auto-shift on context overflow (matches
  llama-server's behaviour). Long multimodal sessions need to be
  segmented at the application level.
- Log silencing is off in the worker isolate — `Pointer.fromFunction`
  callbacks crash when ggml's Metal init logs from a non-Dart thread.
  Move to `NativeCallable.isolateGroupShared` is queued.
- Cosmetic: `ggml_metal_device_free` asserts at process exit because the
  worker doesn't dispose model/context (deliberate — disposing one
  isolate's model crashes another's outstanding ops). The assert fires
  after tests pass; harmless.

---

Older entries are preserved below for context. They describe the 0.2.x
line, which has been removed.

<details>
<summary>0.2.x history</summary>

## 0.2.3
*  **Performance**: Moved image embedding storage to native memory (C heap) to reduce Dart GC pressure and improve stability with high-resolution images.
*  Fix memory leaks in session cancellation and disposal logic.

## 0.2.2
*  allow freeing the active slot by switching/detaching and reselecting a fallback
*  ensure isolate child always replies on dispose/free, even when already torn down
*  keep parent subscription alive through shutdown so free-slot confirmations are received
*  cancel scope work before freeing slots to avoid in-flight races
*  add opt-in KV auto-trim (sliding window) with example `example/auto_trim.dart`

## 0.2.1
* **Android**: Added OpenCL support for GPU acceleration (#91).
* **Vision**:
    * Fixed crash in `mtmd` context disposal.
    * Stable Qwen3-VL support.
* **Chat**: Added experimental support for Qwen3-VL chat format (`_exportQwen3Jinja`).
* **Fixes**:
    * Improved logging initialization (#88).
    * Fixed stream processing crash in chat.
* **Core**: Updated `llama.cpp` submodule.

## 0.2.0 — and earlier
See git history.

</details>
