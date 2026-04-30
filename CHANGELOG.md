# Changelog

## 0.9.0-dev.6 — option coverage pass

Closes the gap between the Dart binding's option surface and the
underlying llama.cpp params. Purely additive — existing code keeps
working with previous defaults.

### Sampling — `SamplerParams`

- `MirostatConfig` (v1 + v2 with `tau`, `eta`, `m`). Terminal sampler
  when enabled — replaces the `dist` stage.
- `GrammarConfig` — GBNF grammar plus optional lazy-trigger patterns
  and trigger tokens (`llama_sampler_init_grammar` /
  `llama_sampler_init_grammar_lazy_patterns`).
- `DryConfig` — DRY sampler (multiplier, base, allowed length, last-N,
  seq breakers).
- `XtcConfig` — XTC sampler (probability, threshold, min keep, seed).
- `DynamicTempConfig` — dynamic temperature (`temp_ext`: range,
  exponent).
- `AdaptivePConfig` — adaptive-P terminal sampler (target, decay,
  seed).
- `LogitBiasEntry` list applied at the start of the chain.
- `topNSigma`, `infill`, and shared `minKeep` for top-p / min-p /
  typical / xtc.
- `SamplerFactory.build(params, model: ...)` — `model:` is now
  required when the chain uses grammar, DRY, infill, logit-bias, or
  Mirostat v1 (anything that needs the vocab or `n_ctx_train`).

### Context — `ContextParams`

- `RopeScalingType`, `PoolingType`, `AttentionType` enums.
- `ropeFreqBase`, `ropeFreqScale`.
- YaRN: `yarnExtFactor`, `yarnAttnFactor`, `yarnBetaFast`,
  `yarnBetaSlow`, `yarnOrigCtx`.
- `defragThreshold`, `noPerf`, `opOffload`, `swaFull`, `kvUnified`.

### Model — `ModelParams`

- `SplitMode` enum + `mainGpu` + `tensorSplit` (allocated to
  `llama_max_devices()` at load time).
- `devices` — list of backend device names (resolved from
  `LlamaBackends.list()`).
- `kvOverrides` — int / float / bool / string GGUF metadata overrides
  with the standard NULL-terminated array layout.
- `useDirectIo`, `useExtraBufts`, `noHost`, `noAlloc`.

### Docs

- README points at the [`aichat`](https://github.com/netdur/imaged-sdk-examples/tree/main/aichat)
  sample app as a working Flutter integration reference.

### Deferred (tracked for a later cycle)

- `progress_callback`, `cb_eval`, `abort_callback` — need a
  `NativeCallable.listener` wrapper with isolate-affinity rules.
- `tensor_buft_overrides` — needs a per-device buffer-type accessor in
  `BackendDevice` first.
- Backend sampler chain (`llama_context_params.samplers`) — still
  marked `[EXPERIMENTAL]` upstream.

## 0.9.0-dev.5 — first pub.dev publish of the rewrite

Consolidates `0.9.0-dev.0` through `0.9.0-dev.5` (none of dev.0–dev.4
were published to pub.dev). The 0.2.x line is a separate package shape
— see [MIGRATION.md](MIGRATION.md).

### Highlights since 0.2.x

- `LlamaEngine` worker isolate is the primary public API. Streaming
  token output via `Stream<GenerationEvent>` (sealed:
  `TokenEvent` | `ShiftEvent` | `DoneEvent`). Cancellation via stream
  subscription cancel.
- `EngineSession` (raw prompt) and `EngineChat` (message-history with
  chat template) on top of the engine isolate.
- Multimodal (vision + audio) via llama.cpp's `mtmd`.
- Persistence: `EngineSession.saveState/loadState` and
  `EngineChat.saveState/loadState` with metadata-validated reload.
- `llama-server`-style context shift (`ContextShiftPolicy.auto`) gated
  on `engine.canShift`.
- Three platform artifacts shipped from GitHub Releases:
  - macOS dylib (for `dart test`)
  - Apple xcframework (`ios-arm64`, `ios-arm64-simulator`, `macos-arm64`)
  - Android AAR for `arm64-v8a`, two flavors: CPU+mtmd (~2 MB) and
    Hexagon NPU + OpenCL + mtmd (~3.7 MB)

### Validated end-to-end on real devices

- Galaxy S23 Ultra (Snapdragon 8 Gen 2, Android 14) — Hexagon NPU
  reachable from a third-party Flutter app on commercial firmware.
- Galaxy Fold7 (Snapdragon 8 Elite, Android 16) — same APK runs unchanged.
- MacBook Pro M1 Max (macOS 26) — Metal via dylib path.
- iPad M1 (iOS 26) — Metal + Accelerate BLAS via the bundled
  CocoaPods `llama_cpp.podspec`.

### Bindings

- **Backend inspection.** `engine.devices` (`List<BackendDevice>`),
  `engine.hasAccelerator`, `engine.primaryAcceleratorName`, and the
  pre-engine `LlamaBackends.list()`. Tells you which backends loaded
  on the current device.
- **`primaryAcceleratorName` priority** orders by registry name (HTP →
  Hexagon → Metal → CUDA → Vulkan) before type, so Snapdragon HTP wins
  over OpenCL even when ggml reports both as `type=gpu`.
- **KV-cache quantization.** `ContextParams.typeK` / `typeV` accept any
  of `KvCacheType.{f32, f16, bf16, q8_0, q4_0, q4_1, q5_0, q5_1}`.
  `q8_0` halves KV memory at small quality cost; useful on 8 GB Android
  devices with longer contexts.
- **Stderr capture.** `LlamaLog.captureToFile(path)` /
  `LlamaLog.restoreStderr()`. Toggleable redirect of llama.cpp/ggml
  log lines for Android, where stderr is not connected to logcat.
- **Auto `ADSP_LIBRARY_PATH`.** `LlamaLibrary.load()` reads
  `/proc/self/maps` on Android and exports `ADSP_LIBRARY_PATH` so
  FastRPC finds `libggml-htp-v*.so` skeleton libs without app-side
  `MethodChannel` plumbing.
- **`LlamaBindings`** is now exported. Lets callers using the raw FFI
  surface type variables / pass them around without reaching into
  `src/`.
- **`LlamaVersion`** is generated at build time. Exposes the package
  version, the llama.cpp submodule SHA + author date, and a runtime
  `systemInfo()` wrapper around `llama_print_system_info()` (e.g.
  `MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ACCELERATE = 1 | ...`).

### Removed since 0.2.x

- The `Llama` god-class.
- `LlamaParent` / `LlamaChild` / `IsolateScope` (replaced by `LlamaEngine`).
- `LlamaService` multi-session scheduler. Mobile apps do one
  conversation at a time; multi-session can be added back as a higher
  layer if needed.
- The MCP client / server / agent surface.
- `TextChunker` (RAG helper).
- Hand-written chat-format classes (ChatML, Alpaca, Gemma, Harmony).
  Modern llama.cpp embeds Jinja templates in the GGUF; we use
  `llama_chat_apply_template` instead.
- All non-mobile platform code: Linux, Windows, CUDA, Vulkan desktop.
  macOS is kept as a dev/test target.
- Bundled binary distribution. Native artifacts ship from GitHub
  Releases instead.

### Known limitations

- Custom Jinja chat templates (some Unsloth quants) require manual
  prompt rendering. Real Jinja support is post-1.0.
- HTP only engages Q4_0 / Q8_0 quants in upstream ggml-hexagon.
  K-quants (`Q4_K_*`, `Q5_K_*`) and I-quants (`IQ*`) run on OpenCL+CPU.
- HTP REPACK budget is ~2 GB per session; ≥7B-class models need a
  multi-session pattern not yet exposed by the binding.
- Multimodal generation does not auto-shift on context overflow
  (matches `llama-server`'s behaviour).
- Cosmetic: `ggml_metal_device_free` asserts at process exit because
  the worker doesn't dispose model/context. Harmless.

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
- Hexagon NPU AAR is built (`tool/build_android_hexagon_aar.sh`,
  using `ghcr.io/snapdragon-toolchain/arm64-android:v0.3`) but **not
  yet validated on a physical Snapdragon device**. The AAR ships
  six HTP DSP variants (v68/v69/v73/v75/v79/v81) covering Snapdragon
  865 → 8 Elite + future. Total ~3.7 MB stripped.
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
