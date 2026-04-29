# llama_cpp_dart — Roadmap & Progress

Live tracking doc for the 0.9.x rewrite line. Updated as work lands.

> **Current state**: 0.9.0-dev.1 tagged. Apple xcframework, macOS dylib,
> Android CPU AAR, and Android Hexagon AAR all build green in CI and
> attach to GitHub Releases. **No on-device validation yet** — that's
> the next gate.

---

## Status dashboard

| ID | Milestone | Status | Notes |
|---|---|---|---|
| M1 | Skeleton + macOS dylib loader | ✅ shipped | 4 modules, smoke test |
| M2 | Generation core | ✅ shipped | Sampler/Request/Generator/Session, streaming token API |
| M3 | Isolate engine | ✅ shipped | LlamaEngine + EngineSession; the public API |
| M4 | Chat (template-driven) | ✅ shipped | EngineChat, ChatTemplate, KnownChatTemplates |
| M5 | Multimodal (vision + audio) | ✅ shipped | LlamaMedia, MultimodalContext, mtmd_helper_eval_chunks |
| M6 | Persistence | ✅ shipped | StateCodec with metadata-validated reload |
| M6.5 | Context shift | ✅ shipped | llama-server-style auto-shift, gated on canShift |
| M7 | iOS xcframework | ✅ shipped | 3 slices: ios-arm64, ios-arm64-simulator, macos-arm64 |
| M8 | Android CPU AAR | ✅ shipped | arm64-v8a, ~2 MB stripped |
| M8.5 | Android Hexagon AAR | ⚠️ built but not validated | 6 HTP variants v68→v81, ~3.7 MB |
| M9 | Release pipeline | ✅ shipped | CI green; v0.9.0-dev.1 attached artifacts |
| **M10** | **On-device validation** | ⚠️ in progress | Android-Hexagon proven on Galaxy S23 Ultra; iOS + vision smoke + binding-gap fixes still pending |
| M11 | Real Jinja chat templates | 🟡 deferred | needed for Gemma-4-Unsloth and similar |
| M12 | pub.dev publish | 🟡 deferred | needs `dart pub publish` from CI |

Status icons: ✅ done · ⚠️ done with caveat · ⏳ in progress · 🟡 deferred · ❌ blocked.

---

## What's left before 1.0

### Hard gates

1. **On-device validation (M10).** Everything we ship for iOS / Android is unverified on real hardware. Likely surprises:
   - Library-load path edge cases on iOS (xcframework + `loadFromProcess`).
   - Metal shader compile time on first-run iPhones.
   - Hexagon HTP runtime variant selection (which `libggml-htp-v*.so` actually loads on a given Snapdragon).
   - Android model file path access (assets vs. external storage).
2. **Real Jinja for chat templates (M11).** `llama_chat_apply_template` substring matching covers ~55 families but fails on Gemma-4-Unsloth and similar custom Jinja. Today's workaround is manual `EngineSession.generate(prompt:)` formatting — works but `EngineChat.addUser` etc. silently break for those models. Two implementation paths:
   - Ship a minimal Jinja subset in Dart (~500 lines).
   - C shim around `common_chat_templates_apply` (links libllama-common into the AAR/xcframework).

### Soft gaps (worth doing in 0.9.x)

3. **pub.dev publishing (M12).** Add `dart pub publish --dry-run` to CI on tag, then a manual confirm.
4. **Stop strings.** Token-level stop only today (`StopEog` / `StopMaxTokens`). Real chat apps want `stopAt: ['<|user|>', 'User:']`.
5. **Thread-safe log silencing in worker.** `LlamaLog.silence()` is off in the worker because `Pointer.fromFunction` callbacks crash when ggml's Metal init logs from a non-Dart thread. Switch to `NativeCallable.isolateGroupShared`.
6. **Multi-image / multi-audio per turn.** Implementation supports it (media is a list); no probe or test exercises it.

### Nice-to-have (post-1.0)

7. **Concurrent decode per engine.** One in-flight generate per worker today.
8. **Benchmarking probe.** No tok/s harness for tuning.
9. **Cosmetic Metal residency assert at process exit.** Annoying but harmless; needs a refcounted backend init.
10. **Context shift on multimodal turns.** llama-server disables; hugind raises `n_keep` to cover the multimodal prefix instead — adopt that when a real long-multimodal use case appears.

---

## Decisions that landed

These were called out as goals/non-goals at the start of the rewrite. Recording them so we don't re-litigate.

- **Target platforms**: iOS + Android primary, macOS as dev/test target. Linux / Windows / desktop CUDA / Vulkan-desktop dropped.
- **Distribution**: Dart package contains no binaries. Native artifacts ship from GitHub Releases (`.dylib`, xcframework, `.aar`). Users embed at app build time.
- **Acceleration**: Apple Metal, Android Hexagon NPU + OpenCL + CPU. No Vulkan on Android. Hexagon is upstream-merged (`ggml-hexagon`); we don't use the haozixu fork.
- **Single ffigen output** covers every platform. mtmd-helper.h included for image/audio file decoding.
- **Architecture**: modular FFI. No god-class. Coupling flows one direction: `session → context → model → ffi`. Each `*Params` DTO is immutable, JSON-serializable, no FFI types.
- **Chat templates**: rendered via `llama_chat_apply_template` against the GGUF's embedded Jinja. Per-format Dart classes (ChatML/Gemma/Alpaca/Harmony) all dropped. `KnownChatTemplates` provides sentinel substrings as a fallback for misclassified models.
- **No MCP, no HTTP server, no agent runtime, no `TextChunker`** — application-layer concerns.
- **Single in-flight generate per engine.** Multi-session uses sequence ids in the same context.
- **Persistence file format** carries metadata (model identity, context params, mmproj fingerprint, token checksum) so loading into an incompatible engine throws with a discriminator instead of corrupting state.
- **Context shift** is opt-in (`ContextShiftPolicy.auto`), gated on `engine.canShift`. Recurrent and iSWA caches return false and the policy throws.
- **Versioning**: `0.9.x` is the rewrite line. Public API may break once more before 1.0.

## M10 device validation — first pass (Galaxy S23 Ultra, SD 8 Gen 2, Android 14)

Reported by the demo-app instance. Same Gemma-4-E2B-it-Q8_0 + mmproj-F16
on both AAR flavors, push-and-go from `/sdcard/Android/data/<app>/files/`.

| Test | CPU AAR | Hexagon AAR |
|---|---|---|
| Spawn (warm) | 3.7 s | 17 s |
| Decode (67-tok reply) | 7.8 tok/s | **12.2 tok/s** (1.6×) |
| `canShift` | true | true |
| `supportsVision` / `supportsAudio` | true | true |
| Reply quality (T=0.3) | identical | identical |

Conclusion: Hexagon AAR is the right default. The 1.6× is Hexagon AAR
vs CPU AAR — could be NPU, OpenCL, or both; backend attribution
requires the binding gaps below.

### Binding gaps surfaced by M10

Queued for `0.9.0-dev.2`:

1. **`ContextParams` missing `typeK` / `typeV`** — no Dart knob for KV
   cache quantization. Patch: add both fields, thread to
   `llama_context_params`.
2. **HANDOFF.md said `minSdk 26`** — Hexagon AAR ships with manifest
   `minSdkVersion 31` because OpenCL needs API 31's `<uses-native-library>`
   opt-in. Fixed in HANDOFF.md.
3. **Hexagon `libllama.so` has hard `DT_NEEDED` on `libggml-opencl.so`.**
   Apps need `<uses-native-library android:name="libOpenCL.so" android:required="false" />`
   in their manifest or `dlopen` of `libllama.so` fails on first load.
   Documented in HANDOFF.md; consider whether a soft-load path is worth
   it on the binding side.
4. ~~**No backend-inspection API.**~~ ✅ shipped. `engine.devices`
   returns the full list captured at spawn; `BackendDevice` carries
   name / description / type (cpu/gpu/igpu/accel/meta) / registry /
   memory. `engine.hasAccelerator` and `engine.primaryAcceleratorName`
   give the common-case shortcut. Probes:
   `example/probes/{list_backends,engine_backends}.dart`. Doesn't
   identify the specific device that ran a single generation
   (ggml-backend's scheduler distributes ops in one graph) but
   answers "is Hexagon loaded?" / "is OpenCL loaded?" cleanly.
5. **No log redirect.** `LlamaLog` has `silence()` / `useDefault()` but
   no `onMessage(cb)` — backend selection messages don't reach logcat.
   Patch: switch worker log to `NativeCallable.isolateGroupShared`,
   expose Dart callback.

### M10 still TODO

- Vision smoke on device (push JPEG, describe via `LlamaMedia.imageFile`).
- iOS validation (xcframework load, Metal shader cold-start time, model on disk via app bundle).
- Q4_K_M tok/s rerun once a smaller model is on hand. Expectation:
  ~22–28 tok/s on Hexagon (memory bandwidth halves vs Q8_0).

## Decisions that flipped during the rewrite

- **Context shifting** went from "skipped — too complex" → "shipped" (M6.5) once we found the llama-server algorithm is ~10 lines on top of the existing primitives.
- **Hexagon NPU** went from "post-1.0, blocked on SDK" → "shipped, pending validation" (M8.5) once we realized the upstream Snapdragon Docker image bundles the Hexagon SDK and works on Apple Silicon under Rosetta 2.
- **Three xcframework slices** (ios-arm64 + ios-arm64-simulator + macos-arm64) instead of two. macOS slice goes in the xcframework so iOS app developers can also use macOS for native debugging without a separate dylib.
- **Custom Jinja support** went from "drop the Dart prompt formats, llama.cpp handles it" → "llama.cpp's matcher misses some custom templates; manual prompt rendering is the workaround". The Jinja gap is the headline thing 0.9.x has to address.
- **Audio support** wasn't on the original radar; turned out mtmd handles audio with the same API surface as vision (LlamaMedia + mtmd_helper_eval_chunks). So we ship audio for free with vision.

---

## Open questions still in play

- **Hexagon device selection UX.** Should the Dart API auto-pick the highest-scoring backend, or always require an explicit `LlamaBackend.list()` choice? Currently auto via ggml-backend, which seems right but unverified on hardware.
- **Android packaging of `libggml-htp-v*.so`.** All six HTP variants in one AAR — works in our local build, but Play Store APK / AAB delivery hasn't been tested. May need per-device split if app stores complain about the bundle size or unused variants.
- **Real Jinja path.** Dart subset vs C shim — the trade-off depends on how much we want libllama-common bound. Wait for users to push back before deciding.

---

## Architecture (authoritative; matches code today)

```
lib/
  llama_cpp_dart.dart            — public exports
  src/
    ffi/         bindings.dart, library_loader.dart, log.dart
    model/       LlamaModel + vocab + ModelParams
    context/     LlamaContext + ContextParams
    batch/       LlamaBatch
    sampling/    Sampler + SamplerFactory + SamplerParams
    tokenizer/   Tokenizer + Utf8Accumulator
    generation/  Generator + Request + GenerationEvent + ContextShift
    chat/        ChatMessage + ChatTemplate + KnownChatTemplates
    multimodal/  MultimodalContext + LlamaMedia + MultimodalParams
    session/     LlamaSession + StateCodec
    isolate/     LlamaEngine + EngineSession + EngineChat + worker
    types/       exception hierarchy

tool/
  build_native.sh                 — macOS dylib for dev/test
  build_apple_xcframework.sh      — iOS device + iOS sim + macOS slices
  build_android_aar.sh            — CPU AAR (arm64-v8a)
  build_android_hexagon_aar.sh    — CPU + OpenCL + Hexagon NPU AAR (Docker)

example/probes/                   — runnable scripts per subsystem
test/                             — pure-Dart and integration tests
.github/workflows/                — test, build-apple, build-android
```

Module rules:
- Each module owns one llama.cpp concept. No cross-cutting state.
- All FFI pointers are wrapped in finalizable handle objects.
- DTOs (`*Params`) are immutable, JSON-serializable, no FFI types.

---

## Public API surface (current)

```dart
// dev / dart test (macOS dylib)
final engine = await LlamaEngine.spawn(
  libraryPath: '/path/to/libllama.dylib',
  modelParams: ModelParams(path: '/path/to/model.gguf', gpuLayers: 99),
  contextParams: const ContextParams(nCtx: 4096),
);

// iOS / macOS app (xcframework embedded by Xcode)
final engine = await LlamaEngine.spawnFromProcess(
  modelParams: ModelParams(path: '...'),
  contextParams: const ContextParams(nCtx: 4096),
);

// Android (AAR or jniLibs/)
final engine = await LlamaEngine.spawn(
  libraryPath: 'libllama.so',  // basename — Android's linker resolves
  modelParams: ModelParams(path: '...'),
  contextParams: const ContextParams(nCtx: 4096),
);

// raw streaming
final session = await engine.createSession();
await for (final ev in session.generate(prompt: 'Hello', maxTokens: 64)) {
  if (ev is TokenEvent) stdout.write(ev.text);
}

// chat
final chat = await engine.createChat();
chat.addSystem('You are concise.');
chat.addUser('Hi');
await for (final ev in chat.generate(maxTokens: 64)) { ... }

// multimodal (vision + audio)
final engineMM = await LlamaEngine.spawn(
  ...,
  multimodalParams: const MultimodalParams(mmprojPath: 'mmproj.gguf'),
);
chat.addUser('Describe.', media: [LlamaMedia.imageFile('cat.jpg')]);

// persistence
await session.saveState('/tmp/state.lcdc');
await otherSession.loadState('/tmp/state.lcdc');

// context shift
session.generate(
  prompt: '...',
  shiftPolicy: ContextShiftPolicy.auto,
);
```

Lower-level types (`LlamaModel`, `LlamaContext`, `LlamaSession`, `LlamaBatch`, `Tokenizer`, `Sampler`) are exported for advanced use but `LlamaEngine` is the recommended entry point.

---

## CI / build

- **`.github/workflows/test.yml`** — `dart analyze` + `dart format --set-exit-if-changed` + pure-Dart utf8 accumulator test on every push/PR.
- **`.github/workflows/build-apple.yml`** — `macos-latest`. Builds macOS dylib zip + 3-slice xcframework zip. Attaches to release on `v*` tag.
- **`.github/workflows/build-android.yml`** — `ubuntu-latest` with two parallel jobs: `cpu` (uses NDK r27c) and `hexagon` (uses Snapdragon Docker image). Both attach AAR + sha256 to release on `v*` tag.

All workflows have `permissions: contents: write` so `softprops/action-gh-release` can create / update the release.

---

## Test coverage

35 tests across 8 files, all passing on macOS:

| File | Tests | Coverage |
|---|---|---|
| `smoke_test.dart` | 4 | model load, vocab, batch, decode |
| `utf8_accumulator_test.dart` | 7 | streaming UTF-8 boundary detection (pure Dart) |
| `generation_test.dart` | 3 | Generator stream, greedy determinism, Session bookkeeping |
| `engine_test.dart` | 3 | Engine spawn, mid-stream cancel, multi-session |
| `chat_test.dart` | 4 + 1 skip | EngineChat history, multi-turn, templateOverride |
| `multimodal_test.dart` | 4 | image via session + chat, imageBytes, capability flags |
| `persistence_test.dart` | 7 | codec round-trip, mismatch detection, EngineSession + EngineChat round-trips |
| `context_shift_test.dart` | 3 | canShift, shiftContext primitive |

---

## Reference implementation

`/Users/adel/Workspace/hugind` (Rust) is the canonical solved-form for these problems. Modules mirrored in Dart (scaled down for single-app mobile):
- module-level separation: `model`, `context`, `batch`, `sampling`, `tokenizer`, `multimodal`, `chat`, `generation`, `session`
- per-request sampler (not singleton)
- `Request` as a value type with state machine
- multimodal behind its own context, not interleaved with text
- KV tiering as a dedicated manager (we use it lighter — RAM/disk only, no VRAM tier on mobile)

We **don't** mirror hugind's continuous-batching scheduler (mobile = single active session) or its multi-process server architecture.

---

## llama.cpp pin

Pinned at the b8965 era (`src/llama.cpp` submodule). Bumps require regen via `dart run ffigen` and a full test pass before retag.
