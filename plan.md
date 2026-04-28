# llama_cpp_dart — Rewrite Plan

Status: draft. Authoritative scope for the v1.0 rewrite. Breaking API expected.
Read this plan before implementation starts; it resolves the packaging, Android backend, and API-shape decisions below.

## 1. Goals

- Dart-only FFI package intended for Flutter/mobile apps, targeting **iOS and Android** as first-class.
- Clean, modular FFI wrapper. No god classes. No experimental side projects.
- Vision/multimodal is a core feature, not a bolt-on.
- Bindings in sync with current llama.cpp APIs (batch, sampler chain, vocab, mtmd).
- The pub package contains Dart code only. Native binaries are published separately on GitHub Releases; users download and integrate the right binary bundle themselves.
- Maintain macOS native builds for fast local development and integration testing. macOS is a maintainer/test target, not a production support target.

## 2. Non-goals

- No standalone Dart/CLI/server target. The `bin/` entrypoints are gone.
- No production desktop targets. macOS is built only to speed up maintainer testing; Linux and Windows are not tested or built.
- No CUDA, OpenCL, or Vulkan builds. Android Vulkan is explicitly out.
- No MCP. `lib/src/mcp/` is removed. Agent/tool plumbing is the app's problem.
- No HTTP server, no OpenAI-compatible surface.
- No RAG helpers (`TextChunker`) unless they fall out naturally.
- No package-bundled binaries, pub post-install hooks, or automatic binary downloads.

## 3. Reference implementation

The Rust project at `/Users/adel/Workspace/hugind` is the canonical solved-form for these problems. Its `src/server/llm/` and `src/server/engine/` modules show the decomposition we want — module-level separation of `model`, `context`, `batch`, `sampling`, `tokenizer`, `multimodal`, `chat`, `generation`, `session`. Mirror that decomposition in Dart, scaled down to single-app mobile.

What hugind solved that the current Dart project still hasn't:
- Tokenizer as its own type, not a method on the model.
- Sampler is per-request, not a singleton on the wrapper.
- `Request` is a value type with an explicit state machine.
- Multimodal lives behind `MultimodalContext`, not interleaved with text generation.
- KV cache tiering is a dedicated manager, not an `if`-branch in the wrapper.

Differences for the Dart/mobile rewrite:
- No continuous-batching scheduler with N concurrent slots — mobile apps rarely need it. Single active session, optional swap-to-disk for the rest.
- Use Dart isolates where hugind uses OS threads. One inference isolate per engine.
- VRAM tier is irrelevant on mobile (unified memory). KV tiering collapses to RAM/disk.

## 4. Target platforms & native distribution

| Platform | Backend | Distribution |
|----------|---------|--------------|
| macOS | Metal | GitHub Release `.dylib` bundle. Primary maintainer/dev target for fast iteration. |
| iOS | Metal | GitHub Release `xcframework` bundle (device + simulator). Users embed/codesign at app build time. |
| Android | CPU + Hexagon NPU | GitHub Release bundle with `libllama.so`, `libggml-cpu.so`, `libggml-hexagon.so`, and `libggml-htp-v73/v75/v79/v81.so`. Backend is selected at runtime. |

- Native binaries are not committed to this repo and are not shipped inside the pub package.
- Users download binaries from the GitHub Release page and embed them in their app. iOS/macOS bundles are embedded at build time because runtime code download/loading is not allowed by the platforms.
- Release artifacts include a manifest with llama.cpp commit, backend, platform, ABI, checksum, and build flags.
- One ffigen output (`lib/src/ffi/bindings.dart`) covers every platform — the Hexagon backend is upstream-merged and exposes no NPU-specific public C API.

Android backend decisions:
- All Vulkan and OpenCL Android paths are removed. They are not fallbacks.
- The Android bundle ships **CPU + Hexagon together**. Backend choice is runtime, via `ggml-backend` device selection (`--device` analogue), not a build-time split.
- Hexagon is upstream-mainline (`src/llama.cpp/ggml/src/ggml-hexagon/`, build flag `-DGGML_HEXAGON=ON`). The earlier `haozixu/llama.cpp-npu` research fork is **not used**.
- Hexagon HTP variants `v73`/`v75`/`v79`/`v81` cover Snapdragon 8 Gen 2 through 8 Elite. All four ship in one bundle; the runtime picks the correct one for the device.
- **No model conversion required.** Standard GGUF files load on Hexagon via the runtime REPACK buffer mechanism. The same `.gguf` runs on CPU, Metal, or Hexagon.
- Multimodal (mtmd) is supported on Hexagon since it's part of upstream llama.cpp.
- Android build uses the official `ghcr.io/snapdragon-toolchain/arm64-android` Docker image (Android NDK + Hexagon SDK 6.4 preinstalled). No self-hosted CI runner needed.

Removed from the repo:
- `darwin/`, `ubuntu/`, `linux/`, `windows/` build scripts and their CMake glue
- `bin/MAC_ARM64/`, `bin/OS64/`, `bin/SIMULATOR64/`, `bin/SIMULATORARM64/` ad-hoc binary dumps
- `src/OpenCL-Headers/`, `src/Vulkan-Headers-main/`, `src/opencl-libs/`, `src/miniaudio/`, `src/stb/`
- Flutter plugin native vendoring/build directories that only exist to ship binaries through the package. Keep maintainer build scripts under `tool/`.

## 5. Architecture

```
lib/
  llama_cpp_dart.dart            — public API surface (small)
  src/
    ffi/
      bindings.dart              — ffigen output (was core/llama_cpp.dart)
      library_loader.dart        — external binary loading for iOS/Android/macOS
    model/
      model.dart                 — Model handle + load/unload
      model_params.dart          — declarative DTO
      vocab.dart                 — vocab pointer cache + special tokens
    context/
      context.dart               — Context handle
      context_params.dart        — declarative DTO
      kv_cache.dart              — kv shift / clear / seq ops
    batch/
      batch.dart                 — modern llama_batch wrapper (single-token + multi-seq)
    sampling/
      sampler.dart               — per-request sampler chain
      sampler_params.dart        — declarative DTO
      sampler_factory.dart       — kept ~as-is, lifted from src/core/service/
    tokenizer/
      tokenizer.dart              — encode/decode, special tokens, UTF-8 streaming
    multimodal/
      mtmd_context.dart          — owns mtmd ctx
      multimodal_params.dart     — mmproj path + mtmd settings
      image.dart                 — LlamaImage (kept)
      vision_helper.dart         — kept, moved here
    generation/
      request.dart               — Request DTO + state machine
      generator.dart             — async stream of tokens for one request
      stop.dart                  — stop conditions, EOS handling
    session/
      session.dart               — one conversation: tokens + KV head + tier
      session_store.dart         — RAM/disk tiering (was KvCacheManager)
      state_codec.dart           — kept, moved here
    chat/
      chat_message.dart          — {role, content} DTO
      chat_template.dart         — wraps llama_chat_apply_template (model's embedded Jinja)
      chat_session.dart          — message history + render-on-each-turn + generate
    isolate/
      engine_isolate.dart        — single inference isolate (replaces parent/child/scope trio)
      messages.dart              — typed commands/events
```

Module rules:
- Each module owns one llama.cpp concept. No cross-cutting state.
- `Model` does not know about `Context`. `Context` does not know about `Session`. Coupling flows one direction: `session → context → model → ffi`.
- All FFI pointers are wrapped in finalizable handle objects. No raw `Pointer<T>` escapes a module.
- DTOs (`*Params`) are immutable, JSON-serializable, no FFI types, no native allocation ownership.
- FFI conversion happens in module-local builders that own temporary native memory for the duration of the call.

## 6. Public API surface (target shape)

```dart
// native library loading
await LlamaLibrary.load(path: '/path/to/release/libllama.dylib');

// loading
final model = await LlamaModel.load(ModelParams(path: '/path/to/model.gguf'));
final ctx   = await model.createContext(ContextParams(nCtx: 4096));

// chat session
final session = LlamaSession(ctx, format: ChatMLFormat());
session.addSystem('You are helpful.');
session.addUser('Hi');

// streaming generation
await for (final token in session.generate(SamplerParams(temperature: 0.7))) {
  stdout.write(token.text);
}

// vision
final mm = await model.createMultimodalContext(
  MultimodalParams(mmprojPath: '/path/to/mmproj.gguf'),
);
final visionSession = LlamaSession(ctx, format: ChatMLFormat(), multimodal: mm);
visionSession.addUser('What is in this image?', images: [LlamaImage.fromFile(path)]);

// state
final blob = await session.saveState();
await session.loadState(blob);

// disposal
await session.dispose();
await ctx.dispose();
await model.dispose();
```

Off-isolate variant (Flutter UI thread):
```dart
final engine = await LlamaEngine.spawn(
  library: LibraryParams(path: '/path/to/release/libllama.so'),
  model: ModelParams(path: '/path/to/model.gguf'),
);
final session = engine.createSession(format: ChatMLFormat());
await for (final t in session.generate(prompt: '...')) { ... }
await engine.dispose();
```

That is the entire user-visible API. No `Llama`, no `LlamaParent`/`LlamaChild`/`IsolateScope`, no `LlamaService`.
`LlamaEngine` is the primary Flutter-facing API. The direct synchronous API exists for tests and advanced use, but it must not be the path encouraged in README examples.

## 7. What's kept (refactored, not rewritten)

- `SamplerFactory` — clean today, just relocate to `src/sampling/`.
- `ServiceSession` UTF-8 accumulator + replay buffer — relocate to `src/tokenizer/` and `src/session/`.
- `ModelCache` — keep for hot-reload during dev; expose as `LlamaModel.shared(...)`.
- `VisionHelper` — relocate to `src/multimodal/`.
- `state_codec.dart` — relocate to `src/session/` and extend with metadata.
- All `*Params` DTOs — keep semantic coverage, but rewrite as immutable values; prune fields no longer in llama.cpp and remove native allocation state.
- Prompt format classes — **all five removed**. Modern llama.cpp embeds the Jinja template in the GGUF and exposes `llama_model_chat_template` + `llama_chat_apply_template`. The wrapper just passes messages through; the model decides the format. ChatML/Gemma/Alpaca/Harmony/ChatML-thinking handled identically. `<think>` tag emission is part of recent templates' Jinja and rendered automatically when the user sets the right flag (some templates honor `enable_thinking`).

## 8. What's removed

- `lib/src/mcp/` (entire folder)
- `lib/src/utils/text_chunker.dart` and its test
- `lib/src/core/llama.dart` (fully replaced by the new module split)
- `lib/src/core/llama_service.dart` (mobile doesn't need the multi-session scheduler)
- `lib/src/isolate/{isolate_parent,isolate_child,isolate_scope,isolate_types}.dart` (replaced by `engine_isolate.dart`)
- `lib/src/prompt/{alpaca_format,chatml_format,chatml_thinking_format,gemma_format,harmony_format,prompt_format}.dart` (all five superseded by `llama_chat_apply_template`)
- `bin/`, `darwin/`, `ubuntu/`, `linux/`, `windows/` build artifacts and scripts
- All `example/*server*`, `*mcp*`, `*ubuntu_cuda*` examples
- `one_session.bin`, `sessions/` (test artifacts in repo root)
- `src/OpenCL-Headers/`, `src/Vulkan-Headers-main/`, `src/opencl-libs/`, `src/miniaudio/`, `src/stb/` (vendored deps for removed backends)
- Pub package binary vendoring paths (`ios/*.xcframework`, `android/src/main/jniLibs/`, and equivalent plugin packaging paths). Binaries live on GitHub Releases only.

## 9. llama.cpp API modernization checklist

- [ ] Use `llama_batch_get_one` only for trivial single-sequence smoke paths. Production generation uses owned `llama_batch_init` builders so positions, seq ids, logits flags, embeddings, multimodal chunks, and restore flows stay explicit.
- [ ] Confirm `llama_model_get_vocab` is cached on `Model` construction; pass vocab pointer everywhere instead of fetching.
- [ ] Keep sampler chain API (`llama_sampler_chain_*`) — already correct.
- [ ] Move from `llama_memory_*` to `llama_kv_cache_*` once stable on upstream.
- [ ] Verify `llama_context_default_params` vs `llama_context_params_default` and follow upstream rename.
- [ ] Re-run ffigen against latest `llama.h` + `mtmd.h`. Diff and adapt.
- [ ] Validate mtmd init path matches current `mtmd_context_params` shape and image-encoding flow.
- [ ] Pin a single upstream llama.cpp commit across all platforms; record it in every release manifest.
- [ ] Replace deprecated state APIs in existing code paths (`llama_get_state_size`, `llama_copy_state_data`, `llama_set_state_data`) with current `llama_state_*` APIs.
- [ ] Add backend-device discovery to the Dart API so apps can list available backends (CPU/Metal/Hexagon-HTP-v73/...) and pick one explicitly.

## 10. Build & CI

- GitHub Actions matrix:
  - `macos-latest` — macOS Metal `.dylib` + iOS Metal `xcframework` (device + simulator).
  - `ubuntu-latest` — Android CPU + Hexagon bundle, built inside `ghcr.io/snapdragon-toolchain/arm64-android` Docker image.
- Workflow triggers on pinned llama.cpp commit bump or version tag.
- Workflow uploads binary bundles plus manifest/checksums to a GitHub Release. `pubspec.yaml` does not reference binaries.
- One-command maintainer rebuild: `tool/build_native.sh --platform macos|ios|android`.
- `tool/regen_ffi.sh` runs ffigen and fails CI if `lib/src/ffi/bindings.dart` is out of sync.
- `tool/verify_release.sh` validates a downloaded release bundle against its manifest and runs a smoke load/generate test on macOS.

## 11. Testing

- Unit tests per module (`test/model_test.dart`, `test/tokenizer_test.dart`, etc.) — no FFI required where possible (mock `Bindings`).
- macOS FFI integration test for fast local iteration, gated by `LLAMA_CPP_DART_LIB` and a tiny GGUF fixture. This is the primary inner-loop test surface.
- Integration test app under `example/` that runs on iOS simulator and Android emulator in CI, loading a tiny GGUF fixture and a downloaded release binary.
- Android CPU smoke test runs on emulator or physical device.
- Android Hexagon smoke/perf test runs on physical Snapdragon 8 Gen 2+ devices.
- Drop the current `test/` files that depend on removed modules.
- Persistence tests must reject incompatible state by checking metadata: model identity/hash, llama.cpp commit/API version, backend, context params, chat format, mmproj identity when used, token history/checksum, and wrapper state-codec version.

## 12. Phasing

The phasing is **macOS-first, then iOS, then Android**. macOS Metal is the development and integration target — every Dart-side feature lands and is validated on macOS before any mobile packaging work. Mobile platforms are pure packaging exercises against an already-stable Dart API.

### Phase 1 — macOS + full code rewrite

**M1 — Skeleton + macOS binary loader.** New `lib/src/` layout, ffigen regen against pinned llama.cpp commit, external library loader (`LlamaLibrary.load`), `Model` + `Context` + `Batch` + `Tokenizer` modules. macOS Metal `.dylib` build script under `tool/`. First smoke test: load model, tokenize, decode one batch.

**M2 — Generation core.** `Sampler`, `Request`, `Generator`, basic `Session` with in-RAM history. Streaming token API works synchronously on macOS.

**M3 — Isolate engine.** `LlamaEngine` isolate wrapper. The primary public API. All later milestones use it.

**M4 — Chat.** `ChatMessage` DTO, `ChatTemplate` thin wrapper around `llama_chat_apply_template`, `EngineChat` with role-tagged history that renders via the model's embedded Jinja template each turn. No per-format Dart code; the GGUF tells us what to do. Public API of section 6 is now complete on macOS.

**M5 — Multimodal.** `MultimodalContext`, `MultimodalParams(mmprojPath: ...)`, `LlamaImage`. Vision validated on Gemma-3 and Qwen2.5-VL on macOS.

**M6 — Persistence.** `SessionStore` with RAM+disk tiers, metadata-rich `state_codec`, save/load rejection on incompatible state, app-lifecycle hooks documented. Phase 1 ends here: full Dart API works end-to-end on macOS, no mobile packaging touched yet.

### Phase 2 — iOS

**M7 — iOS xcframework.** Two-slice iOS xcframework: `ios-arm64` (device) + `ios-arm64-simulator` (Apple Silicon Mac). x86_64 simulator skipped — Apple Silicon-only dev assumption; can add later. macOS is NOT bundled (the dev dylib at `build/macos/` stays separate). Metal on by default with `GGML_METAL_EMBED_LIBRARY=ON` so shader source is baked into the binary. Embed-at-build-time documentation. Run the example app on iOS Simulator and a physical device. Sanity-check that no Dart code changes are needed beyond the library loader path.

### Phase 3 — Android

**M8 — Android CPU AAR.** Build with the local Android NDK against `arm64-v8a`. Package as a standard `.aar` with `libllama.so` + `libggml{,-base,-cpu}.so` + `libmtmd.so` under `jni/arm64-v8a/`. CPU only — works on every modern Android device. Ship via `pubspec.yaml` `flutter.plugin.platforms.android` or as a downloadable artifact.

**M8.5 — Android Hexagon NPU (post-1.0).** Add Hexagon backend via `-DGGML_HEXAGON=ON`. Requires the Hexagon SDK (separate Qualcomm download — QAIRT/QNN provides the runtime libs but **not** the DSP compiler toolchain). Build via the `ghcr.io/snapdragon-toolchain/arm64-android` Docker image which bundles NDK + Hexagon SDK + OpenCL. Output adds `libggml-hexagon.so` + `libggml-htp-v73/v75/v79/v81.so` to the AAR; runtime backend selection picks NPU on Snapdragon 8 Gen 2+, falls back to CPU elsewhere. Defer until Hexagon SDK is properly installed and a Snapdragon device is available for validation.

### Phase 4 — Ship

**M9 — Release.** GitHub Actions workflow (macOS + Linux/Docker matrix), release bundles per platform with manifests, README install instructions, migration guide from 0.x. Tag 0.9.0 — leaves headroom for breaking-API adjustments before a real 1.0.

## 13. Open questions

- **Concurrent sessions on mobile:** is one active session + N swapped-to-disk enough, or does any real app need true parallel decode? Default answer: enough.
- **Thinking tokens:** handled by the model's Jinja template via `enable_thinking` arg or equivalent — no Dart-side string surgery needed. Resolved.
- **Custom Jinja chat templates (Gemma-4-Unsloth, etc.):** `llama_chat_apply_template` is a substring-pattern matcher across ~55 known families, NOT a Jinja parser. Models with bespoke templates (macros, tool-call helpers, custom turn markers like Unsloth's `<|turn>...<turn|>`) fail to match. Three workarounds available today: (1) `KnownChatTemplates.X` constants pass a sentinel substring that selects a recognized renderer (works for mainline Gemma, ChatML, Llama3); (2) manual prompt rendering via `EngineSession.generate(prompt:)` with the model's actual turn markers (works for everything, demonstrated with Gemma-4-Unsloth); (3) future: ship a minimal Jinja subset in Dart or a C shim around `common_chat_templates_apply`. Option 3 is post-1.0 unless real users block on it.
- **Context-shift policy on multimodal sessions (hugind's pattern):** llama-server hard-disables ctx-shift when mmproj is loaded; hugind takes a smarter approach — allows shift but raises `n_keep` to cover the entire multimodal prompt prefix (`mm_keep_floor = pos_offset + prompt_tokens.len()`), so the shift only discards generated tokens past the multimodal prefix. We don't do shift in our multimodal path today; adopt hugind's pattern in M-future when long multimodal sessions become a real use case.
- **Old API parity:** do we ship a thin `legacy.dart` shim for 0.x users, or just publish a migration guide and break cleanly? Default answer: clean break.
- **Hexagon device selection UX:** should the Dart API auto-pick the highest-scoring backend by default, or always require an explicit choice from `LlamaBackend.list()`? Default answer: auto-pick with override.
- **Android packaging of `libggml-htp-v*.so`:** confirm Play Store delivery accepts the four-variant bundle (no per-device APK splitting needed since all four are small and ABI-compatible).
