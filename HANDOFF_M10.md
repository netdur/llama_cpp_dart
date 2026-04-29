# Handoff to next Claude on `llama_cpp_dart`: M10 device-validation outcome and what to finalize

> **Date written:** 2026-04-29
> **Author of this note:** the Claude instance that ran M10 on a real device
> **Audience:** the next Claude instance picking up `llama_cpp_dart` to finalize the AAR / binding code before `0.9.0-dev.2` (or a stable `0.9.0`).

You've inherited a partially-validated 0.9.0-dev.1. This note tells you exactly what we proved on real hardware, what works, what's missing, and what to ship next. Don't re-run the investigations — they're done. Spend your time on the binding-side fixes listed at the bottom.

---

## TL;DR

- **`llama_cpp_dart 0.9.0-dev.1` works end-to-end on real devices on four platforms.** Validated on a **Galaxy S23 Ultra (SD 8 Gen 2, Android 14)**, **Galaxy Fold7 (SD 8 Elite, Android 16)**, **MacBook (Apple M1 Max, macOS 26.3)**, and **iPad M1 (iOS 26.3)**, all driving the same Flutter demo app via `path:` dependency on the binding.
- **Hexagon NPU is reachable from a third-party Flutter app on commercial Samsung firmware.** This was previously assumed to be system-app-only. It is not. There is a precise recipe (below) and once it's in place, HTP delivers ~50–60% more decode tok/s than OpenCL alone, on quants where ggml-hexagon has kernels (legacy Q4_0 / Q8_0; not K-quants).
- **macOS Metal works out of the box** in dev mode (dylib path) on M1 Max — 45.4 tok/s on Gemma-4-Q4_0. The binding's `LlamaEngine.spawn(libraryPath: '/abs/path/libllama.dylib')` resolves sibling `libggml-metal.dylib` and lights up Metal automatically.
- **iOS Metal validated on iPad M1** — 18.4 tok/s on Q4_0 with `LlamaEngine.spawnFromProcess()` via xcframework integration. The xcframework's binaries are **static archives** (not dynamic), which means a `-force_load` linker flag is mandatory — the README's "drag in and Embed & Sign" instruction is **wrong as written today** and produces a runtime `dlsym(RTLD_DEFAULT, llama_backend_init): symbol not found`. A working CocoaPods podspec ([llama_cpp.podspec](llama_cpp.podspec)) is now committed to this repo and was the path validated in M10.
- **The HANDOFF.md you have today is incomplete in two places.** The Hexagon section tells the app builder to drop the AAR and add `<uses-native-library libOpenCL.so>`. That's not enough — three more ingredients are required (full recipe below). And the Apple section glosses over the static-vs-dynamic xcframework distinction, the `-force_load` requirement for iOS, sandbox / entitlements on macOS, and the Flutter Profile-xcconfig gap that breaks Pods builds. Both sections need updating.
- **`engine.devices` / `LlamaBackends.list()` works correctly on every platform.** The new backend-inspection API closed gap #4 from the earlier known-gaps list and was the single most useful diagnostic during M10. Keep it.
- **Three binding gaps remain open** and are worth closing before declaring 1.0: KV-cache quant (`typeK`/`typeV`), log-redirect-to-logcat, and a small `primaryAcceleratorName` ordering quirk. Detail at the end.

---

## Test environments (frozen)

### Android — primary

| | |
|---|---|
| Device A | Samsung Galaxy S23 Ultra (model SM-S918B), 12 GB RAM |
| SoC A | Snapdragon 8 Gen 2 / SM8550 (Hexagon arch v73, Adreno 740) |
| Android A | 14, selinux Enforcing |
| Device B | Samsung Galaxy Fold7 (model SM-F966B), 12 GB RAM |
| SoC B | Snapdragon 8 Elite / SM8750 (Hexagon arch v79, Adreno 830) |
| Android B | 16, selinux Enforcing |
| Connection | USB-3, ~33 MB/s adb push throughput |
| Binding pin | `0.9.0-dev.1` head, llama.cpp submodule at commit `5d56effde` (the URI-fallback fix is in) |
| Demo app | `imaged-sdk-examples/aichat`, Flutter 3.x, Kotlin DSL Gradle 8.14, AGP 8.11.1, NDK r29 |

The same APK build with the same recipe ran cleanly on both phones — the recipe is portable across Snapdragon generations. Hex AAR auto-selected `libggml-htp-v73.so` on S23U and `libggml-htp-v79.so` on Fold7 with no app-level changes.

### macOS — dev/test (Path A, loose dylib)

| | |
|---|---|
| Machine | MacBook Pro 14", Apple M1 Max, 32 GB unified memory |
| OS | macOS 26.3 (darwin-arm64) |
| Native artifact | `build/macos/bin/libllama.dylib` (loaded by absolute path) |
| Acceleration | Metal — `MTLGPUFamilyApple7` / `MTLGPUFamilyMetal4`, recommendedMaxWorkingSetSize 26800 MB |
| Sandbox | **disabled** in `DebugProfile.entitlements` so the dev tree (`/Users/adel/Workspace/...`) is readable |

### iOS — shippable (Path B, xcframework via CocoaPods)

| | |
|---|---|
| Device | iPad (4th gen iPad Pro 11", model 00008103) |
| SoC | Apple M1 (8-core GPU), 8 GB unified memory |
| OS | iOS 26.3.1, USB development connection |
| Native artifact | `build/apple/llama.xcframework` (static, all 3 slices) |
| Integration | CocoaPods — `pod 'llama_cpp', :path => '...'` referencing [llama_cpp.podspec](llama_cpp.podspec) |
| Linker setting | `-force_load $(PODS_XCFRAMEWORKS_BUILD_DIR)/llama_cpp/llama.framework/llama` (in podspec's `user_target_xcconfig` and `pod_target_xcconfig`) |
| Acceleration | Metal (8-core Apple M1 GPU) + Accelerate BLAS (iOS-only backend) + CPU |
| Code signing | Free Apple ID dev profile (3-app cap — see gotcha below) |

---

## The Hexagon recipe — write this into `HANDOFF.md`

To make the Hexagon AAR's HTP backend actually register inside a Flutter Android app, **all five** of the following must be true. Missing any one of them and the HTP backend silently disappears from `engine.devices`:

### 1. Hexagon AAR + minSdk 31 + arm64-v8a-only Gradle

`android/app/build.gradle.kts`:

```kotlin
android {
    defaultConfig {
        minSdk = 31                  // Hexagon AAR's manifest requires API 31+
        ndk { abiFilters += listOf("arm64-v8a") }
    }
}
dependencies {
    implementation(files("libs/llama-cpp-dart-hexagon.aar"))
    // NB: `flatDir { dirs("libs") }` + `implementation(name=...)` does NOT work
    // reliably with Gradle 8 / AGP 8 — `files(...)` does.
}
```

The current `HANDOFF.md` says minSdk 26. That's correct only for the CPU AAR; the Hexagon AAR's bundled `AndroidManifest.xml` declares `uses-sdk minSdkVersion="31"` and Gradle's manifest merger refuses to lower it. (HANDOFF was already updated for this — keep it that way.)

### 2. `<uses-native-library>` opt-ins in `AndroidManifest.xml`

```xml
<application ... android:extractNativeLibs="true">
    <uses-native-library android:name="libOpenCL.so"  android:required="false" />
    <uses-native-library android:name="libcdsprpc.so" android:required="false" />
    ...
</application>
```

**Both** lines are mandatory if you've shipped the Hexagon AAR:

- `libOpenCL.so` — the Hexagon-built `libllama.so` has a hard `DT_NEEDED` on `libggml-opencl.so`, which itself transitively needs `libOpenCL.so` from `/vendor/lib64`. Without this declaration `LlamaLibrary.load()` throws on the very first `dlopen`.
- `libcdsprpc.so` — `ggml-hexagon` does `dlopen("libcdsprpc.so")` by basename. Even though this lib lives in `/vendor/etc/public.libraries.txt` and is technically reachable, Android's per-app linker namespace **only** grants access if the app opted in via this manifest tag. Skip it and the HTP backend silently fails to register (it logs "failed to load libcdsprpc.so: not found" but llama.cpp's stderr is invisible on Android — see gap #2 below).

The current `HANDOFF.md` mentions `libOpenCL.so` only. **Add `libcdsprpc.so`.** This is the single most important documentation fix.

### 3. `android:extractNativeLibs="true"`

```xml
<application ... android:extractNativeLibs="true">
```

Modern AGP defaults to `extractNativeLibs="false"` — `.so` files stay inside the APK and are mmaped from there at runtime by the app's own linker. That works for the app's own `dlopen`s, but **not** for FastRPC. The cDSP-side skeleton libs (`libggml-htp-v73.so` etc.) are loaded by `cdsprpcd` (a separate system process) which does a `fopen()` — and you cannot `fopen()` a path inside an APK.

With `extractNativeLibs="true"`, Android extracts every `.so` to `applicationInfo.nativeLibraryDir` at install time, where `cdsprpcd` can read them. This is required for HTP. Without it the app sees:

```
W cdsprpcd: apps_std_fopen_with_env failed with 0x2 for libggml-htp-v73.so (No such file or directory)
E ggml-hex: remote_handle_open_domain: dynamic loading failed
```

(Side effect: APK size grows from ~25 MB to ~12 MB compressed-on-disk, but ~25 MB extracted — net wash. Not worth optimising.)

### 4. `setenv("ADSP_LIBRARY_PATH", nativeLibraryDir)` from the app, *before* `LlamaLibrary.load`

Hexagon SDK's FastRPC layer searches `ADSP_LIBRARY_PATH` to find the DSP-side skeleton libs. The wrapper scripts `scripts/snapdragon/adb/run-*.sh` in upstream llama.cpp set this for shell-mode runs — for an Android app you need to set it via Android NDK from Dart:

```dart
import 'dart:ffi';
import 'package:ffi/ffi.dart';

typedef _SetenvNative = Int32 Function(Pointer<Utf8>, Pointer<Utf8>, Int32);
typedef _SetenvDart = int Function(Pointer<Utf8>, Pointer<Utf8>, int);

int _setenv(String key, String value) {
  final fn = DynamicLibrary.process()
      .lookupFunction<_SetenvNative, _SetenvDart>('setenv');
  final k = key.toNativeUtf8();
  final v = value.toNativeUtf8();
  try { return fn(k, v, 1); } finally { malloc.free(k); malloc.free(v); }
}

// In the app's startup, before any LlamaEngine.spawn call:
final nativeLibDir = await _channel.invokeMethod<String>('nativeLibraryDir');
_setenv('ADSP_LIBRARY_PATH', nativeLibDir!);
```

`nativeLibraryDir` is fetched from Java/Kotlin via a one-shot `MethodChannel` (`applicationInfo.nativeLibraryDir`).

**Suggestion to the binding:** consider doing this `setenv` automatically inside `LlamaLibrary.load()` when running on Android, derived from the loaded library path. The user-facing API would be cleaner (one less landmine for app developers). The `_libraryDir` field already exists; on Android it's currently `null` because library load goes by basename, but Java-side `Context.getApplicationInfo().nativeLibraryDir` is queryable from a Flutter plugin. Worth considering whether the Dart binding should ship a tiny native android plugin that auto-handles this — or document the recipe loudly and let app devs do it.

### 5. The llama.cpp pin must include the URI-fallback fix

The Hexagon AAR's bundled `libllama.so` must come from a llama.cpp commit that includes the [`FASTRPC_GET_URI` fallback patch](https://github.com/ggml-org/llama.cpp/issues/16911) (merged Nov 5, 2025; visible in `ggml/src/ggml-hexagon/ggml-hexagon.cpp` around line 2006: `// fallback to single session uris`).

Before this patch, on Snapdragon 8 Gen 2 specifically, the `remote_session_control(FASTRPC_GET_URI, ...)` call returns `error 0x14` ("AEE_EBADITEM"-ish) and ggml-hex aborts the whole HTP init. After the patch it logs the warning and constructs the URI manually as `file:///libggml-htp-v{arch}.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp` — which works.

The current `0.9.0-dev.1` AAR has this fix (verified). **When you bump the llama.cpp pin in `0.9.0-dev.2`, do not regress past commit `63d2fc46` without re-validating on 8 Gen 2.**

---

## The Apple recipe — write this into `HANDOFF.md` as a new section

There are **two distinct paths** for Apple platforms and the existing HANDOFF / README don't separate them clearly. App developers will get bitten if they pick the wrong one.

### Path A — Dev/test on macOS (loose dylib)

This is what we used in M10 and it works perfectly for "does Metal accelerate?" smoke tests, dart-test runs from CLI, and quick iteration.

```dart
final engine = await LlamaEngine.spawn(
  libraryPath: '/abs/path/to/libllama.dylib',
  modelParams: ModelParams(path: '/abs/path/model.gguf', gpuLayers: 99),
  contextParams: const ContextParams(nCtx: 2048),
);
```

The binding's `LlamaLibrary.load(path: '/abs/.../libllama.dylib')` correctly resolves siblings (`libggml-metal.dylib`, `libggml-base.dylib`, `libggml-cpu.dylib`, `libmtmd.dylib`) from the same directory via `_dirOf(path)` + `ggml_backend_load_all_from_path`. Metal lights up automatically on first matmul.

For a Flutter macOS app using this path you also need:

- **Disable App Sandbox in `DebugProfile.entitlements`** if the dylib or model files live outside the app bundle: set `com.apple.security.app-sandbox` to `<false/>`. Sandboxed apps cannot read arbitrary `/Users/...` paths.
- **Keep `Release.entitlements` sandboxed** for distribution — but then your Release build can't use Path A. It needs Path B.

### Path B — Shippable iOS / macOS app (xcframework, embedded)

Required for any iOS app and for any sandboxed Release macOS app. Uses the prebuilt `build/apple/llama.xcframework` from the binding repo. **Validated on iPad M1 (iOS 26) during M10 — the recipe below is what actually works, not the README's drag-into-Xcode stub.**

The xcframework's binaries are **static archives wrapped in `.framework` directories** (`file` reports `current ar archive`), not dynamic frameworks. This matters because static frameworks need `-force_load` on the consuming target — without it, the linker drops every llama symbol since nothing in `AppDelegate.swift` references them, and `DynamicLibrary.process()` returns `dlsym(RTLD_DEFAULT, llama_backend_init): symbol not found` at runtime.

The two practical integrations:

#### Option B1 — CocoaPods (recommended, what M10 used)

A podspec ships in this repo at [llama_cpp.podspec](llama_cpp.podspec) and is validated on both macOS and iOS. From a Flutter app's `ios/Podfile` (and `macos/Podfile` similarly):

```ruby
platform :ios, '14.0'   # xcframework's MinimumOSVersion is 14.0
...
target 'Runner' do
  use_frameworks!
  flutter_install_all_ios_pods File.dirname(File.realpath(__FILE__))
  pod 'llama_cpp', :path => '../../llama_cpp_dart'   # adjust path
  ...
end
```

The podspec sets `OTHER_LDFLAGS = -force_load $(PODS_XCFRAMEWORKS_BUILD_DIR)/llama_cpp/llama.framework/llama` for both `pod_target_xcconfig` and `user_target_xcconfig`. Without that, M10 reproduced the exact `dlsym` failure above.

After `pod install`, Flutter's `Runner.debug.dylib` (the actual app binary; `Runner` is just a stub) goes from ~89 KB to ~5.4 MB and `nm -gU` shows 200+ `_llama_*` symbols. That's the success criterion — verify with `nm -gU build/ios/iphoneos/Runner.app/Runner.debug.dylib | grep llama_backend_init`.

#### Option B2 — Drag-into-Xcode (manual, only if avoiding CocoaPods)

1. Drag `llama.xcframework` into the Xcode project. Add to "Frameworks, Libraries, and Embedded Content" with "Do Not Embed" (since it's static). "Embed & Sign" with a static framework is wrong — it creates an extra runtime artifact that's never loaded.
2. In Build Settings → "Other Linker Flags" for the Runner target, add: `-force_load $(BUILD_DIR)/.../llama.framework/llama`. The exact path varies; the simplest is to drag the framework's binary file from the navigator while holding Option to insert the resolved path.
3. Code-sign as usual via the project's signing identity.

We don't recommend B2 because the path-to-binary in `-force_load` is fragile across clean builds and Xcode versions. CocoaPods abstracts this.

#### Then, in Dart:

```dart
final engine = await LlamaEngine.spawnFromProcess(
  modelParams: ModelParams(path: modelPath, gpuLayers: 99),
  contextParams: const ContextParams(nCtx: 2048),
);
```

`spawnFromProcess` (rather than `spawn(libraryPath: ...)`) is required because the xcframework is now part of the running binary; there's nothing to `dlopen` by path on iOS.

### iOS gotcha: the Flutter Profile xcconfig is missing by default

When you scaffold a Flutter iOS app with `flutter create`, only `Flutter/Debug.xcconfig` and `Flutter/Release.xcconfig` are generated. But the Xcode project has three configs (Debug / Profile / Release), and the Profile config inherits from `Flutter/Release.xcconfig` — **which doesn't include the Pods xcconfig for Profile builds**. CocoaPods warns about this:

```
[!] CocoaPods did not set the base configuration of your project because your project already has a custom config set ...
    ... include the `Target Support Files/Pods-Runner/Pods-Runner.profile.xcconfig` in your build configuration (`Flutter/Release.xcconfig`).
```

If you ignore the warning and Xcode happens to build with Profile (some workflows do, like AppStore archive builds), the build fails with `Framework 'Pods_Runner' not found / Linker command failed`.

**Fix once per project:**

1. Create `ios/Flutter/Profile.xcconfig`:
   ```
   #include? "Pods/Target Support Files/Pods-Runner/Pods-Runner.profile.xcconfig"
   #include "Generated.xcconfig"
   ```
2. Patch `ios/Runner.xcodeproj/project.pbxproj` so the Runner target's Profile config has `baseConfigurationReference` pointing at the new `Profile.xcconfig` instead of `Release.xcconfig`. The diff is a single XCBuildConfiguration entry — you can do it in Xcode (right-click Runner project → "Set Configurations" → Profile → Flutter/Profile.xcconfig) or by hand in pbxproj. The aichat demo app has a worked example committed.

This isn't actually llama_cpp_dart's bug — it's Flutter's scaffolding gap that bites any pod-using app on Profile builds. But it's worth calling out in HANDOFF because anyone integrating the binding will hit it.

### iOS-specific gotchas (validated on iPad M1, iOS 26.3)

- **iOS apps cannot `dlopen` arbitrary dylib paths.** Path A is fundamentally not available on iOS. Always use Path B.
- **Model files must be inside the app's Documents directory** (`getApplicationDocumentsDirectory()`). Anywhere else is sandbox-denied. Set `UIFileSharingEnabled = YES` and `LSSupportsOpeningDocumentsInPlace = YES` in `Info.plist` to push files via `xcrun devicectl device copy to --domain-type appDataContainer --domain-identifier <bundle-id> --source <local> --destination Documents/<file>`. M10 used this for a 2.83 GB Gemma-4-Q4_0.
- **Free dev profile = max 3 installed apps.** If your developer account is a free Apple ID (not paid), iOS caps at 3 apps from that team. Installing a fourth fails silently — `flutter run` gets stuck at "Dart VM Service was not discovered after 60 seconds". The actual iOS dialog says "This device has reached the maximum number of installed apps using a free developer profile". Delete one of the team's other apps first.
- **First-run Metal shader compilation is ~18s on iPad M1** (vs ~9.5s on M1 Max). 18s of `LlamaEngine.spawn` blocking with no UI is a UX problem — show a splash. After the binary cache is built, subsequent spawns are sub-second.
- **mmproj loads cleanly through `spawnFromProcess`.** Validated on iPad M1 with both Gemma-4 mmproj-F16 (940 MB) and Gemma-4-Q4_0. `engine.supportsVision` and `engine.supportsAudio` both return true.
- **The xcframework has three slices**: `ios-arm64` (device), `ios-arm64-simulator`, `macos-arm64`. Xcode picks the right one per target automatically — no manual selection.
- **`engine.devices` on iOS reports a 3-backend list, not 2.** `BLAS (accel, BLAS): Accelerate` shows up alongside `MTL0` and `CPU` because the iOS slice was compiled with `GGML_BLAS=ON` linking Apple's Accelerate framework. Slightly different from macOS slice which only reports MTL + CPU. Both work.
- **Container UUIDs change per install** (`/var/mobile/Containers/Data/Application/<UUID>/Documents/`). iOS preserves app data across reinstalls *if* the bundle id and signing identity haven't changed, so files pushed via `devicectl` survive a `flutter run` reinstall. The Documents path the app sees changes; `getApplicationDocumentsDirectory()` always returns the live path.

### Suggested binding-side improvements for Apple (post-M10)

- **Auto-`spawnFromProcess` detection.** Today `LlamaEngine.spawn(libraryPath: 'libllama.so')` (basename) works on Android. There's no equivalent ergonomic shortcut on Apple — you're forced to either pass an absolute dylib path (Path A, dev only) or call a different method (`spawnFromProcess`, Path B). It would be cleaner if `spawn` with `libraryPath: null` (or a sentinel) did the right thing on every platform. The binding already has `LlamaLibrary.loadFromProcess()` internally; expose that consistently from the engine.
- **Ship the Podspec with the binding.** A working `llama_cpp.podspec` (with the `-force_load` setting) is now committed to this repo at [llama_cpp.podspec](llama_cpp.podspec). Treat it as part of the binding's public surface from `0.9.0-dev.2` onwards. Document the `pod 'llama_cpp', :path => '...'` snippet in HANDOFF.md / README.md.
- **Update README/HANDOFF to be explicit about static-vs-dynamic xcframeworks.** The current docs say "drag into Xcode → Embed & Sign" — that's wrong for a static framework. App developers will silently get the `dlsym not found` error at runtime if they follow that guidance verbatim. Either fix the docs (use "Do Not Embed" + `-force_load`) or rebuild the xcframework as a dynamic framework so "Embed & Sign" works.
- **Document the sandbox tradeoff prominently** in HANDOFF.md. App developers reading the macOS section will hit "file not found" at runtime if they keep the default sandbox + use Path A. Either tell them to pick Path B from the start, or tell them to disable the sandbox.
- **Optional: rebuild xcframework as a dynamic framework.** Static is simpler and works, but requires `-force_load`. Dynamic would let "Embed & Sign" work as written in current docs and is what most Apple developers expect. Tradeoff: dynamic costs an extra disk binary in the .ipa (~5 MB), static costs the +5 MB inside the Runner.debug.dylib. Net wash. Pick whichever matches the docs you want to write.

---

## Validation results — what you can quote in `plan.md`

All numbers are from `/Users/adel/Workspace/imaged-sdk-examples/aichat`, single-token-stream, prompt = "Write a short paragraph (about 4 sentences) describing why the sky is blue", T=0.3, top_p=0.9, max_tokens=128, fresh session per run, average over 1 run (deterministic — re-runs varied <2% on warm cache).

### Android — same APK, same model, two SoC generations

| Device | Quant | tok/s | Effective backend(s) |
|---|---|---:|---|
| S23U Q8_0 (CPU AAR) | Q8_0 | 7.8 | CPU only |
| S23U Q8_0 (Hex AAR) | Q8_0 | 12.0 | OpenCL + CPU |
| S23U Q4_K_XL | UD-Q4_K_XL | 13.0 | OpenCL + CPU |
| S23U SmolLM3 | Q4_K_M | 13.3 | OpenCL + CPU |
| S23U IQ2_M | UD-IQ2_M | 5.3 | **CPU only** (no OpenCL/HTP kernel for IQ-quants) |
| **S23U** | **Q4_0** | **18.5** | **HTP + OpenCL + CPU** |
| **S23U SmolVLM2** | **Q8_0** | **20.1** | **HTP + OpenCL + CPU** |
| **Fold7** | **Q4_0** | **21.5** | **HTP (v79) + OpenCL (Adreno 830) + CPU** |

### Apple

| Device | Quant | tok/s | Effective backend(s) |
|---|---|---:|---|
| **MacBook M1 Max** (32 GB unified) | **Q4_0** | **45.4** | **Metal (32-core GPU, MTLGPUFamilyApple7) + CPU** |
| **iPad M1** (8 GB unified, iOS 26) | **Q4_0** | **18.4** | **Metal (8-core GPU) + Accelerate BLAS + CPU** |

### Cross-platform comparison on the same model

Gemma-4-E2B-it Q4_0 + mmproj-F16, identical Dart code (only platform branch differs):

| Platform | Accelerator | tok/s | Engine spawn | Δ vs CPU baseline |
|---|---|---:|---:|---:|
| Apple M1 Max (MacBook) | Metal (32-core GPU) | **45.4** | 1.83s (warm) | **5.8×** |
| SD 8 Elite (Fold7) | Hex v79 + Adreno 830 | **21.5** | ~14s | **2.8×** |
| Apple M1 (iPad) | Metal (8-core GPU) + Accelerate | **18.4** | 18.3s first-run, sub-1s warm | **2.4×** |
| SD 8 Gen 2 (S23U) | Hex v73 + Adreno 740 | **18.3** | ~14s | **2.4×** |
| SD 8 Gen 2 (S23U, CPU AAR) | CPU only | 7.8 | ~6s | (baseline) |

Notable cross-platform observations:

- **iPad M1 ≈ Snapdragon 8 Gen 2** at the same tok/s. The 2020-era M1 (8-core GPU) lands within 1% of the 2022-era SD 8 Gen 2's combined Hex+OpenCL stack. Snapdragon 8 Elite (2024) is +17% faster.
- **M1 Max → iPad M1 = 2.5× drop**, lining up with the 32-vs-8 GPU core count ratio. Memory bandwidth difference (LPDDR5x-6400 in iPad vs LPDDR5-6400 in MacBook M1 Max with much wider bus) is the other factor; not separated here.

Headline numbers:

- **Hexagon AAR vs CPU AAR on the same Q8 model: 1.5×** (12.0 vs 7.8 tok/s) from OpenCL alone. The "CPU-only" baseline is unfair to the Hexagon AAR — even when HTP doesn't engage, you still get OpenCL.
- **Q4_0 vs K-quants on the same Hex AAR: +42%** (18.5 vs 13.0 tok/s on Gemma-4) — this is HTP's contribution. K-quants (`Q4_K_*`, `Q5_K_*`) and I-quants (`IQ*`) **do not have HTP kernels in upstream ggml-hexagon as of llama.cpp `5d56effde`**. They run on OpenCL+CPU only. Document this so app developers don't ship a `Q4_K_M` quant expecting NPU acceleration.
- **HTP REPACK budget is ~2 GB per session.** Models above that get partial-offload (some layers HTP, rest OpenCL) per `ggml-backend`'s topology placement. The `NDEV=2` / `D=HTP0,HTP1` multi-session pattern in `docs/backend/snapdragon/README.md` lifts the budget to ~4 GB and is needed for ≥7B-class models. We did not test multi-session.
- **Engine spawn time on Hex AAR is ~11–13s** (model fully resident). On CPU AAR it's ~6s. The extra 5–7s is OpenCL kernel pre-compilation (large `loading OpenCL kernels...` block in the log) plus HTP session setup + REPACK buffer mmap. Worth noting in HANDOFF — first-launch UX suffers.
- **8 Elite vs 8 Gen 2 generational uplift on the same recipe: +16%** (21.5 vs 18.3 tok/s). Comes from a mix of newer Hexagon HVX (v79 vs v73, more threads + higher clocks), Adreno 830 vs 740 (~30% more compute), and Android 16 vs 14 (negligible). The recipe doesn't need any per-SoC tweaking — the same APK ran on both phones unchanged.
- **Apple M1 Max + Metal vs Snapdragon 8 Elite + Hexagon: 2.1×** (45.4 vs 21.5 tok/s). On Apple silicon, Metal carries 100% of the matmul work (all 36 layers offloaded with `gpuLayers=99`); on Android we only get partial HTP offload because of the REPACK budget. Closing this gap on Android requires either (a) `NDEV=2` multi-session HTP for >2 GB models, or (b) waiting for ggml-hexagon to gain K-quant kernels so the small-but-not-tiny K-quants engage HTP. Out of scope for this binding — it's a llama.cpp upstream concern.
- **First-run Metal library compile is ~9.5s** on M1 Max, ~12 ms on subsequent runs (cached to a binary blob). On iPhone the first-run cost is likely worse — plan a splash screen.

---

## What we observed about backend selection

`engine.primaryAcceleratorName` returns **`"GPUOpenCL"`**, not `"HTP0"`, on the S23 Ultra with the Hexagon AAR loaded — even though both backends register and HTP is the faster device for matmul. The HANDOFF.md table currently claims `primaryAcceleratorName = HTP0` for the Hexagon-AAR-on-S23U case. **That documentation row is wrong as written today.** Two ways to fix:

- **Fix the doc:** update the HANDOFF table to say `GPUOpenCL` (current observed behavior), since first-registered-wins is what `engine.primaryAcceleratorName` does today.
- **Fix the binding:** sort registrations by accelerator priority (Hexagon > OpenCL/igpu > CPU) before picking the primary. This matches what an app developer would expect ("primary accelerator" = "fastest accelerator"), but it's a behavior change.

I'd argue for the binding fix — `primaryAcceleratorName` is meant to be a UX hint, not a registration-order accident — but flag it for whoever owns the API.

Concretely: the current backend list comes back as

```
GPUOpenCL (gpu, OpenCL): QUALCOMM Adreno(TM)
HTP0 (gpu, HTP): Hexagon
CPU (cpu, CPU): CPU [11046 / 11046 MiB free]
```

`HTP0` is reported as `(gpu, HTP)` — `BackendDeviceType.gpu` — not `(accel, HTP)`. The HANDOFF table predicted `(accel, Hexagon)`. Since llama.cpp's ggml-backend reports HTP as `GGML_BACKEND_DEVICE_TYPE_GPU` (because Hexagon is treated as a GPU-class offload from a graph-placement standpoint), this is technically correct. But the HANDOFF table should be aligned with reality.

---

## Binding gaps to close before `0.9.0-dev.2`

Listed in priority order. Each is a real thing we hit and worked around — none are speculative.

### Gap A — `ContextParams` doesn't expose `typeK` / `typeV` (KV cache quantization)

`llama_context_params` in C has `type_k` and `type_v` (default `GGML_TYPE_F16`). Common practical setting on memory-constrained mobile is `q8_0` for both, which roughly halves KV-cache memory. Today `ContextParams` exposes `flashAttn` and `offloadKqv` but no KV type knob.

**Fix:** add two enum fields `kvCacheTypeK` / `kvCacheTypeV` (or a single `kvCacheType` if you want the common case only), thread through to the C call, JSON-serialize like the rest. See [`lib/src/context/context_params.dart`](lib/src/context/context_params.dart).

We didn't hit OOM on 12 GB S23U with F16 KV at nCtx 2048. We *will* hit it on 8 GB devices with longer contexts. Worth shipping.

### Gap B — llama.cpp's stderr never reaches logcat

The single biggest debugging blocker during M10. Without redirecting `ggml-hex:` and `ggml_opencl:` log lines to logcat, you cannot see *why* a backend fails to register. We had to:

1. Cross-compile a native arm64 probe (~50 lines of C calling `dlopen` + `ggml_backend_dev_count`)
2. Push it to the device alongside extracted AAR libs
3. Run via `adb shell` — where `stderr` goes to the terminal — to finally see the messages

This shouldn't be required.

**Fix options (in order of effort):**

1. **Easy:** add an `LlamaLog.captureToFile(String path)` API that calls `freopen(path, "w", stderr)` early in `LlamaLibrary.load`. App can read the file and forward to `debugPrint` periodically. Not real-time but trivial to ship.
2. **Medium:** spawn a pthread inside the binding that pipes `stderr` (via `pipe2` + `dup2`) to `__android_log_print`. Real-time, no app-side wiring. Needs a small C shim — `lib/src/ffi/log.dart` is the natural home.
3. **Right thing:** install a Dart-side log callback via `NativeCallable.isolateGroupShared` (already noted in `plan.md` as "soft gap 5"). This is the proper API surface.

The existing `LlamaLog.silence()` / `useDefault()` is too coarse for diagnostics — `useDefault` writes to a stderr that goes nowhere on Android.

### Gap C — `primaryAcceleratorName` ordering

See "What we observed about backend selection" above. Decide: fix the doc or fix the picker.

### Gap D — `nativeLibraryDir` is implicit

App developers must know to:

- Plumb a `MethodChannel` from Dart → Kotlin to fetch `applicationInfo.nativeLibraryDir`
- Call FFI `setenv` before `LlamaLibrary.load`

This is brittle. Two cleaner options:

1. **Ship a tiny Android plugin** as part of `llama_cpp_dart` that exposes `nativeLibraryDir` via a Dart API. Optional dep — only added if the app needs Hexagon. ~30 lines of Kotlin + 10 lines of Dart.
2. **Discover at runtime** by reading `/proc/self/maps` for the loaded `libllama.so` path and using its dirname. Hacky but pure-Dart, no plugin needed. Works because once `LlamaLibrary.load("libllama.so")` succeeds, the resolved path is in `/proc/self/maps`.

Both are improvements. I'd ship (1) — cleaner, more discoverable.

If you do (1), also auto-`setenv("ADSP_LIBRARY_PATH", nativeLibraryDir)` from inside `LlamaLibrary.load` on Android. Removes ingredient #4 from the recipe entirely.

### Gap E — small things

- The "Failed to punch uncompressed elf file: libggml-htp-v*.so" install-time warning. Cosmetic, harmless (those are DSP-target ELFs the Android linker can't classify). HANDOFF already documents this. Don't need to fix.
- The Hexagon AAR's `tmp/llama_cpp_dart_hexagon_empty` zero-byte sentinel file gets bundled into the APK at `tmp/llama_cpp_dart_hexagon_empty`. Not a problem, but worth dropping from the AAR build if you can — pollutes the APK contents.

---

## What's still un-validated

Be honest in `plan.md` about what M10 covered:

- ✅ Android CPU AAR — text-only, decode + cancel, on real device (S23U).
- ✅ Android Hexagon AAR — HTP backend registers, Q4_0/Q8_0 actually engage NPU, mmproj + multimodal context loads, decode + cancel work. Tested on **two SoC generations** (8 Gen 2 / S23U and 8 Elite / Fold7) with the same APK.
- ✅ macOS Metal — dev/test path (Path A, dylib by absolute path). 45.4 tok/s on M1 Max. mmproj loads. `engine.devices` reports `MTL0 (gpu, MTL): Apple M1 Max` correctly.
- ✅ iOS Metal on iPad M1 — Path B (xcframework via CocoaPods + `-force_load` + `spawnFromProcess`). 18.4 tok/s on Q4_0. mmproj loads. `engine.devices` reports 3 backends: `MTL0`, `BLAS` (Accelerate), `CPU`.
- ✅ Backend inspection API (`engine.devices`, `LlamaBackends.list()`) — works correctly on every platform.
- ⚠️ **Multimodal vision generation on device** — not yet exercised. mmproj loads (Gemma-4, SmolVLM2, all platforms); we did not push an image and run a describe-this turn. Worth doing as a final smoke before sign-off; the demo app's code path for it is already in place.
- ⚠️ **Multimodal audio** — same situation; mmproj reports `supportsAudio = true`, no on-device audio test ran.
- ⚠️ **Persistence (`saveState`/`loadState`) on device** — not exercised. Should round-trip fine since the format is platform-agnostic, but don't claim it without verifying.
- ⚠️ **Context shift on device** — `engine.canShift = true` is reported correctly on every platform; we didn't actually trigger a shift mid-generation.
- ⚠️ **macOS Path B (xcframework, sandboxed Release)** — only Path A (loose dylib, sandbox disabled) was validated on macOS. The same podspec used successfully on iOS should work on macOS, but it wasn't tested in M10. Worth a quick sanity run before declaring App-Store-ready.
- ❌ **NDEV=2 multi-session HTP** — needed for ≥4 GB models. Not tested. Pixel 8/9 might be where you'd validate this — or someone with a Snapdragon dev kit.
- ❌ **TestFlight / App Store distribution** — only Debug builds with development signing tested on iOS. Distribution profile + ad-hoc / TestFlight signing not exercised.

---

## Reference material

If you want to dig further into the Hexagon stack, the canonical sources we used:

- llama.cpp Snapdragon README: <https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md> — has the working `D=HTP0` / `M=Llama-3.2-1B-Q4_0` example showing 51 tok/s on HTP for a 1B Q4_0 model. Best benchmark target if you want a "definitely engages HTP" reference run.
- llama.cpp issue #16911 ("ggml-hexagon failed to get URI for session 0 : error 0x14"): <https://github.com/ggml-org/llama.cpp/issues/16911> — full context on the 8 Gen 2 URI bug + workaround. Read all comments.
- PR #16987 (l3utterfly's `dlopen` + weak-symbol approach): <https://github.com/ggml-org/llama.cpp/pull/16987>
- chraac's branch with the merged URI-fallback: <https://github.com/chraac/llama.cpp/blob/dev-8gen2-support/ggml/src/ggml-hexagon/ggml-hexagon.cpp>
- Demo app proving everything above: `/Users/adel/Workspace/imaged-sdk-examples/aichat`. The `lib/main.dart` smoke screen is a minimal reference for what an app needs to do.

---

## Suggested order of operations for `0.9.0-dev.2`

1. **Update `HANDOFF.md`** with both corrected sections — Hexagon (5-ingredient recipe) and Apple (Path A vs Path B, podspec + `-force_load`, Profile xcconfig fix) (1–2 hours). This unblocks any other app developer immediately.
2. **Polish the Podspec** ([llama_cpp.podspec](llama_cpp.podspec)) — the M10 version uses the local `build/apple/llama.xcframework` path. For pub.dev / GitHub-Releases workflows, point at the released xcframework zip instead. ~1 hour.
3. **Land Gap A** (`typeK`/`typeV` in `ContextParams`) — small, mechanical, ~2 hours.
4. **Land Gap B option 1** (file-based stderr capture) as a stopgap — 1 hour. Defer the real `NativeCallable.isolateGroupShared` callback to `0.9.x` if it's risky.
5. **Land Gap D option 1** (auto-`setenv` from a tiny Android plugin) — 4–6 hours including testing. Removes a major footgun.
6. **Decide Gap C** (doc-fix vs picker-fix) — 30 min one way or the other.
7. **Validate macOS Path B** — same podspec as iOS, but with sandboxed Release entitlements + `xcframework via Pods`. Should be ~30 min if it just works (it should — same recipe).
8. **Bump llama.cpp submodule** to whatever's current upstream and re-run M10 quickly to verify nothing regressed on 8 Gen 2 + 8 Elite + M1 + iPad M1.
9. Tag `0.9.0-dev.2`.

That's roughly a day to a day-and-a-half of focused work. After that the binding is in shape to ship `0.9.0`.

Good luck.
