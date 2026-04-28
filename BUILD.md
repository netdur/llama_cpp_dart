# Building native artifacts

llama_cpp_dart ships three platform artifacts. The Dart package itself
contains no binaries — these are produced from the pinned `src/llama.cpp`
submodule and uploaded to GitHub Releases.

```bash
git submodule update --init src/llama.cpp
```

## macOS dylib (development / `dart test`)

```bash
tool/build_native.sh --platform macos --with-mtmd
```

Output: `build/macos/install/lib/libllama.dylib` plus `libggml*.dylib` +
`libmtmd.dylib`. `@loader_path` rpath baked in; sibling dylibs are picked
up by `dyld` automatically.

Used by:

```dart
LlamaEngine.spawn(libraryPath: '<repo>/build/macos/install/lib/libllama.dylib', ...);
```

Requirements: Xcode command-line tools, CMake 3.28+, Ninja
(`brew install ninja`).

## Apple xcframework (iOS device + iOS simulator + macOS)

```bash
tool/build_apple_xcframework.sh
```

Output: `build/apple/llama.xcframework/` with three slices:

- `ios-arm64` (real iPhone/iPad, iOS 14+)
- `ios-arm64-simulator` (Apple Silicon Mac running iOS Simulator)
- `macos-arm64` (Apple Silicon macOS app, 12+)

Each slice is a static `llama.framework/llama` binary (~9 MB) merged
from libllama + libggml{base,cpu,metal,blas} + libmtmd. Metal shader
source is embedded (`GGML_METAL_EMBED_LIBRARY=ON`).

Used by iOS / macOS apps:

```dart
LlamaEngine.spawnFromProcess(...)  // Xcode static-links the framework
```

Drag the `.xcframework` into Xcode, mark "Embed & Sign" on the app
target. No `LlamaLibrary.load` path needed.

Requirements: Xcode 15+ (with `xcodebuild` and `libtool`).

## Android AAR (arm64-v8a, CPU + mtmd)

```bash
tool/build_android_aar.sh                                # auto-discovers NDK
tool/build_android_aar.sh --ndk /path/to/ndk --abi arm64-v8a
```

Output: `build/android/llama-cpp-dart.aar` — ~2 MB containing
`jni/arm64-v8a/{libllama,libggml,libggml-base,libggml-cpu,libmtmd}.so`.
minSdk 26 (Android 8.0).

Used by Flutter Android apps. Two integration paths:

1. `android/app/libs/llama-cpp-dart.aar` + `implementation files('libs/llama-cpp-dart.aar')` in Gradle.
2. Drop the raw `.so` files into `android/src/main/jniLibs/arm64-v8a/`.

Then in Dart:

```dart
LlamaEngine.spawn(libraryPath: 'libllama.so', ...);  // basename — Android resolves
```

Requirements: Android NDK r25+ (path via `--ndk` or `$ANDROID_NDK_ROOT`,
or auto-discover under `~/Library/Android/sdk/ndk/<latest>`), CMake,
Ninja.

## Android AAR with Hexagon NPU + OpenCL (Snapdragon-only)

```bash
tool/build_android_hexagon_aar.sh                # first run pulls 8 GB
tool/build_android_hexagon_aar.sh --no-pull      # subsequent runs
tool/build_android_hexagon_aar.sh --clean        # wipe and rebuild
```

Output: `build/android-hexagon/llama-cpp-dart-hexagon.aar` — ~3.7 MB
containing 11 `.so` files in `jni/arm64-v8a/`:

- ARM64 host-side: `libllama.so`, `libggml{,-base,-cpu,-opencl,-hexagon}.so`, `libmtmd.so`
- Hexagon DSP-side: `libggml-htp-v{68,69,73,75,79,81}.so`

Variant-to-device mapping:

| HTP variant | Snapdragon |
|---|---|
| v68 | 865 / 870 (2020) |
| v69 | 8 Gen 1 (2022) |
| v73 | 8 Gen 2 (2023) |
| v75 | 8 Gen 3 (2024) |
| v79 | 8 Gen 4 / 8 Elite (2025) |
| v81 | future |

The runtime picks the right HTP at load time. On non-Snapdragon Android
devices the Hexagon backend reports unavailable and the engine falls
back to CPU + OpenCL automatically.

Used by:

```dart
LlamaEngine.spawn(libraryPath: 'libllama.so', ...);
// Same API — backend selection happens inside ggml-backend.
```

Requirements:

- Docker (Docker Desktop on macOS, native daemon on Linux).
- ~8 GB free disk for the toolchain image.
- On Apple Silicon: Docker runs the amd64 image through Rosetta 2,
  which is slow (~15–20 min for a clean build) and memory-tight
  (the script disables `-flto` and caps `-j 2` to avoid OOM-killing
  cpp-httplib). Native amd64 hosts get a clean ~5–10 min build.

The Snapdragon toolchain image
(`ghcr.io/snapdragon-toolchain/arm64-android:v0.3`) bundles:

- Android NDK r28b
- Hexagon SDK 6.4.0.2 (DSP compiler + IDL tools)
- OpenCL SDK
- CMake, Ninja

## Regenerating bindings

```bash
dart pub get
dart run ffigen
```

Reads `pubspec.yaml`'s `ffigen:` block, regenerates
`lib/src/ffi/bindings.dart` against the pinned llama.cpp + mtmd headers.
Run this whenever `src/llama.cpp` is bumped.
