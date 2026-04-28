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

## Hexagon NPU (post-1.0, M8.5)

Not built today. Requires the Qualcomm **Hexagon SDK** (separate
download — QAIRT/QNN does not include the DSP compiler toolchain that
`ggml-hexagon` needs). Plan: use the
`ghcr.io/snapdragon-toolchain/arm64-android` Docker image once a
validation device is on hand.

## Regenerating bindings

```bash
dart pub get
dart run ffigen
```

Reads `pubspec.yaml`'s `ffigen:` block, regenerates
`lib/src/ffi/bindings.dart` against the pinned llama.cpp + mtmd headers.
Run this whenever `src/llama.cpp` is bumped.
