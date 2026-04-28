#!/usr/bin/env bash
#
# Build an Android AAR containing libllama.so + libggml.so family +
# libmtmd.so for arm64-v8a. CPU + mtmd; no Hexagon (that needs the
# Hexagon SDK + Snapdragon Docker toolchain — see plan.md M8.5).
#
# Output: build/android/llama-cpp-dart.aar
#
# Usage: tool/build_android_aar.sh [--ndk PATH] [--abi arm64-v8a] [--clean]
#
# Defaults:
#   --ndk   $ANDROID_NDK_ROOT or /Users/adel/Library/Android/sdk/ndk/<latest>
#   --abi   arm64-v8a (the only one that ships)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SRC="$PROJECT_ROOT/src/llama.cpp"
OUT_ROOT="$PROJECT_ROOT/build/android"
AAR_OUT="$OUT_ROOT/llama-cpp-dart.aar"

NDK="${ANDROID_NDK_ROOT:-}"
ABI="arm64-v8a"
CLEAN=0
ANDROID_PLATFORM=26 # Android 8.0 (Oreo). Common floor for modern apps.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ndk) NDK="$2"; shift 2 ;;
    --abi) ABI="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Auto-discover NDK if not specified.
if [[ -z "$NDK" ]]; then
  CANDIDATE="/Users/adel/Library/Android/sdk/ndk"
  if [[ -d "$CANDIDATE" ]]; then
    # Pick the lexicographically-largest version (safe enough for our SDKs).
    NDK="$CANDIDATE/$(ls -1 "$CANDIDATE" | sort -V | tail -1)"
  fi
fi
if [[ ! -d "$NDK" ]]; then
  echo "error: Android NDK not found. Pass --ndk PATH or set ANDROID_NDK_ROOT." >&2
  exit 2
fi
echo "using NDK: $NDK"

if ! command -v cmake >/dev/null; then
  echo "error: cmake not found (brew install cmake)" >&2
  exit 2
fi
if ! command -v ninja >/dev/null; then
  echo "error: ninja not found (brew install ninja)" >&2
  exit 2
fi
if ! command -v zip >/dev/null; then
  echo "error: zip not found" >&2
  exit 2
fi

BUILD_DIR="$OUT_ROOT/build-$ABI"
INSTALL_DIR="$OUT_ROOT/install-$ABI"
STAGE_DIR="$OUT_ROOT/aar-stage"

if [[ "$CLEAN" -eq 1 ]]; then
  rm -rf "$BUILD_DIR" "$INSTALL_DIR" "$STAGE_DIR" "$AAR_OUT"
fi
mkdir -p "$BUILD_DIR"

echo "==== configuring llama.cpp for Android $ABI (platform $ANDROID_PLATFORM)"
cmake -G Ninja -B "$BUILD_DIR" -S "$LLAMA_SRC" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ABI" \
  -DANDROID_PLATFORM="android-$ANDROID_PLATFORM" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG"

echo "==== building"
cmake --build "$BUILD_DIR" --parallel
cmake --install "$BUILD_DIR"

# Collect runtime .so files. cmake --install drops them under
# $INSTALL_DIR/lib/. Filter to the runtime set we ship.
RUNTIME_SOS=()
for so in libllama.so libggml.so libggml-base.so libggml-cpu.so libmtmd.so; do
  if [[ -f "$INSTALL_DIR/lib/$so" ]]; then
    RUNTIME_SOS+=("$INSTALL_DIR/lib/$so")
  else
    # Some configs lay them under build/; fall back to find.
    found="$(find "$BUILD_DIR" -name "$so" -type f 2>/dev/null | head -1)"
    if [[ -n "$found" ]]; then
      RUNTIME_SOS+=("$found")
    else
      echo "warning: $so not found, skipping" >&2
    fi
  fi
done
if [[ ${#RUNTIME_SOS[@]} -eq 0 ]]; then
  echo "error: no .so files found to package" >&2
  exit 1
fi

echo
echo "==== staging AAR"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/jni/$ABI"

for so in "${RUNTIME_SOS[@]}"; do
  cp "$so" "$STAGE_DIR/jni/$ABI/"
done

# Strip debug info to shrink the .so files.
STRIP="$NDK/toolchains/llvm/prebuilt/$(ls "$NDK/toolchains/llvm/prebuilt/" | head -1)/bin/llvm-strip"
if [[ -x "$STRIP" ]]; then
  for so in "$STAGE_DIR/jni/$ABI"/*.so; do
    "$STRIP" --strip-unneeded "$so"
  done
  echo "stripped debug info"
fi

cat >"$STAGE_DIR/AndroidManifest.xml" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="dev.netdur.llama_cpp_dart">
    <uses-sdk android:minSdkVersion="$ANDROID_PLATFORM" />
</manifest>
EOF

# AAR requires a (possibly empty) classes.jar. Empty jar = empty zip.
cd "$STAGE_DIR"
echo -n "" > /tmp/llama_cpp_dart_empty
zip -q classes.jar /tmp/llama_cpp_dart_empty
rm /tmp/llama_cpp_dart_empty
zip -d classes.jar llama_cpp_dart_empty 2>/dev/null || true

# R.txt is also expected; empty is fine.
: > R.txt

# Pack the AAR.
mkdir -p "$(dirname "$AAR_OUT")"
zip -qr "$AAR_OUT" \
  AndroidManifest.xml \
  classes.jar \
  R.txt \
  jni

cd - >/dev/null

echo
echo "==== built $AAR_OUT"
unzip -l "$AAR_OUT"
echo
echo "size: $(du -h "$AAR_OUT" | awk '{print $1}')"
