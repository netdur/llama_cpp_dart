#!/usr/bin/env bash
#
# Build an Android AAR with the Hexagon NPU + OpenCL backends enabled.
#
# Uses the upstream Snapdragon toolchain Docker image
# (ghcr.io/snapdragon-toolchain/arm64-android:v0.3), which bundles the
# Android NDK, Hexagon SDK 6.4, and OpenCL SDK pre-installed. This is
# the same image llama.cpp's Hexagon docs recommend.
#
# Output: build/android-hexagon/llama-cpp-dart-hexagon.aar
#
# Usage: tool/build_android_hexagon_aar.sh [--clean] [--no-pull]
#
# Notes:
#   - First run pulls a ~5GB image; expect 5-10 minutes on a fast pipe.
#   - The image is amd64. On Apple Silicon it runs through Rosetta 2,
#     so the build itself is slow (~15-20 minutes for a clean build).
#   - The output AAR works on Snapdragon 8 Gen 2 (HTP v73) through
#     8 Elite / 8 Gen 4 (v81). On non-Snapdragon devices the runtime
#     falls back to CPU automatically.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SRC="$PROJECT_ROOT/src/llama.cpp"
OUT_ROOT="$PROJECT_ROOT/build/android-hexagon"
PKG_DIR="$OUT_ROOT/pkg"
AAR_OUT="$OUT_ROOT/llama-cpp-dart-hexagon.aar"
ABI="arm64-v8a"
ANDROID_PLATFORM=31

IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.3"
CLEAN=0
DO_PULL=1

for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --no-pull) DO_PULL=0 ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v docker >/dev/null; then
  echo "error: docker not found (install Docker Desktop)" >&2
  exit 2
fi
if ! command -v zip >/dev/null; then
  echo "error: zip not found" >&2
  exit 2
fi

if [[ "$DO_PULL" -eq 1 ]]; then
  echo "==== pulling $IMAGE (large; first-time only)"
  docker pull --platform linux/amd64 "$IMAGE"
fi

if [[ "$CLEAN" -eq 1 ]]; then
  rm -rf "$OUT_ROOT"
fi
mkdir -p "$OUT_ROOT"

# llama.cpp's Hexagon preset reads CMakeUserPresets.json from the project
# root. Stage the upstream-supplied one there for the duration of the
# build. The container will see it via the volume mount.
PRESETS_SRC="$LLAMA_SRC/docs/backend/snapdragon/CMakeUserPresets.json"
if [[ ! -f "$PRESETS_SRC" ]]; then
  echo "error: $PRESETS_SRC not found — submodule out of date?" >&2
  exit 2
fi
cp "$PRESETS_SRC" "$LLAMA_SRC/CMakeUserPresets.json"
trap 'rm -f "$LLAMA_SRC/CMakeUserPresets.json"' EXIT

# Where the build lands inside the container (and on the host via the
# bind mount).
BUILD_DIR_REL="build/android-hexagon/build"
PKG_DIR_REL="build/android-hexagon/pkg"

mkdir -p "$PROJECT_ROOT/$BUILD_DIR_REL"

echo
echo "==== running build inside Snapdragon toolchain container"
docker run --rm \
  --platform linux/amd64 \
  -u "$(id -u):$(id -g)" \
  --volume "$PROJECT_ROOT:/workspace" \
  --workdir "/workspace/src/llama.cpp" \
  -e LLAMA_BUILD_TOOLS=ON \
  "$IMAGE" \
  bash -c "
    set -e
    echo '[container] starting Hexagon build for $ABI'
    # The upstream Snapdragon preset enables -flto which doubles peak
    # compile-time memory and OOM-kills cpp-httplib under Rosetta.
    # Drop -flto here — small runtime perf hit, much smaller memory
    # footprint. The vector / fast-math flags stay.
    NOLTO_C_FLAGS='-march=armv8.7a+fp16 -fvectorize -ffp-model=fast -fno-finite-math-only -D_GNU_SOURCE'
    cmake --preset arm64-android-snapdragon-release \
      -B /workspace/$BUILD_DIR_REL \
      -DCMAKE_INSTALL_PREFIX=/workspace/$PKG_DIR_REL \
      -DCMAKE_C_FLAGS=\"\$NOLTO_C_FLAGS\" \
      -DCMAKE_CXX_FLAGS=\"\$NOLTO_C_FLAGS\" \
      -DBUILD_SHARED_LIBS=ON \
      -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_CURL=OFF \
      -DANDROID_PLATFORM=android-$ANDROID_PLATFORM
    # Cap parallel jobs at 2 to keep peak memory bounded under Rosetta.
    cmake --build /workspace/$BUILD_DIR_REL -j 2
    cmake --install /workspace/$BUILD_DIR_REL
  "

# Collect runtime .so files. The Snapdragon preset produces:
#   libllama.so, libggml.so, libggml-base.so, libggml-cpu.so,
#   libggml-opencl.so, libggml-hexagon.so,
#   libggml-htp-v73/v75/v79/v81.so, libmtmd.so
echo
echo "==== staging AAR"
STAGE_DIR="$OUT_ROOT/aar-stage"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/jni/$ABI"

LIB_DIR="$PKG_DIR/lib"
if [[ ! -d "$LIB_DIR" ]]; then
  echo "error: install dir $LIB_DIR not found — build failed?" >&2
  exit 1
fi

WANT_LIBS=(
  libllama.so
  libggml.so
  libggml-base.so
  libggml-cpu.so
  libggml-opencl.so
  libggml-hexagon.so
  # DSP-side libs for Hexagon Tensor Processor variants:
  #   v68 = SD 865/870, v69 = SD 8 Gen 1, v73 = SD 8 Gen 2,
  #   v75 = SD 8 Gen 3, v79 = SD 8 Gen 4 / 8 Elite, v81 = future.
  # These are Hexagon ELF, not ARM; runtime picks the right one.
  libggml-htp-v68.so
  libggml-htp-v69.so
  libggml-htp-v73.so
  libggml-htp-v75.so
  libggml-htp-v79.so
  libggml-htp-v81.so
  libmtmd.so
)
COPIED=0
for lib in "${WANT_LIBS[@]}"; do
  if [[ -f "$LIB_DIR/$lib" ]]; then
    cp "$LIB_DIR/$lib" "$STAGE_DIR/jni/$ABI/"
    COPIED=$((COPIED + 1))
  else
    echo "  warning: $lib not in install dir" >&2
  fi
done
if [[ "$COPIED" -eq 0 ]]; then
  echo "error: no .so files copied" >&2
  exit 1
fi
echo "  copied $COPIED libs"

# Strip ARM .so files (huge unstripped — libllama starts at 30 MB).
# Skip the HTP DSP libs; their ELF format isn't recognized by llvm-strip
# in the same way.
NDK_LATEST="$(ls /Users/adel/Library/Android/sdk/ndk 2>/dev/null | sort -V | tail -1 || true)"
if [[ -n "$NDK_LATEST" ]]; then
  STRIP_CANDIDATES=(
    "/Users/adel/Library/Android/sdk/ndk/$NDK_LATEST/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-strip"
    "/Users/adel/Library/Android/sdk/ndk/$NDK_LATEST/toolchains/llvm/prebuilt/darwin-arm64/bin/llvm-strip"
  )
  for cand in "${STRIP_CANDIDATES[@]}"; do
    if [[ -x "$cand" ]]; then
      for so in "$STAGE_DIR/jni/$ABI/"*.so; do
        # Skip Hexagon ELF (Machine: Qualcomm Hexagon, e_machine = 164).
        if [[ "$(basename "$so")" == libggml-htp-* ]]; then continue; fi
        "$cand" --strip-unneeded "$so" 2>/dev/null || true
      done
      echo "  stripped ARM libs (HTP DSP libs left intact)"
      break
    fi
  done
fi

cat >"$STAGE_DIR/AndroidManifest.xml" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="dev.netdur.llama_cpp_dart_hexagon">
    <uses-sdk android:minSdkVersion="$ANDROID_PLATFORM" />
</manifest>
EOF

cd "$STAGE_DIR"
echo -n "" > /tmp/llama_cpp_dart_hexagon_empty
zip -q classes.jar /tmp/llama_cpp_dart_hexagon_empty
rm /tmp/llama_cpp_dart_hexagon_empty
zip -d classes.jar llama_cpp_dart_hexagon_empty 2>/dev/null || true
: > R.txt

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
