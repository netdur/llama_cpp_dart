#!/usr/bin/env bash
#
# Build native llama.cpp shared libraries for a target platform.
#
# Usage:
#   tool/build_native.sh --platform macos [--with-mtmd] [--clean]
#
# Outputs:
#   build/<platform>/install/lib/libllama.dylib (or .so)
#   build/<platform>/install/lib/libggml*.dylib
#   build/<platform>/install/lib/libmtmd.dylib (when --with-mtmd)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SRC="$PROJECT_ROOT/src/llama.cpp"

PLATFORM=""
WITH_MTMD=0
CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    --with-mtmd) WITH_MTMD=1; shift ;;
    --clean) CLEAN=1; shift ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PLATFORM" ]]; then
  echo "error: --platform is required (macos|ios|android)" >&2
  exit 2
fi

if [[ ! -d "$LLAMA_SRC" ]]; then
  echo "error: $LLAMA_SRC not found. Run: git submodule update --init src/llama.cpp" >&2
  exit 2
fi

BUILD_DIR="$PROJECT_ROOT/build/$PLATFORM"
INSTALL_DIR="$BUILD_DIR/install"

if [[ "$CLEAN" -eq 1 ]]; then
  rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

case "$PLATFORM" in
  macos)
    if [[ "$(uname -s)" != "Darwin" ]]; then
      echo "error: macOS build must run on a Darwin host" >&2
      exit 2
    fi

    CMAKE_ARGS=(
      -S "$LLAMA_SRC"
      -B "$BUILD_DIR"
      -G "Ninja"
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_OSX_ARCHITECTURES=arm64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
      -DBUILD_SHARED_LIBS=ON
      # Make every shared library find its siblings in the same directory.
      # libllama.dylib lists @rpath/libggml*.dylib; with rpath=@loader_path,
      # dyld resolves those to "next to me" — matching how we ship binaries.
      -DCMAKE_MACOSX_RPATH=ON
      -DCMAKE_INSTALL_RPATH=@loader_path
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
      -DCMAKE_INSTALL_NAME_DIR=@rpath
      -DGGML_METAL=ON
      -DGGML_METAL_EMBED_LIBRARY=ON
      -DGGML_BLAS=OFF
      -DGGML_OPENMP=OFF
      -DLLAMA_BUILD_TESTS=OFF
      -DLLAMA_BUILD_EXAMPLES=OFF
      -DLLAMA_CURL=OFF
    )

    if [[ "$WITH_MTMD" -eq 1 ]]; then
      CMAKE_ARGS+=(-DLLAMA_BUILD_TOOLS=ON)
    else
      CMAKE_ARGS+=(-DLLAMA_BUILD_TOOLS=OFF)
    fi

    if ! command -v ninja >/dev/null; then
      echo "error: ninja not found. Install with: brew install ninja" >&2
      exit 2
    fi

    cmake "${CMAKE_ARGS[@]}"
    cmake --build "$BUILD_DIR" --parallel
    cmake --install "$BUILD_DIR"

    echo
    echo "macOS build complete:"
    ls -1 "$INSTALL_DIR/lib"
    ;;

  ios|android)
    echo "error: $PLATFORM build is not implemented yet (planned for M7/M8)" >&2
    exit 2
    ;;

  *)
    echo "error: unknown platform '$PLATFORM' (expected macos|ios|android)" >&2
    exit 2
    ;;
esac
