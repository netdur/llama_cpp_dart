#!/usr/bin/env bash
#
# Build an Apple xcframework with three slices: ios-arm64,
# ios-arm64-simulator, macos-arm64. Each slice is a static `.framework`
# bundle containing a merged `libllama` archive (llama + ggml + mtmd),
# all public headers, and a module map.
#
# Output: build/apple/llama.xcframework/
#
# Usage: tool/build_apple_xcframework.sh [--clean]
#
# Requirements: Xcode (for xcodebuild + libtool), CMake 3.28+, Apple
# Silicon Mac (arm64 host).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SRC="$PROJECT_ROOT/src/llama.cpp"
OUT_ROOT="$PROJECT_ROOT/build/apple"
XCF_OUT="$OUT_ROOT/llama.xcframework"

CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this script must run on macOS" >&2
  exit 2
fi
if ! command -v xcodebuild >/dev/null; then
  echo "error: xcodebuild not found (install Xcode)" >&2
  exit 2
fi
if ! command -v cmake >/dev/null; then
  echo "error: cmake not found (brew install cmake)" >&2
  exit 2
fi

if [[ "$CLEAN" -eq 1 ]]; then
  rm -rf "$OUT_ROOT"
fi
mkdir -p "$OUT_ROOT"

IOS_MIN=14.0
MACOS_MIN=12.0

# Common cmake args. Static libs only — combined into one framework
# binary per slice so the xcframework ships as a single dependency.
COMMON_ARGS=(
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_CURL=OFF
  -DLLAMA_BUILD_TOOLS=ON   # for libmtmd (vision + audio)
  -DGGML_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON
  -DGGML_METAL_USE_BF16=ON
  -DGGML_BLAS_DEFAULT=ON
  -DGGML_NATIVE=OFF
  -DGGML_OPENMP=OFF
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
)

# ----- header set we ship in every slice's framework -----
HEADERS=(
  "$LLAMA_SRC/include/llama.h"
  "$LLAMA_SRC/ggml/include/ggml.h"
  "$LLAMA_SRC/ggml/include/ggml-alloc.h"
  "$LLAMA_SRC/ggml/include/ggml-backend.h"
  "$LLAMA_SRC/ggml/include/ggml-cpu.h"
  "$LLAMA_SRC/ggml/include/ggml-metal.h"
  "$LLAMA_SRC/ggml/include/ggml-opt.h"
  "$LLAMA_SRC/ggml/include/gguf.h"
  "$LLAMA_SRC/tools/mtmd/mtmd.h"
  "$LLAMA_SRC/tools/mtmd/mtmd-helper.h"
)

# Build one slice. Args:
#   $1 slice name (e.g. "ios-arm64", "ios-arm64-simulator", "macos-arm64")
#   $2 cmake system name ("iOS" | "Darwin")
#   $3 cmake osx sysroot ("iphoneos" | "iphonesimulator" | "macosx")
#   $4 deployment target ("14.0" | "12.0")
build_slice() {
  local slice="$1"
  local sys_name="$2"
  local sysroot="$3"
  local min_version="$4"

  local build_dir="$OUT_ROOT/build-$slice"
  local fw_dir="$build_dir/framework/llama.framework"

  echo
  echo "==== building slice: $slice"
  echo "      system=$sys_name sysroot=$sysroot deployment=$min_version"

  rm -rf "$build_dir"
  mkdir -p "$fw_dir/Headers" "$fw_dir/Modules"

  cmake -G Xcode -B "$build_dir" -S "$LLAMA_SRC" \
    -DCMAKE_SYSTEM_NAME="$sys_name" \
    -DCMAKE_OSX_SYSROOT="$sysroot" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$min_version" \
    "${COMMON_ARGS[@]}"

  cmake --build "$build_dir" --config Release --parallel

  # Collect every static archive cmake produced, merge into one library.
  # Xcode names build dirs Release/, Release-iphoneos/, Release-iphonesimulator/
  # depending on the slice — match all three with one path glob.
  local archives=()
  while IFS= read -r -d '' a; do archives+=("$a"); done < <(
    find "$build_dir" -name '*.a' \( -path '*/Release/*' -o -path '*/Release-*' \) -print0 2>/dev/null
  )
  # Drop archives we don't want to ship (httplib, llama-common server-only bits).
  local kept=()
  for a in "${archives[@]}"; do
    case "$a" in
      */libllama-common*) ;;
      */libcpp-httplib*) ;;
      *) kept+=("$a") ;;
    esac
  done
  archives=("${kept[@]}")
  if [[ ${#archives[@]} -eq 0 ]]; then
    echo "error: no .a archives found under $build_dir" >&2
    exit 1
  fi
  echo "  merging ${#archives[@]} archives into framework binary"
  libtool -static -o "$fw_dir/llama" "${archives[@]}" 2>/dev/null

  # Copy headers.
  for h in "${HEADERS[@]}"; do
    [[ -f "$h" ]] || { echo "error: missing header $h" >&2; exit 1; }
    cp "$h" "$fw_dir/Headers/"
  done

  # Module map for Swift interop (harmless for Dart users).
  cat >"$fw_dir/Modules/module.modulemap" <<EOF
framework module llama {
  umbrella header "llama.h"
  export *
  module * { export * }
}
EOF

  # Info.plist describing the framework binary.
  /usr/libexec/PlistBuddy -c "Clear dict" "$fw_dir/Info.plist" >/dev/null 2>&1 || true
  cat >"$fw_dir/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>llama</string>
  <key>CFBundleIdentifier</key><string>org.ggml.llama</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>llama</string>
  <key>CFBundlePackageType</key><string>FMWK</string>
  <key>CFBundleShortVersionString</key><string>0.9.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>MinimumOSVersion</key><string>$min_version</string>
</dict>
</plist>
EOF

  echo "  slice ready: $fw_dir"
}

# ----- build each slice -----
build_slice "ios-arm64"           "iOS"    "iphoneos"        "$IOS_MIN"
build_slice "ios-arm64-simulator" "iOS"    "iphonesimulator" "$IOS_MIN"
build_slice "macos-arm64"         "Darwin" "macosx"          "$MACOS_MIN"

# ----- assemble xcframework -----
rm -rf "$XCF_OUT"
xcodebuild -create-xcframework \
  -framework "$OUT_ROOT/build-ios-arm64/framework/llama.framework" \
  -framework "$OUT_ROOT/build-ios-arm64-simulator/framework/llama.framework" \
  -framework "$OUT_ROOT/build-macos-arm64/framework/llama.framework" \
  -output "$XCF_OUT"

echo
echo "==== built $XCF_OUT"
ls -1 "$XCF_OUT"
