#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 🛠  CONFIGURATION – adjust names/paths if your project changes
###############################################################################
FRAMEWORK_NAME="Llama"
MAIN_LIB_NAME="libllama.dylib"
FRAMEWORK_VERSION="0.0.9"
FRAMEWORK_ID="io.github.netdur.llama_cpp_dart"

MIN_IOS_VERSION="13.0"
MIN_MACOS_VERSION="12.0"

DEPENDENCY_LIBS=(
  "libggml.dylib" "libggml-metal.dylib" "libggml-base.dylib"
  "libggml-cpu.dylib" "libggml-blas.dylib"
)

# paths where YOUR build system placed the thin architectures
IOS_DEVICE_LIB_DIR="bin/OS64"
IOS_SIM_ARM64_LIB_DIR="bin/SIMULATORARM64"
IOS_SIM_X86_64_LIB_DIR="bin/SIMULATOR64"
MACOS_ARM64_LIB_DIR="bin/MAC_ARM64"

HEADER_SOURCE_DIRS=(
  "src/llama.cpp/include"
  "src/llama.cpp/ggml/include"
)

PLIST_TEMPLATE="darwin/Info.plist"

###############################################################################
# 🔑  PICK A SIGNING IDENTITY (optional) – leave empty for ad-hoc
###############################################################################
IDENTITY="${SIGN_IDENTITY:-${EXPANDED_CODE_SIGN_IDENTITY:-}}"
if [[ -z "$IDENTITY" ]]; then
  echo "🔓  No signing identity supplied – will output *UNSIGNED* XCFramework"
else
  echo "🔐  Using signing identity: $IDENTITY"
fi
echo

codesign_bin() {
  local bin="$1"
  if [[ -z "$IDENTITY" ]]; then return; fi      # skip if unsigned build
  if /usr/bin/codesign --force --options=runtime --timestamp \
                       --sign "$IDENTITY" "$bin" &>/dev/null; then
    echo "✅  codesigned  $(basename "$bin")"
  else
    echo "❌  failed to sign $(basename "$bin")" >&2
    exit 1
  fi
}

strip_signature() {
  /usr/bin/codesign --remove-signature "$1" 2>/dev/null || true
}

###############################################################################
# 📂  BUILD DIRS
###############################################################################
TMP_DIR="build/xcframework_temp"
OUTPUT_DIR="dist"
OUTPUT_XCFW="${OUTPUT_DIR}/${FRAMEWORK_NAME}.xcframework"

rm -rf "$TMP_DIR" "$OUTPUT_XCFW"
mkdir -p "$TMP_DIR" "$OUTPUT_DIR"

IOS_DEVICE_FW="$TMP_DIR/ios-arm64/${FRAMEWORK_NAME}.framework"
IOS_SIM_FW="$TMP_DIR/ios-arm64_x86_64-simulator/${FRAMEWORK_NAME}.framework"
MACOS_FW="$TMP_DIR/macos-arm64/${FRAMEWORK_NAME}.framework"

EXECUTABLE_NAME="$FRAMEWORK_NAME"
INSTALL_NAME="@rpath/${FRAMEWORK_NAME}.framework/$EXECUTABLE_NAME"

###############################################################################
# 🔧  HELPER: build one slice
###############################################################################
build_slice() {
  local TARGET_DIR=$1 PLATFORM=$2 MIN_OS=$3 MAIN_LIB=$4 DEP_DIR=$5

  echo "▶️  Building slice: $PLATFORM"
  mkdir -p "$TARGET_DIR/Headers" "$TARGET_DIR/Frameworks"

  # main dylib
  cp "$MAIN_LIB" "$TARGET_DIR/$EXECUTABLE_NAME"
  install_name_tool -id "$INSTALL_NAME" "$TARGET_DIR/$EXECUTABLE_NAME"

  # dependencies
  for DEP in "${DEPENDENCY_LIBS[@]}"; do
    [[ -f "$DEP_DIR/$DEP" ]] || { echo "⚠️   missing $DEP in $DEP_DIR"; continue; }
    cp "$DEP_DIR/$DEP" "$TARGET_DIR/Frameworks/"
    install_name_tool -change "@rpath/$DEP" "@loader_path/Frameworks/$DEP" \
                      "$TARGET_DIR/$EXECUTABLE_NAME"
  done

  # headers
  for H in "${HEADER_SOURCE_DIRS[@]}"; do cp -R "$H/." "$TARGET_DIR/Headers/"; done

  # Info.plist
  sed -e "s/__NAME__/${FRAMEWORK_NAME}/g" \
      -e "s/__EXECUTABLE__/${EXECUTABLE_NAME}/g" \
      -e "s/__IDENTIFIER__/${FRAMEWORK_ID}/g" \
      -e "s/__VERSION__/${FRAMEWORK_VERSION}/g" \
      -e "s/__MIN_OS_VERSION__/${MIN_OS}/g" \
      "$PLIST_TEMPLATE" > "$TARGET_DIR/Info.plist"

  # ✂️  strip old sigs, 🔏  re-sign
  strip_signature "$TARGET_DIR/$EXECUTABLE_NAME"
  codesign_bin     "$TARGET_DIR/$EXECUTABLE_NAME"
  for DEP in "${DEPENDENCY_LIBS[@]}"; do
    [[ -f "$TARGET_DIR/Frameworks/$DEP" ]] || continue
    strip_signature "$TARGET_DIR/Frameworks/$DEP"
    codesign_bin     "$TARGET_DIR/Frameworks/$DEP"
  done
  echo
}

###############################################################################
# 🛠  BUILD SLICES
###############################################################################
build_slice "$IOS_DEVICE_FW" "iOS Device arm64" \
            "$MIN_IOS_VERSION" "$IOS_DEVICE_LIB_DIR/$MAIN_LIB_NAME" "$IOS_DEVICE_LIB_DIR"

build_slice "$MACOS_FW" "macOS arm64" \
            "$MIN_MACOS_VERSION" "$MACOS_ARM64_LIB_DIR/$MAIN_LIB_NAME" "$MACOS_ARM64_LIB_DIR"

# iOS Simulator (fat)
echo "▶️  Building slice: iOS Simulator universal"
mkdir -p "$IOS_SIM_FW/Headers" "$IOS_SIM_FW/Frameworks"
lipo -create "$IOS_SIM_ARM64_LIB_DIR/$MAIN_LIB_NAME" \
              "$IOS_SIM_X86_64_LIB_DIR/$MAIN_LIB_NAME" \
     -output "$IOS_SIM_FW/$EXECUTABLE_NAME"
install_name_tool -id "$INSTALL_NAME" "$IOS_SIM_FW/$EXECUTABLE_NAME"
for DEP in "${DEPENDENCY_LIBS[@]}"; do
  [[ -f "$IOS_SIM_ARM64_LIB_DIR/$DEP" && -f "$IOS_SIM_X86_64_LIB_DIR/$DEP" ]] || continue
  lipo -create "$IOS_SIM_ARM64_LIB_DIR/$DEP" "$IOS_SIM_X86_64_LIB_DIR/$DEP" \
       -output "$IOS_SIM_FW/Frameworks/$DEP"
  install_name_tool -change "@rpath/$DEP" "@loader_path/Frameworks/$DEP" \
                    "$IOS_SIM_FW/$EXECUTABLE_NAME"
done
for H in "${HEADER_SOURCE_DIRS[@]}"; do cp -R "$H/." "$IOS_SIM_FW/Headers/"; done
sed -e "s/__NAME__/${FRAMEWORK_NAME}/g" \
    -e "s/__EXECUTABLE__/${EXECUTABLE_NAME}/g" \
    -e "s/__IDENTIFIER__/${FRAMEWORK_ID}/g" \
    -e "s/__VERSION__/${FRAMEWORK_VERSION}/g" \
    -e "s/__MIN_OS_VERSION__/${MIN_IOS_VERSION}/g" \
    "$PLIST_TEMPLATE" > "$IOS_SIM_FW/Info.plist"
strip_signature "$IOS_SIM_FW/$EXECUTABLE_NAME";   codesign_bin "$IOS_SIM_FW/$EXECUTABLE_NAME"
for DEP in "${DEPENDENCY_LIBS[@]}"; do
  [[ -f "$IOS_SIM_FW/Frameworks/$DEP" ]] || continue
  strip_signature "$IOS_SIM_FW/Frameworks/$DEP"; codesign_bin "$IOS_SIM_FW/Frameworks/$DEP"
done
echo

###############################################################################
# 🎁  CREATE XCFRAMEWORK
###############################################################################
echo "📦  Assembling XCFramework…"
xcodebuild -quiet -create-xcframework \
  -framework "$IOS_DEVICE_FW" \
  -framework "$IOS_SIM_FW" \
  -framework "$MACOS_FW" \
  -output "$OUTPUT_XCFW"
echo "✅  XCFramework written to $OUTPUT_XCFW"
echo

###############################################################################
# 🔍  FINAL VERIFICATION TABLE
###############################################################################
printf "┌%-38s┬%-12s┐\n" " Binary " "Team ID"
for BIN in \
  "$OUTPUT_XCFW"/macos-arm64/"$FRAMEWORK_NAME".framework/$EXECUTABLE_NAME \
  "$OUTPUT_XCFW"/macos-arm64/"$FRAMEWORK_NAME".framework/Frameworks/*.dylib
do
  TEAM=$(/usr/bin/codesign -dv "$BIN" 2>&1 | grep -Eo 'TeamIdentifier=[A-Z0-9]+' || echo "TeamIdentifier=adhoc")
  printf "│ %-37s│ %-11s│\n" "$(basename "$BIN")" "${TEAM#TeamIdentifier=}"
done
printf "└%s┴%s┘\n" "$(printf '─%.0s' {1..38})" "$(printf '─%.0s' {1..12})"

[[ -z "$IDENTITY" ]] \
  && echo "🔓  Framework is UNSIGNED; Xcode will sign it when you embed it." \
  || echo "🔏  All binaries signed by: $IDENTITY"