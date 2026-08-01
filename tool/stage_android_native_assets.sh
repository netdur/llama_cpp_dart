#!/usr/bin/env bash
# Validate an Android AAR and stage it for Flutter native-assets builds/pub.
#
# Usage: tool/stage_android_native_assets.sh AAR_PATH [LLVM_READELF]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_AAR="${1:-}"
READELF="${2:-llvm-readelf}"
DEST_DIR="$PROJECT_ROOT/native/android"
DEST_AAR="$DEST_DIR/llama-cpp-dart.aar"

if [[ -z "$SOURCE_AAR" || ! -f "$SOURCE_AAR" ]]; then
  echo "error: AAR not found: ${SOURCE_AAR:-<missing argument>}" >&2
  exit 2
fi

"$PROJECT_ROOT/tool/check_android_aar_alignment.sh" "$SOURCE_AAR" "$READELF"

mkdir -p "$DEST_DIR"
if [[ "$(cd "$(dirname "$SOURCE_AAR")" && pwd)/$(basename "$SOURCE_AAR")" != "$DEST_AAR" ]]; then
  cp "$SOURCE_AAR" "$DEST_AAR"
fi
(
  cd "$DEST_DIR"
  shasum -a 256 "$(basename "$DEST_AAR")" > "$(basename "$DEST_AAR").sha256"
)

echo "staged $DEST_AAR"
cat "$DEST_AAR.sha256"
