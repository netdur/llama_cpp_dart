#!/usr/bin/env bash
# Verify that every ELF LOAD segment in an Android AAR is aligned for 16 KB
# page-size devices.
#
# Usage: tool/check_android_aar_alignment.sh AAR_PATH [LLVM_READELF]

set -euo pipefail

AAR_PATH="${1:-}"
READELF="${2:-llvm-readelf}"
MIN_ALIGNMENT=$((16 * 1024))

if [[ -z "$AAR_PATH" || ! -f "$AAR_PATH" ]]; then
  echo "error: AAR not found: ${AAR_PATH:-<missing argument>}" >&2
  exit 2
fi
if [[ "$READELF" == */* ]]; then
  if [[ ! -x "$READELF" ]]; then
    echo "error: llvm-readelf is not executable: $READELF" >&2
    exit 2
  fi
elif ! command -v "$READELF" >/dev/null 2>&1; then
  echo "error: llvm-readelf not found: $READELF" >&2
  exit 2
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "error: unzip is required" >&2
  exit 2
fi

CHECK_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$CHECK_DIR"
}
trap cleanup EXIT

unzip -q "$AAR_PATH" 'jni/*.so' -d "$CHECK_DIR"

so_count=0
failed=0
while IFS= read -r -d '' so_path; do
  so_count=$((so_count + 1))
  load_count=0
  headers="$("$READELF" -lW "$so_path")"

  while IFS= read -r alignment; do
    load_count=$((load_count + 1))
    if (( alignment < MIN_ALIGNMENT )); then
      printf 'error: %s LOAD segment %d has alignment %s; expected at least 0x4000\n' \
        "${so_path#"$CHECK_DIR"/}" "$load_count" "$alignment" >&2
      failed=1
    fi
  done < <(awk '$1 == "LOAD" { print $NF }' <<<"$headers")

  if (( load_count == 0 )); then
    echo "error: no ELF LOAD segments found in ${so_path#"$CHECK_DIR"/}" >&2
    failed=1
  fi
done < <(find "$CHECK_DIR/jni" -type f -name '*.so' -print0)

if (( so_count == 0 )); then
  echo "error: no native libraries found under jni/ in $AAR_PATH" >&2
  exit 1
fi
if (( failed != 0 )); then
  exit 1
fi

echo "verified $so_count native libraries: all ELF LOAD segments are 16 KB aligned"
