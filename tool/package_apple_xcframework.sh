#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

xcframework="$PROJECT_ROOT/build/apple/llama.xcframework"
output="$PROJECT_ROOT/llama-xcframework.zip"

usage() {
  cat <<'EOF'
Usage: tool/package_apple_xcframework.sh [options]

Package an Apple XCFramework for GitHub Releases and Swift Package Manager.

Options:
  --xcframework PATH  XCFramework to package
                      (default: build/apple/llama.xcframework)
  --output PATH       Destination zip
                      (default: llama-xcframework.zip)
  -h, --help          Show this help

The script preserves framework symlinks, verifies the extracted archive and
code signatures, and writes a matching <output>.sha256 file. It also prints
the checksum to copy into Package.swift.
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --xcframework)
      [[ $# -ge 2 ]] || fail "--xcframework requires a path"
      xcframework="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || fail "--output requires a path"
      output="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

if [[ "$xcframework" != /* ]]; then
  xcframework="$PROJECT_ROOT/$xcframework"
fi
if [[ "$output" != /* ]]; then
  output="$PROJECT_ROOT/$output"
fi

[[ -d "$xcframework" ]] || fail "XCFramework not found: $xcframework"
[[ -f "$xcframework/Info.plist" ]] || fail "missing XCFramework Info.plist"
[[ "$output" == *.zip ]] || fail "output must have a .zip extension"

for command_name in ditto plutil codesign swift find; do
  command -v "$command_name" >/dev/null 2>&1 || \
    fail "required command not found: $command_name"
done

package_temp="$(mktemp -d)"
cleanup() {
  rm -rf "$package_temp"
}
trap cleanup EXIT

archive_name="$(basename "$output")"
temporary_archive="$package_temp/$archive_name"
extracted_root="$package_temp/extracted"
mkdir -p "$extracted_root"

# Unlike `zip -r`, ditto preserves the symlinks used by versioned macOS
# frameworks. Flattening those links makes the consuming app fail strict
# code-signature validation.
ditto -c -k --sequesterRsrc --keepParent \
  "$xcframework" "$temporary_archive"
ditto -x -k "$temporary_archive" "$extracted_root"

extracted_xcframework="$extracted_root/$(basename "$xcframework")"
[[ -f "$extracted_xcframework/Info.plist" ]] || \
  fail "archive does not contain $(basename "$xcframework")/Info.plist"
plutil -lint "$extracted_xcframework/Info.plist" >/dev/null

framework_list="$package_temp/frameworks.txt"
find "$extracted_xcframework" -type d -name '*.framework' -print > "$framework_list"
[[ -s "$framework_list" ]] || fail "archive does not contain any frameworks"

while IFS= read -r framework; do
  if [[ -d "$framework/Versions" && ! -L "$framework/Versions/Current" ]]; then
    fail "versioned framework symlink was flattened: $framework/Versions/Current"
  fi
  codesign --verify --deep --strict --verbose=2 "$framework"
done < "$framework_list"

checksum="$(swift package compute-checksum "$temporary_archive")"
checksum_file="$temporary_archive.sha256"
printf '%s  %s\n' "$checksum" "$archive_name" > "$checksum_file"

mkdir -p "$(dirname "$output")"
mv -f "$temporary_archive" "$output"
mv -f "$checksum_file" "$output.sha256"

echo "Packaged: $output"
echo "Checksum file: $output.sha256"
echo "SwiftPM checksum: $checksum"
