#!/usr/bin/env bash
# Build a temporary Flutter app and verify that native assets package the AAR.
# The verified AAR must first be staged with stage_android_native_assets.sh.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGED_AAR="$PROJECT_ROOT/native/android/llama-cpp-dart.aar"

if [[ ! -f "$STAGED_AAR" ]]; then
  echo "error: staged AAR not found: $STAGED_AAR" >&2
  echo "run tool/stage_android_native_assets.sh first" >&2
  exit 2
fi
if ! command -v flutter >/dev/null 2>&1; then
  echo "error: flutter is required" >&2
  exit 2
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "error: unzip is required" >&2
  exit 2
fi

TEST_ROOT="$(mktemp -d)"
cleanup() {
  if [[ "${LLAMA_KEEP_ANDROID_PROBE:-0}" == "1" ]]; then
    echo "kept Android integration probe at $TEST_ROOT"
  else
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT
APP_DIR="$TEST_ROOT/android_native_assets_probe"

flutter create \
  --empty \
  --platforms=android \
  --org=dev.netdur \
  --project-name=android_native_assets_probe \
  "$APP_DIR"

flutter pub add \
  --directory "$APP_DIR" \
  --no-example \
  "llama_cpp_dart@{path: $PROJECT_ROOT}"

(
  cd "$APP_DIR"
  flutter build apk \
    --debug \
    --target-platform=android-arm64
)

APK="$APP_DIR/build/app/outputs/flutter-apk/app-debug.apk"
APK_ENTRIES="$(unzip -Z1 "$APK")"
echo "native APK entries:"
grep '^lib/' <<<"$APK_ENTRIES" || true
for library in libllama.so libmtmd.so libggml.so libggml-base.so libggml-cpu.so; do
  if ! grep -qx "lib/arm64-v8a/$library" <<<"$APK_ENTRIES"; then
    echo "error: $library was not packaged in $APK" >&2
    exit 1
  fi
done

echo "verified Flutter Android native-assets packaging: $APK"
