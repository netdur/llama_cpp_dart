# Releasing `llama_cpp_dart`

Maintainer runbook. Validated end-to-end on `0.9.0-dev.12`.

Two steps in this sequence are ordered counter-intuitively and have each burned
a release. Read [Ordering traps](#ordering-traps) before deviating.

## Overview

```
bump + changelog → commit → push → tag → CI builds artifacts
    → re-pin Package.swift → stage AAR → dart pub publish
```

Everything after the tag depends on artifacts CI produces *from* that tag, so
the last three steps cannot be folded into the release commit.

## 1. Prepare the release commit

```bash
tool/generate_version.sh                      # regenerates lib/src/version.dart
# edit pubspec.yaml:  version: 0.9.0-dev.N
# add a CHANGELOG.md section
```

`lib/src/version.dart` is generated from `pubspec.yaml` + the `src/llama.cpp`
submodule SHA. CI regenerates it and fails on any drift, so run the script
*after* bumping the version, not before.

Run the same gates CI will, so a red tag is impossible:

```bash
dart analyze
dart format --output=none --set-exit-if-changed lib/ test/
dart test test/utf8_accumulator_test.dart \
          test/mobile_chat_api_test.dart \
          test/android_native_assets_test.dart
```

If you touched the engine, also run the model-backed suite (needs a local
dylib and any GGUF). Use `-j 1` — parallel runs still hit the cross-isolate
log-callback race:

```bash
LLAMA_CPP_DART_LIB="$PWD/build/macos/install/lib/libllama.dylib" \
LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
  dart test -j 1 test/engine_test.dart test/smoke_test.dart test/chat_test.dart
```

Then commit and push:

```bash
git commit -am "release: 0.9.0-dev.N"
git push origin main
```

## 2. Tag

```bash
git tag -a v0.9.0-dev.N -m "0.9.0-dev.N: <summary>"
git push origin v0.9.0-dev.N
```

All three workflows trigger on `push: tags: 'v*'`:

| Workflow | Produces |
|---|---|
| `test` | analyze, format check, `version.dart` drift check, pure-Dart tests |
| `build-android` | `llama-cpp-dart.aar`, `llama-cpp-dart-hexagon.aar` (+ `.sha256`) |
| `build-apple` | `llama-xcframework.zip`, `macos-libllama.zip` (+ `.sha256`) |

`build-android` additionally gates on 16 KB ELF alignment, builds a throwaway
Flutter app and asserts all five `.so` files reach the APK, and greps
`pub publish --dry-run` so the AAR cannot silently drop out of the package.

Wait for green before continuing:

```bash
gh run list --limit 3
gh run watch <run-id> --exit-status
```

## 3. Re-pin the SwiftPM manifest

`darwin/llama_cpp_dart/Package.swift` pins a binary target to an immutable
release asset, so it can only be updated once CI has uploaded it.

```bash
gh release download v0.9.0-dev.N --pattern 'llama-xcframework.zip' --dir /tmp
swift package compute-checksum /tmp/llama-xcframework.zip
```

Update both the `url` (new tag) and `checksum` in `Package.swift`, verify, then
commit:

```bash
swift package dump-package --package-path darwin/llama_cpp_dart >/dev/null
git commit -am "build(apple): pin the SwiftPM binary target to v0.9.0-dev.N"
git push origin main
```

## 4. Stage the Android AAR

`native/android/*.aar` is gitignored but deliberately **not** in `.pubignore` —
the published package is what carries it to consumers. It must exist locally
before publishing.

```bash
gh release download v0.9.0-dev.N --pattern 'llama-cpp-dart.aar' --dir /tmp
tool/stage_android_native_assets.sh /tmp/llama-cpp-dart.aar \
  "$ANDROID_SDK/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-readelf"
```

The script re-verifies 16 KB alignment before staging.

## 5. Publish

Confirm the three things that have gone missing before:

```bash
dart pub publish --dry-run | grep -E 'aar|Package.swift|hook/'
```

Expect `hook/build.dart`, `darwin/llama_cpp_dart/Package.swift`, and
`native/android/llama-cpp-dart.aar`. Then:

```bash
dart pub publish
```

Verify what actually shipped, not just that upload succeeded:

```bash
curl -sL https://pub.dev/api/archives/llama_cpp_dart-0.9.0-dev.N.tar.gz \
  | tar tz | grep -E '^(hook|native|darwin)/'
```

## Ordering traps

**`Package.swift` is necessarily pinned *after* the tag it references.** CI
builds `llama-xcframework.zip` from the tag, so its checksum does not exist
until afterward. Do **not** re-tag to fold the pin in: that rebuilds and
replaces the asset, yielding a new checksum and invalidating the pin again —
an infinite loop. The value that reaches consumers is the `Package.swift`
inside the published pub.dev package, not the one in the git tag. A tag tree
whose manifest points at the previous release is expected and harmless.

**pub.dev accepts only `build.dart` and `link.dart` under `hook/`.** Any other
file there is rejected at upload:

```
Message from server: Hook files are experimental and
`hook/android_native_assets.dart` is not allowed yet.
```

Helpers belong in `lib/src/hook/`, imported by package URI. This surfaced only
at the upload step of `0.9.0-dev.11` — after the tag was cut and all artifacts
were built — forcing a bump to `dev.12`. If a release is burned this way, the
version number is not consumed on pub.dev (nothing was published), but the tag
and GitHub release remain and should be deleted.

## Checklist

- [ ] `tool/generate_version.sh` run after the version bump
- [ ] CHANGELOG section added
- [ ] `dart analyze`, format check, pure-Dart tests pass locally
- [ ] release commit pushed to `main`
- [ ] tag pushed; all three workflows green
- [ ] `Package.swift` re-pinned to this tag's xcframework + checksum verified
- [ ] AAR staged into `native/android/`
- [ ] `--dry-run` shows `hook/build.dart`, `Package.swift`, and the AAR
- [ ] published, and the published archive re-checked
