# Android release asset

`llama-cpp-dart.aar` is staged here while testing or publishing a release. It
is intentionally ignored by Git: CI builds it from the pinned llama.cpp
submodule, verifies every ELF `LOAD` segment is 16 KB aligned, and then stages
it with:

```bash
tool/stage_android_native_assets.sh build/android/llama-cpp-dart.aar /path/to/llvm-readelf
```

Flutter runs `hook/build.dart`, extracts the libraries for each requested ABI,
and bundles them as native code assets. The release AAR currently contains
`arm64-v8a` CPU and mtmd libraries.
