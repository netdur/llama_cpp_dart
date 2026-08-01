# Example probes

The scripts in `example/probes/` are small command-line programs for testing
individual parts of `llama_cpp_dart` without creating a Flutter application.

Build the macOS libraries first:

```bash
tool/build_native.sh --platform macos --with-mtmd --clean
```

Then set the library and model used by most probes:

```bash
export LLAMA_CPP_DART_LIB="$PWD/build/macos/install/lib/libllama.dylib"
export LLAMA_CPP_DART_MODEL="/path/to/model.gguf"
```

Common probes:

```bash
# Package, llama.cpp revision, and runtime information
dart run example/probes/version_info.dart

# Stream a completion through the isolate engine
dart run example/probes/engine_stream.dart "Once upon a time"

# Run two chat turns using the model's embedded template
dart run example/probes/chat_two_turn.dart

# Inspect GGUF model metadata
dart run example/probes/inspect_model.dart

# Verify tokenization and detokenization
dart run example/probes/tokenize_round_trip.dart "Hello, world"

# List the native backends visible at runtime
dart run example/probes/list_backends.dart
```

Multimodal probes additionally require a projector:

```bash
export LLAMA_CPP_DART_MMPROJ="/path/to/mmproj.gguf"
dart run example/probes/vision_describe.dart /path/to/image.jpg
dart run example/probes/audio_describe.dart /path/to/audio.wav
dart run example/probes/video_describe.dart /path/to/video.mp4
```

Every probe documents its accepted arguments and environment variables at the
top of its source file. Use a release library instead of a local build by
pointing `LLAMA_CPP_DART_LIB` at the downloaded `libllama` artifact.
