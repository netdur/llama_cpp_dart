# Examples & Subsystem Probes

Runnable Dart CLI scripts demonstrating individual subsystems of `llama_cpp_dart`.

## Environment Setup

All probes require the compiled `Llama` native library and a GGUF model binary path passed via environment variables:

- **`LLAMA_CPP_DART_LIB`**: Path to the native dynamic library (e.g. `./build/apple/build-macos-universal/framework/Llama.framework/Llama` or `libllama.so` / `llama.dll`).
- **`LLAMA_CPP_DART_MODEL`**: Path to a quantized `.gguf` model file.

---

## Running Examples

### 1. Version & Runtime Inspection (No model required)
```bash
dart run example/probes/version_info.dart
```

### 2. Single-Turn Text Generation
Stream tokens directly from `LlamaEngine`:
```bash
LLAMA_CPP_DART_LIB="./build/apple/build-macos-universal/framework/Llama.framework/Llama" \
LLAMA_CPP_DART_MODEL="/path/to/model.gguf" \
dart run example/probes/engine_stream.dart "Once upon a time"
```

### 3. Multi-Turn Interactive Chat
Maintain conversation history and system prompts using embedded Jinja chat templates:
```bash
LLAMA_CPP_DART_LIB="./build/apple/build-macos-universal/framework/Llama.framework/Llama" \
LLAMA_CPP_DART_MODEL="/path/to/model.gguf" \
dart run example/probes/chat_two_turn.dart
```

### 4. Model Metadata & Inspection
Print GGUF metadata key-value pairs, tensor counts, and context limits:
```bash
LLAMA_CPP_DART_MODEL="/path/to/model.gguf" \
dart run example/probes/inspect_model.dart
```

---

> **Note for Intel Macs (`x86_64`):** If your Dart SDK runs under Intel CPU or Rosetta, ensure the native library slice target is built for `x86_64` or `universal` (`arm64;x86_64`) architecture.
