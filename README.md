# LLAMA.CPP DART

A high-performance Dart binding for llama.cpp, enabling advanced text generation capabilities in both Dart and Flutter applications with flexible integration options.

## Overview

This library provides three levels of abstraction for integrating llama.cpp into your Dart/Flutter projects, allowing you to choose the right balance between control and convenience:

1. **Low-Level FFI Bindings**: Direct access to llama.cpp functions
2. **High-Level Wrapper**: Simplified, object-oriented API
3. **Managed Isolate**: Flutter-friendly, non-blocking implementation

## Usage Examples

### Low-Level FFI Bindings
Direct llama.cpp integration with maximum control:
```dart
import 'package:llama_cpp_dart/src/llama_cpp.dart';

void main() {
  final lib = llama_cpp(DynamicLibrary.open("libllama.dylib"));
  // Initialize model, context, and sampling parameters
  // See examples/low_level.dart for complete example
}
```

check examples:
- [simple](example/simple.dart)
- [embedding](example/embedding_raw.dart)

### High-Level Wrapper
Simplified API for common use cases:
```dart
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  Llama.libraryPath = "libllama.dylib";
  final llama = Llama("path/to/model.gguf");
  
  llama.setPrompt("2 * 2 = ?");
  while (true) {
    var (token, done) = llama.getNext();
    print(token);
    if (done) break;
  }
  llama.dispose();
}
```

check examples:
- [test](example/test.dart)
- [rag](example/rag.dart)
- [chat](example/chat_cli.dart)


### Managed Isolate
Perfect for Flutter applications:
```dart
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  final loadCommand = LlamaLoad(
    path: "path/to/model.gguf",
    modelParams: ModelParams(),
    contextParams: ContextParams(),
    samplingParams: SamplerParams(),
    format: ChatMLFormat(),
  );

  final llamaParent = LlamaParent(loadCommand);
  await llamaParent.init();

  llamaParent.stream.listen((response) => print(response));
  llamaParent.sendPrompt("2 * 2 = ?");
}
```

check examples:
- [test](example/test_isolated.dart)
- [chat](example/chat_cli_isolated.dart)

## Getting Started

### Prerequisites
- Dart SDK (for console applications)
- Flutter SDK (for Flutter applications)
- Compiled llama.cpp shared library

### Building llama.cpp Library

1. Clone the llama.cpp repository:
```bash
git clone https://github.com/ggml-org/llama.cpp
```

2. Compile into a shared library:
- Windows: Outputs .dll
- Linux: Outputs .so
- macOS: Outputs .dylib

check [BUILD.md](BUILD.md)

3. Place the compiled library in your project's accessible directory

## Installation

Add to your `pubspec.yaml`:
```yaml
dependencies:
  llama_cpp_dart: ^latest_version
```

## Model Selection Guide

When choosing and using LLM models with this library, consider the following:

### Use-Case Specific Models

Different models excel at different tasks:

- **Text Generation**: Most LLMs work well for general text generation.
- **Embeddings**: Not all models produce high-quality embeddings for semantic search. For example, while Gemma 3 can generate embeddings, it's not optimized for vector search. Instead, consider dedicated embedding models like E5, BGE, or SGPT.
- **Code Generation**: Models like CodeLlama or StarCoder are specifically trained for code.
- **Multilingual**: Some models have better support for non-English languages.

### Chat Formats

Each model family expects prompts in a specific format:

- **Llama 2**: Uses a specific format with `[INST]` and `[/INST]` tags
- **ChatML**: Used by models like Claude and GPT
- **Gemma**: Has its own system prompt format
- **Mistral/Mixtral**: Uses `<s>` tags in a particular way

Using the correct format is critical for optimal results. Our library provides common format templates:

```dart
// Example of setting the right chat format
final loadCommand = LlamaLoad(
  path: "path/to/llama2.gguf",
  format: Llama2ChatFormat(), // Choose the correct format for your model
);

// Other available formats
// ChatMLFormat()
// GemmaChatFormat()
// MistralChatFormat()
// Custom formats can be created by implementing the ChatFormat interface
```

### Model Size Considerations

Balance quality and performance:

- **7B models**: Fastest, lowest memory requirements, but less capable
- **13-14B models**: Good balance of performance and quality
- **30-70B models**: Highest quality, but significantly higher memory and processing requirements

### Quantization

Models come in different quantization levels that affect size, speed, and quality:

- **F16**: Highest quality, largest size
- **Q4_K_M**: Good balance of quality and size
- **Q3_K_M**: Smaller size, slightly reduced quality
- **Q2_K**: Smallest size, noticeable quality degradation

For most applications, Q4_K_M provides an excellent balance.

### Hardware Considerations

- **CPU**: All models work on CPU, but larger models require more RAM
- **Metal (Apple)**: Significant speed improvements on Apple Silicon
- **CUDA (NVIDIA)**: Best performance for NVIDIA GPUs
- **ROCm (AMD)**: Support for AMD GPUs

Ensure your compiled llama.cpp library includes support for your target hardware.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.