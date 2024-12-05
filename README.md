# LLAMA.CPP DART

A high-performance Dart binding for llama.cpp, enabling advanced text generation capabilities in both Dart and Flutter applications with flexible integration options.

## Overview

This library provides three levels of abstraction for integrating llama.cpp into your Dart/Flutter projects, allowing you to choose the right balance between control and convenience:

1. **Low-Level FFI Bindings**: Direct access to llama.cpp functions
2. **High-Level Wrapper**: Simplified, object-oriented API
3. **Managed Isolate**: Flutter-friendly, non-blocking implementation

## Features

- Asynchronous text generation using Dart isolates
- Flexible configuration through customizable parameters
- Multiple integration approaches to suit different needs
- Real-time text generation with stream-based output
- Support for different prompt formats (ChatML, Alpaca)
- Comprehensive parameter control for model, context, and sampling

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

## Getting Started

### Prerequisites
- Dart SDK (for console applications)
- Flutter SDK (for Flutter applications)
- Compiled llama.cpp shared library

### Building llama.cpp Library

1. Clone the llama.cpp repository:
```bash
git clone https://github.com/ggerganov/llama.cpp
```

2. Compile into a shared library:
- Windows: Outputs .dll
- Linux: Outputs .so
- macOS: Outputs .dylib

3. Place the compiled library in your project's accessible directory

## Installation

Add to your `pubspec.yaml`:
```yaml
dependencies:
  llama_cpp_dart: ^latest_version
```

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.