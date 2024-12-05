# llama_cpp - Generated FFI Bindings

The `llama_cpp` class provides low-level FFI bindings to llama.cpp's C API, enabling direct access to model inference capabilities. This approach gives fine-grained control over model loading, context management, and token generation.

## Example Usage

```dart
import 'package:llama_cpp_dart/src/llama_cpp.dart';

void main() {
  final lib = llama_cpp(DynamicLibrary.open("path/to/libllama.dylib"));
  lib.llama_backend_init();
  
  // Load model with default parameters
  var modelParams = lib.llama_model_default_params();
  final model = lib.llama_load_model_from_file(modelPathPtr, modelParams);
  
  // Initialize context
  var ctxParams = lib.llama_context_default_params();
  final ctx = lib.llama_new_context_with_model(model, ctxParams);
  
  // Process tokens and generate response
  // [See full example for token processing and generation logic]
  
  // Cleanup
  lib.llama_free(ctx);
  lib.llama_free_model(model);
  lib.llama_backend_free();
}
```

See [example/simple.dart](../example/simple.dart) for a complete implementation showing token-by-token text generation using the raw FFI bindings.