# llama_cpp - Generated FFI Bindings

The `llama_cpp` class provides low-level **dart:ffi** bindings to `llama.cpp`'s C API. It serves as the bridge between Dart and the native C++ library.

**Note:** Most users should use the high-level `Llama` class instead of using this directly. Use this only if you need experimental features or fine-grained control over the C lifecycle.

## ⚠️ ABI Compatibility Warning

This file contains manual struct definitions (like `llama_context_params` and `llama_model_params`). These **MUST** match the binary layout of the `libllama.so` / `libllama.dylib` you are loading.

If you update the shared library to a newer version of `llama.cpp`, you must ensure the Dart struct definitions (order of fields and data types) match the C header `llama.h` exactly. Mismatches will cause **Segmentation Faults**.

## Example Usage (Raw FFI)

```dart
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:llama_cpp_dart/src/llama_cpp.dart';

void main() {
  // 1. Load the dynamic library
  final lib = llama_cpp(DynamicLibrary.open("path/to/libllama.dylib"));
  
  // 2. Initialize backend
  lib.llama_backend_init();
  
  // 3. Load Model
  final modelParams = lib.llama_model_default_params();
  final modelPath = "model.gguf".toNativeUtf8().cast<Char>();
  
  final model = lib.llama_load_model_from_file(modelPath, modelParams);
  if (model == nullptr) {
    print("Failed to load model");
    return;
  }
  
  // 4. Initialize Context
  final ctxParams = lib.llama_context_default_params();
  ctxParams.n_ctx = 2048;
  
  final ctx = lib.llama_new_context_with_model(model, ctxParams);
  if (ctx == nullptr) {
    print("Failed to create context");
    return;
  }
  
  // ... Perform low-level tokenization, batch creation, and decoding ...
  
  // 5. Cleanup
  malloc.free(modelPath);
  lib.llama_free(ctx);
  lib.llama_free_model(model);
  lib.llama_backend_free();
}
```

## Key Components

### Structs
- `llama_model_params`: Configuration for loading the model (GPU layers, split mode).
- `llama_context_params`: Configuration for the active context (Context size, Flash Attention, Cache Type).
- `llama_batch`: The standard container for submitting tokens to the model for decoding.

### Functions
- `llama_backend_init()`: Must be called once at startup.
- `llama_load_model_from_file()`: Loads weights into VRAM/RAM.
- `llama_new_context_with_model()`: Allocates KV Cache.
- `llama_decode()`: The core function that computes logits.
- `llama_get_logits()`: Retrieves raw output probabilities.