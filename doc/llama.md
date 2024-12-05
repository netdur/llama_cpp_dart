# Llama Class Documentation

A Dart wrapper for llama.cpp that provides text generation capabilities using LLaMA models. This class handles model loading, context management, text generation, and resource cleanup.

## Core Components

### Status and State
- `LlamaStatus`: Enum tracking instance state
  - `uninitialized`: Initial state
  - `ready`: Ready for generation
  - `generating`: Currently generating text
  - `error`: Error state
  - `disposed`: Instance disposed

### Initialization
```dart
Llama(String modelPath, [
  ModelParams? modelParamsDart,
  ContextParams? contextParamsDart,
  SamplerParams? samplerParams
])
```
Creates a new Llama instance with specified parameters.

## Main Methods

### Text Generation
```dart
void setPrompt(String prompt, {void Function(int current, int total)? onProgress})
```
Sets the input prompt for text generation.

```dart
(String, bool) getNext()
```
Generates the next token, returns (generated text, is complete).

```dart
Stream<String> generateText()
```
Provides a stream of generated text tokens.

### Text Processing
```dart
List<int> tokenize(String text, bool addBos)
```
Converts text to token IDs.

### State Management
```dart
void clear()
```
Resets the instance state for new generation.

```dart
void dispose()
```
Releases all resources.

## Properties
```dart
LlamaStatus get status
bool get isDisposed
```

## Error Handling

### LlamaException
Custom exception class for Llama-specific errors:
```dart
class LlamaException implements Exception {
  final String message;
  final dynamic originalError;
}
```

## Example Usage

```dart
// Initialize Llama
final llama = Llama(
  'path/to/model.gguf',
  ModelParams(),
  ContextParams(),
  SamplerParams()
);

// Generate text
try {
  llama.setPrompt("Once upon a time");
  
  // Stream approach
  await for (final text in llama.generateText()) {
    print(text);
  }
  
  // Or step-by-step
  while (true) {
    final (text, isDone) = llama.getNext();
    if (isDone) break;
    print(text);
  }
} finally {
  llama.dispose();
}
```

## Memory Management
- Uses FFI for native library interaction
- Automatically manages native pointers
- Requires explicit `dispose()` call
- Includes safeguards against using disposed instances

## Notes
- Supports both Android (.so) and native platform libraries
- Handles model loading, tokenization, and text generation
- Provides comprehensive sampling parameter configuration
- Includes built-in error handling and status tracking

## Best Practices
1. Always dispose of instances when done
2. Use try-finally blocks for proper cleanup
3. Check instance status before operations
4. Handle LlamaException in generation code
5. Clear instance between different generation tasks