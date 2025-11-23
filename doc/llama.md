# Llama Class Documentation

A Dart wrapper for `llama.cpp` that provides high-performance text generation, embedding extraction, and multimodal capabilities using GGUF models.

## Core Components

### Status
- `LlamaStatus`: Enum tracking instance state
  - `uninitialized`: Initial state
  - `loading`: Model is loading from disk
  - `ready`: Idle and ready for generation
  - `generating`: Currently processing/generating text
  - `error`: Error state
  - `disposed`: Resources freed

### Initialization (v0.2 Update)
The constructor now uses **Named Parameters**.

```dart
Llama(
  String modelPath, {
  ModelParams? modelParams,
  ContextParams? contextParams,
  SamplerParams? samplerParams,
  String? mmprojPath, // Path to multimodal projector (for vision)
  bool verbose = false,
})
```

## Core Methods

### Text Generation
The class is **Stateful**. `setPrompt` appends tokens to the current context window in VRAM.

```dart
void setPrompt(String prompt, {void Function(int current, int total)? onProgress})
```
Tokenizes and evaluates the prompt. Appends to existing context unless `clear()` is called.

```dart
Stream<String> generateText()
```
Returns a stream of generated tokens. Automatically handles UTF-8 decoding (including split emojis) and stops at EOS.

```dart
Future<String> generateCompleteText({int? maxTokens})
```
Convenience method to generate a full response as a single string.

### Multimodal (Vision)
```dart
Stream<String> generateWithMedia(String prompt, {required List<LlamaInput> inputs})
```
Generates text based on an image + text prompt. Requires `mmprojPath` to be set during initialization.

### Embeddings
```dart
List<double> getEmbeddings(String prompt, {bool normalize = true})
```
Calculates vector embeddings for the given text.
*   *Note:* This temporarily uses the active context slot.

## Advanced Features (New in v0.2)

### Multi-Tenancy (Context Slots)
Manage multiple independent conversations in VRAM using a single loaded model.

```dart
void createSlot(String slotId)
```
Allocates a new KV cache (context) in VRAM for a user.

```dart
void setSlot(String slotId)
```
Swaps the active "brain" to the specified slot. Instant switching (pointer swap).

### State Management (Hot-Swap)
Move conversations between VRAM and RAM/Disk.

```dart
Uint8List saveState()
```
Snapshots the current KV cache (Brain) to a RAM byte array.

```dart
void loadState(Uint8List stateData)
```
Restores the KV cache from a RAM byte array.

### Session Persistence (Disk)
Optimized for low memory usage (Direct Disk-to-C Streaming).

```dart
void saveSession(String path)
bool loadSession(String path)
```
Saves/Loads the context to a file on disk.

## Cleanup
```dart
void clear()
```
Wipes the current VRAM context (Amnesia). `_nPos` resets to 0.

```dart
void dispose()
```
Frees all native resources (Model, Contexts, Batch, Samplers). **Must be called.**

## Example Usage

### 1. Standard Chat
```dart
final llama = Llama(
  'model.gguf',
  contextParams: ContextParams()..nCtx = 2048,
  verbose: true,
);

try {
  // 1. User says hello
  llama.setPrompt("Hello!");
  await for (final text in llama.generateText()) stdout.write(text);

  // 2. User asks follow-up (Appends to VRAM automatically)
  llama.setPrompt("What did I just say?");
  await for (final text in llama.generateText()) stdout.write(text);

} finally {
  llama.dispose();
}
```

### 2. Context Swapping (Multi-User)
```dart
// Setup
llama.createSlot("user_1");
llama.createSlot("user_2");

// Handle User 1
llama.setSlot("user_1");
llama.setPrompt("My name is Alice");
await llama.generateCompleteText();

// Switch to User 2 (Instant)
llama.setSlot("user_2");
llama.setPrompt("My name is Bob");
await llama.generateCompleteText();

// Switch back to Alice (She is still in VRAM)
llama.setSlot("user_1");
llama.setPrompt("What is my name?"); 
// Output: "Alice"
```

## Best Practices
1.  **Lifecycle:** Wrap usage in `try/finally` to ensure `dispose()` is called.
2.  **UTF-8:** Do not manually decode tokens from `getNext()`. Use `generateText()` stream which handles multi-byte characters safely.
3.  **Performance:** Use `saveSession` instead of keeping the app open if the user is inactive for long periods.
4.  **Threads:** Set `ContextParams.nThreads` to match your CPU's physical performance cores (usually 4-8) for best speed.