# Isolate & Async Architecture Documentation

The high-level API that runs `llama.cpp` in a background `Isolate` to keep the UI thread jank-free. It handles asynchronous communication, request queuing, and state management.

## Core Components

### LlamaParent
The main controller class. It spawns the background isolate, manages the message queue, and broadcasts generated text.

#### Initialization
```dart
final parent = LlamaParent(
  LlamaLoad(
    path: "model.gguf",
    modelParams: ModelParams(),
    contextParams: ContextParams(),
    samplerParams: SamplerParams(),
  )
);

await parent.init(); // Spawns isolate and loads model
```

#### Key Properties
- `stream` (`Stream<String>`): Global stream of all generated text.
- `isGenerating` (`bool`): Whether the model is currently busy.
- `messages` (`List<Map<String, dynamic>>`): Chat history used by the internal formatter.
- `status` (`LlamaStatus`): Current health of the background worker.

#### Core Methods
- `sendPrompt(String prompt, {Object? scope})`: Queues a text generation request. Returns a `Future<String>` prompt ID.
- `sendPromptWithImages(...)`: Queues a multimodal generation request.
- `stop()`: Aborts the current generation immediately.
- `getEmbeddings(String prompt)`: Calculates vector embeddings (requires `embeddings: true` in context params).
- `dispose()`: Kills the isolate and cleans up resources.

#### Synchronization
- `waitForCompletion(String promptId)`: Returns a Future that completes when a specific request finishes.

---

### LlamaScope
A helper class to manage a specific "session" or "conversation view" within the parent stream. It filters the global stream to only show tokens relevant to its own prompts.

#### Usage
```dart
// Create a scope (e.g., for a specific chat window)
final scope = parent.getScope();

// Listen to this scope ONLY
scope.stream.listen((token) {
  print("Scope received: $token");
});

// Send prompt tied to this scope
await scope.sendPrompt("Hello!");
```

#### Methods
- `sendPrompt(String prompt)`: Convenience wrapper around parent.sendPrompt.
- `stop()`: Cancels any active generation started by this scope.
- `dispose()`: Closes the scope's stream controllers.

---

### CompletionEvent
Emitted when a generation request finishes (successfully or with error).

#### Properties
- `promptId` (`String`): ID of the completed request.
- `success` (`bool`): True if finished normally, false if error.
- `error` (`String?`): Error details if failed.

---

## Example: Full Lifecycle

```dart
// 1. Setup
final parent = LlamaParent(
  LlamaLoad(
    path: "gemma.gguf",
    modelParams: ModelParams(),
    contextParams: ContextParams(),
    samplerParams: SamplerParams(),
  )
);

// 2. Initialize
await parent.init();

// 3. Send Prompt
final id = await parent.sendPrompt("Why is the sky blue?");

// 4. Listen
final sub = parent.stream.listen((token) {
  stdout.write(token);
});

// 5. Wait for finish
await parent.waitForCompletion(id);
print("\nDone!");

// 6. Cleanup
await sub.cancel();
await parent.dispose();
```

## Architecture Notes
1.  **Queueing:** `LlamaParent` maintains a FIFO queue (`_promptQueue`). If you send 3 prompts, they execute one by one.
2.  **Statefulness:** The background `Llama` instance is stateful. `LlamaParent` manages clearing/resetting logic implicitly when switching between requests if necessary (though typically it appends).
3.  **Error Handling:** The isolate communicates errors back to the main thread, which are emitted via streams or thrown by `init()`.