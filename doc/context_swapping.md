# Context Swapping & Multi-Tenancy

This library supports **Context Swapping** (also known as Multi-Tenancy or Slots). This feature allows a single loaded model to handle multiple independent conversations simultaneously without reloading the heavy model weights.

## How it Works

In the Llama architecture, memory is split into two parts:

1.  **The Model (Weights):** Read-only, very large (e.g., 4GB). Loaded once.
2.  **The Context (KV Cache):** Read-write, dynamic (e.g., 200MB per user). Stores the "memory" of the current conversation.

**Context Swapping** allows you to allocate multiple **Contexts** (Slots) that share the single **Model**.

*   **Switching is Instant:** Changing from User A to User B takes 0ms (pointer swap).
*   **State is Preserved:** User A's conversation remains in VRAM while you talk to User B.

---

## 1. High-Level Usage (Isolates)

For Flutter apps, use `LlamaParent` and `LlamaScope`. The library handles the complex slot management automatically in the background isolate.

### Using `LlamaScope`

Think of a `LlamaScope` as a "Session" or "Tab". Every prompt sent through a scope is isolated to its own memory slot.

```dart
// 1. Initialize the main controller
final parent = LlamaParent(LlamaLoad(...));
await parent.init();

// 2. Create independent scopes (e.g., two chat windows)
final chatWindowA = parent.getScope();
final chatWindowB = parent.getScope();

// 3. Talk to A
// The isolate allocates Slot A in VRAM and generates.
await chatWindowA.sendPrompt("My name is Alice."); 

// 4. Talk to B
// The isolate creates Slot B, switches context, and generates.
// Model does NOT reload.
await chatWindowB.sendPrompt("My name is Bob.");

// 5. Talk to A again
// The isolate switches back to Slot A.
// It remembers "My name is Alice" immediately.
await chatWindowA.sendPrompt("What is my name?"); 
// Result: "Alice"

// 6. Cleanup
await chatWindowA.dispose(); // Frees Slot A from VRAM
await parent.dispose();      // Frees Model and Slot B
```

### How it works internally
1.  `parent.getScope()` generates a unique ID (e.g., `scope_123`).
2.  When you send a prompt, that ID is passed to the background isolate.
3.  The background worker checks if a VRAM slot exists for `scope_123`.
    *   **If no:** It calls `llama.createSlot("scope_123")`.
    *   **If yes:** It calls `llama.setSlot("scope_123")`.
4.  Generation proceeds using that specific memory context.

---

## 2. Low-Level Usage (Direct API)

If you are using the `Llama` class directly (without isolates), you manage slots manually using `createSlot` and `setSlot`.

```dart
final llama = Llama('model.gguf', ...);

// --- User 1 arrives ---
llama.createSlot("user_1"); // Allocates VRAM
llama.setSlot("user_1");    // Switches active context
llama.setPrompt("Hello, I am User 1");
await llama.generateCompleteText();

// --- User 2 arrives ---
llama.createSlot("user_2"); // Allocates new VRAM block
llama.setSlot("user_2");    // Switches active context
llama.setPrompt("Hello, I am User 2");
await llama.generateCompleteText();

// --- User 1 returns ---
llama.setSlot("user_1");    // Instant switch
llama.setPrompt("Who am I?");
// The model sees: "Hello, I am User 1\nWho am I?"
await llama.generateCompleteText(); 
```

### Memory Management
Each slot consumes VRAM equal to the `nCtx` (context size) parameter.

```dart
// Free a specific user's memory
llama.freeSlot("user_1");

// Free everything
llama.dispose(); 
```

---

## 3. Advanced: Hot-Swapping (RAM <-> VRAM)

If you have many users (e.g., 100) but limited VRAM (can only fit 5 slots), you can **Swap to RAM**.

This allows you to "freeze" a conversation, save it to system RAM (cheap), and free up the GPU for someone else.

```dart
// 1. Save User A's state to Dart RAM
llama.setSlot("user_a");
Uint8List frozenState = llama.saveState();

// 2. Free User A from GPU
llama.freeSlot("user_a");

// ... Allow User B to use the GPU ...

// 3. User A returns
llama.createSlot("user_a"); // Allocate empty GPU slot
llama.setSlot("user_a");
llama.loadState(frozenState); // Inject previous memory

// 4. Continue
llama.setPrompt("I'm back!");
```

## Limitations

1.  **VRAM Limits:** `createSlot` will throw an exception if the GPU runs out of memory. You must handle this by freeing other slots or disposing the instance.
2.  **Queueing:** The `LlamaParent` processes requests sequentially. Even with multiple slots, the GPU computes one token stream at a time. Context Swapping makes the *switching* instant, but it does not enable parallel inference on a single GPU.
3.  **Shared Sampler:** Currently, sampling parameters (Temperature, Top-K) are shared. If you change temperature for User A, ensure you change it back for User B (or set it per-request).