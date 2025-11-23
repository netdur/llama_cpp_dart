# Chat Communication Classes Documentation

A set of classes and enums that manage chat conversations, message formatting, context window management, and chat history export functionality.

## Enums

### ChatFormat
```dart
enum ChatFormat {
  chatml,
  alpaca,
  gemma,  // Replaces 'gemini', used for Google Gemma models
  harmony // Used for models like Mistral/Miqu
}
```
Supported formats for chat history export and prompting.

### Role
```dart
enum Role {
  unknown,
  system,
  user,
  assistant
}
```
Different roles in a chat conversation.

## Message Class

A class representing a single message in a chat conversation.

### Properties
- `role` (`Role`): The role of the message sender.
- `content` (`String`): The message content.

### Methods
- `Message.fromJson(Map<String, dynamic> json)`: Creates a message from JSON.
- `Map<String, dynamic> toJson()`: Converts message to JSON.
- `String toString()`: String representation of the message.

## ChatHistory Class

Manages a collection of chat messages with advanced capabilities for Context Management (trimming) and Stateful Generation (incremental export).

### Properties
- `messages` (`List<Message>`): The current active context (may be trimmed).
- `fullHistory` (`List<Message>`): The complete conversation history (never trimmed).
- `length` (`int`): Number of messages in the active context.
- `keepRecentPairs` (`int`): Number of user/assistant pairs to preserve when auto-trimming.

### Core Methods

#### Message Management
```dart
void addMessage({
  required Role role,
  required String content,
})
```
Adds a new message to both `messages` and `fullHistory`.

```dart
void clear()
```
Removes all messages from the active context.

#### Format Export (Stateful vs Stateless)

**1. Incremental Export (Recommended for v0.2+)**
```dart
String getLatestTurn(ChatFormat format)
```
Exports **only** the most recent turn (User message + optional empty Assistant placeholder).
*   **Use Case:** Appending to an existing stateful `Llama` context (`setPrompt` without `clear`).
*   **Benefit:** Lightning-fast generation, zero re-processing of history.

**2. Full Export (Classic)**
```dart
String exportFormat(ChatFormat format, {bool leaveLastAssistantOpen = false})
```
Exports the **entire** active history.
*   **Use Case:** Initializing a new context, or re-loading after a `llama.clear()`.

#### Context Management
```dart
bool autoTrimForSpace(Llama llama, {int reserveTokens = 100})
```
Checks the `Llama` instance's remaining context. If space is low, it removes old messages (preserving System prompt and recent pairs) to free up space. Returns `true` if trimming occurred.

```dart
bool shouldTrimBeforePrompt(Llama llama, String newPrompt)
```
Helper to check if adding `newPrompt` would exceed the context limit, allowing you to trim *before* sending data.

### Export Formats

#### ChatML Format
```
<|im_start|>role
content
<|im_end|>
```

#### Gemma Format (Google)
```
<start_of_turn>role
content
<end_of_turn>
```

#### Harmony Format
```
<|role|>
content
<|end|>
```

#### Alpaca Format
```
### Instruction/Input/Response:

content
```

### Serialization
```dart
ChatHistory.fromJson(Map<String, dynamic> json)
Map<String, dynamic> toJson()
```

## Example Usage

### 1. Standard Stateful Chat (Best Performance)
```dart
final chat = ChatHistory();
final llama = Llama(...);

// User sends a message
chat.addMessage(role: Role.user, content: "Hello!");

// Get ONLY the new part
String prompt = chat.getLatestTurn(ChatFormat.gemma);

// Append to VRAM (Stateful)
llama.setPrompt(prompt);

// Stream response...
```

### 2. Context Overflow Handling
```dart
// Check if we are running out of memory
if (chat.shouldTrimBeforePrompt(llama, userInput)) {
  // 1. Trim internal list
  chat.autoTrimForSpace(llama);
  
  // 2. Clear VRAM (Amnesia)
  llama.clear();
  
  // 3. Re-ingest the trimmed history (The "Short Term Memory")
  String refreshedContext = chat.exportFormat(ChatFormat.gemma);
  llama.setPrompt(refreshedContext);
}
```

## Best Practices
1.  **Prefer `getLatestTurn`**: Avoids re-processing thousands of tokens on every turn.
2.  **Monitor Context**: Use `autoTrimForSpace` to prevent crashes on long conversations.
3.  **Format Selection**: Ensure the `ChatFormat` matches your loaded model (e.g., use `gemma` for Gemma 2, `chatml` for Qwen/Yi).
4.  **History Preservation**: `fullHistory` keeps the entire log even if `messages` is trimmed for the AI, allowing you to still show the full chat in your UI.