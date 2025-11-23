# PromptFormat Class Documentation

An abstract class that defines formatting rules and control sequences for LLM (Large Language Model) prompts. It handles the construction of prompt strings by wrapping user, system, and assistant messages with the specific tokens required by the model architecture.

## Core Components

### PromptFormatType Enum
```dart
enum PromptFormatType {
  raw,      // Raw text format
  chatml,   // ChatML format (Qwen, Yi)
  alpaca,   // Alpaca format (Llama 2 finetunes)
  gemma,    // Gemma format (Google Gemma / Gemini)
  harmony   // Harmony/Mistral format
}
```

## Class Structure

### Properties
- `type` (`PromptFormatType`): Type of prompt format.
- `inputSequence` (`String`): Prefix for user input (e.g., `<|im_start|>user\n`).
- `outputSequence` (`String`): Prefix for model output (e.g., `<|im_start|>assistant\n`).
- `systemSequence` (`String`): Prefix for system messages.
- `stopSequence` (`String?`): Optional suffix to mark end of turns (e.g., `<|im_end|>`).

### Constructor
```dart
PromptFormat(
  PromptFormatType type, {
  required String inputSequence,
  required String outputSequence,
  required String systemSequence,
  String? stopSequence
})
```

## Methods

### Message Formatting
```dart
String formatPrompt(String prompt)
```
Formats a single-turn prompt (User Input -> Assistant Trigger).
*   **Returns:** `inputSequence + prompt + stopSequence + outputSequence`

```dart
String formatMessages(List<Map<String, dynamic>> messages)
```
Formats a list of messages into a full conversation string.
*   **Input:** List of Maps with `role` and `content`.
*   **Logic:** Iterates through messages, applying the correct start/end sequences for each role.

## Message Structure
Input for `formatMessages` is expected in the format:
```dart
{
  'role': String,     // 'user', 'assistant', or 'system'
  'content': String   // Message content
}
```

## Example Usage

### Using a Pre-defined Format
The library provides subclasses for common formats:
```dart
final format = ChatMLFormat(); 
// Automatically handles <|im_start|> tags
```

### Creating a Custom Format
```dart
class CustomFormat extends PromptFormat {
  CustomFormat() : super(
    PromptFormatType.raw,
    inputSequence: "User: ",
    outputSequence: "AI: ",
    systemSequence: "System: ",
    stopSequence: "\n"
  );
}

final format = CustomFormat();
String prompt = format.formatPrompt("Hello");
// Result: "User: Hello\nAI: "
```

## Notes
- **Abstract Base:** While you can instantiate this logic via custom classes, it is recommended to use the provided subclasses (`GemmaFormat`, `ChatMLFormat`, etc.) as they contain specific logic for newlines and generation triggers.
- **State:** This class is stateless and thread-safe.
- **Filtering:** Previous versions included response filtering. This has been removed in v0.2 as `llama.cpp` now handles EOS token detection natively.

## Best Practices
1.  **Match the Model:** Ensure you select the format that matches your `.gguf` model (e.g., use `GemmaFormat` for Gemma 2, `ChatMLFormat` for Qwen).
2.  **Triggers:** Ensure your format ends with the `outputSequence` (Assistant tag) to trigger the model to start generating immediately.
3.  **Clean Content:** Ensure message content does not contain the control tokens themselves to prevent injection/hallucination.