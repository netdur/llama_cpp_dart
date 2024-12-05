# PromptFormat Class Documentation

An abstract class that defines formatting rules and sequences for LLM (Large Language Model) prompts. It handles message formatting, response filtering, and sequence management for different prompt formats.

## Core Components

### PromptFormatType Enum
```dart
enum PromptFormatType {
  raw,      // Raw text format
  chatml,   // ChatML format
  alpaca    // Alpaca format
}
```

## Class Structure

### Properties
- `type` (`PromptFormatType`): Type of prompt format
- `inputSequence` (`String`): Sequence for user input
- `outputSequence` (`String`): Sequence for model output
- `systemSequence` (`String`): Sequence for system messages
- `stopSequence` (`String?`): Optional sequence to mark end of responses
- `_filters` (`List<SequenceFilter>`): Internal sequence filters

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

### Response Processing
```dart
String? filterResponse(String response)
```
Processes response through sequence filters to ensure proper formatting.

### Message Formatting
```dart
String formatPrompt(String prompt)
```
Formats a single prompt with appropriate sequences.

```dart
String formatMessages(List<Map<String, dynamic>> messages)
```
Formats a list of messages with role-specific sequences.

## Message Structure
Messages are expected in the format:
```dart
{
  'role': String,     // 'user', 'assistant', or 'system'
  'content': String   // Message content
}
```

## Example Usage

```dart
// Create a custom prompt format
class CustomFormat extends PromptFormat {
  CustomFormat() : super(
    PromptFormatType.chatml,
    inputSequence: "<|user|>",
    outputSequence: "<|assistant|>",
    systemSequence: "<|system|>",
    stopSequence: "<|end|>"
  );
}

// Use the format
final format = CustomFormat();

// Format a single prompt
String formatted = format.formatPrompt("Hello, how are you?");

// Format a conversation
final messages = [
  {'role': 'system', 'content': 'You are a helpful assistant'},
  {'role': 'user', 'content': 'Hello'},
  {'role': 'assistant', 'content': 'Hi there!'}
];
String conversation = format.formatMessages(messages);
```

## Filter Processing
- Manages sequence boundaries across message chunks
- Handles incomplete sequences
- Maintains conversation structure
- Filters out unwanted sequences

## Notes
- Abstract class requires implementation for specific formats
- Thread-safe for single instance use
- Supports system, user, and assistant message roles
- Optional stop sequence support
- Built-in sequence filtering

## Best Practices
1. Implement format-specific classes for different LLMs
2. Handle null returns from filterResponse
3. Maintain consistent message structure
4. Consider stop sequence requirements
5. Follow role-based message formatting