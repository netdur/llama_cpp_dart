# Chat Communication Classes Documentation

A set of classes and enums that manage chat conversations, message formatting, and chat history export functionality.

## Enums

### ChatFormat
```dart
enum ChatFormat {
  chatml,
  alpaca
}
```
Supported formats for chat history export.

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
- `role` (`Role`): The role of the message sender
- `content` (`String`): The message content

### Methods
- `Message.fromJson(Map<String, dynamic> json)`: Creates a message from JSON
- `Map<String, dynamic> toJson()`: Converts message to JSON
- `String toString()`: String representation of the message

## ChatHistory Class

Manages a collection of chat messages with export capabilities.

### Properties
- `messages` (`List<Message>`): Collection of chat messages
- `length` (`int`): Number of messages in history

### Core Methods

#### Message Management
```dart
void addMessage({
  required Role role,
  required String content,
})
```
Adds a new message to the history.

```dart
void clear()
```
Removes all messages from history.

#### Format Export
```dart
String exportFormat(ChatFormat format)
```
Exports chat history in specified format.

### Export Formats

#### ChatML Format
```
<|im_start|>role
content
<|im_end|>
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

```dart
// Create a new chat history
final chat = ChatHistory();

// Add messages
chat.addMessage(
  role: Role.system,
  content: "You are a helpful assistant"
);
chat.addMessage(
  role: Role.user,
  content: "Hello!"
);

// Export in different formats
String chatml = chat.exportFormat(ChatFormat.chatml);
String alpaca = chat.exportFormat(ChatFormat.alpaca);

// Serialize
Map<String, dynamic> json = chat.toJson();
ChatHistory restored = ChatHistory.fromJson(json);
```

## Notes
- Thread-safe for single instance use
- Supports multiple export formats
- JSON serialization support
- Clear role-based message structure
- Efficient string building for exports

## Best Practices
1. Use appropriate roles for messages
2. Handle unknown roles gracefully
3. Choose appropriate export format for target model
4. Clear history when starting new conversations
5. Validate message content before adding