/// Represents supported chat formats for export
enum ChatFormat {
  chatml,
  alpaca;

  String get value => name;
}

/// Represents different roles in a chat conversation
enum Role {
  unknown,
  system,
  user,
  assistant;

  /// Converts Role enum to its string representation
  String get value => switch (this) {
        Role.unknown => 'unknown',
        Role.system => 'system',
        Role.user => 'user',
        Role.assistant => 'assistant',
      };

  /// Creates a Role from a string value
  static Role fromString(String value) => switch (value.toLowerCase()) {
        'unknown' => Role.unknown,
        'system' => Role.system,
        'user' => Role.user,
        'assistant' => Role.assistant,
        _ => Role.unknown,
      };
}

/// Represents a single message in a chat conversation
class Message {
  final Role role;
  final String content;

  const Message({
    required this.role,
    required this.content,
  });

  /// Creates a Message from JSON
  factory Message.fromJson(Map<String, dynamic> json) {
    return Message(
      role: Role.fromString(json['role'] as String),
      content: json['content'] as String,
    );
  }

  /// Converts Message to JSON
  Map<String, dynamic> toJson() => {
        'role': role.value,
        'content': content,
      };

  @override
  String toString() => 'Message(role: ${role.value}, content: $content)';
}

/// Manages a collection of chat messages
class ChatHistory {
  final List<Message> messages;

  ChatHistory() : messages = [];

  /// Adds a new message to the chat history
  void addMessage({
    required Role role,
    required String content,
  }) {
    messages.add(Message(role: role, content: content));
  }

  /// Exports chat history in the specified format
  String exportFormat(ChatFormat format) {
    switch (format) {
      case ChatFormat.chatml:
        return _exportChatML();
      case ChatFormat.alpaca:
        return _exportAlpaca();
    }
  }

  /// Exports chat history in ChatML format
  String _exportChatML() {
    final buffer = StringBuffer();

    for (final message in messages) {
      buffer.writeln('<|im_start|>${message.role.value}');
      buffer.writeln(message.content);
      buffer.writeln('<|im_end|>');
    }

    return buffer.toString();
  }

  /// Exports chat history in Alpaca format
  String _exportAlpaca() {
    final buffer = StringBuffer();

    for (final message in messages) {
      switch (message.role) {
        case Role.system:
          buffer.writeln('### Instruction:');
        case Role.user:
          buffer.writeln('### Input:');
        case Role.assistant:
          buffer.writeln('### Response:');
        case Role.unknown:
          buffer.writeln('### Unknown:');
      }
      buffer.writeln();
      buffer.writeln(message.content);
      buffer.writeln();
    }

    return buffer.toString();
  }

  /// Creates a ChatHistory from JSON
  factory ChatHistory.fromJson(Map<String, dynamic> json) {
    final chatHistory = ChatHistory();
    final messagesList = json['messages'] as List<dynamic>;

    for (final message in messagesList) {
      chatHistory.messages
          .add(Message.fromJson(message as Map<String, dynamic>));
    }

    return chatHistory;
  }

  /// Converts ChatHistory to JSON
  Map<String, dynamic> toJson() => {
        'messages': messages.map((message) => message.toJson()).toList(),
      };

  /// Clears all messages from the chat history
  void clear() => messages.clear();

  /// Returns the number of messages in the chat history
  int get length => messages.length;

  @override
  String toString() => 'ChatHistory(messages: $messages)';
}
