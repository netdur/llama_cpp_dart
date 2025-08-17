import '../llama_cpp_dart.dart';

/// Represents supported chat formats for export
enum ChatFormat {
  chatml,
  alpaca,
  gemini,
  harmony;

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

  /// Full conversation history (preserved even when trimming context)
  final List<Message> fullHistory;

  /// Number of recent message pairs (user+assistant) to keep when trimming
  final int keepRecentPairs;

  ChatHistory({this.keepRecentPairs = 2})
      : messages = [],
        fullHistory = [];

  /// Adds a new message to the chat history
  void addMessage({
    required Role role,
    required String content,
  }) {
    final message = Message(role: role, content: content);
    messages.add(message);
    fullHistory.add(message);
  }

  /// Automatically trim context based on remaining space
  /// Returns true if trimming occurred
  bool autoTrimForSpace(Llama llama, {int reserveTokens = 100}) {
    // Get remaining space from llama
    int remainingSpace = llama.getRemainingContextSpace();

    // If we have enough space, no need to trim
    if (remainingSpace > reserveTokens) {
      return false;
    }

    // Find system messages to preserve
    List<Message> systemMessages =
        messages.where((msg) => msg.role == Role.system).toList();

    // Find recent messages to keep (last N pairs)
    List<Message> recentMessages = [];
    int pairsFound = 0;
    for (int i = messages.length - 1;
        i >= 0 && pairsFound < keepRecentPairs;
        i--) {
      recentMessages.insert(0, messages[i]);
      if (messages[i].role == Role.user) {
        pairsFound++;
      }
    }

    // Rebuild messages list with only preserved content
    messages.clear();
    messages.addAll(systemMessages);
    messages.addAll(recentMessages);

    return true;
  }

  /// Checks if we should trim before adding a new message
  bool shouldTrimBeforePrompt(Llama llama, String newPrompt) {
    // Estimate tokens for the new prompt (rough estimate: 1 token per 4 chars)
    int estimatedTokens = (newPrompt.length / 4).ceil() + 50; // Add buffer
    int remainingSpace = llama.getRemainingContextSpace();
    
    return remainingSpace < estimatedTokens;
  }

  /// Exports chat history in the specified format
  String exportFormat(ChatFormat format,
      {bool leaveLastAssistantOpen = false}) {
    switch (format) {
      case ChatFormat.chatml:
        return _exportChatML();
      case ChatFormat.alpaca:
        return _exportAlpaca();
      case ChatFormat.gemini:
        return _exportGemini(leaveLastAssistantOpen: leaveLastAssistantOpen);
      case ChatFormat.harmony:
        return _exportHarmony(leaveLastAssistantOpen: leaveLastAssistantOpen);
    }
  }

  /// Exports chat history in ChatML format
  String _exportChatML() {
    final buffer = StringBuffer();

    for (int i = 0; i < messages.length; i++) {
      final message = messages[i];
      buffer.writeln('<|im_start|>${message.role.value}');
      buffer.writeln(message.content);
      // buffer.writeln('<|im_end|>');
      if (i < messages.length - 1) {
        buffer.writeln('<|im_end|>');
      }
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

  /// Exports chat history in Gemini format
  /// If leaveLastAssistantOpen is true and the last message is an empty assistant message,
  /// it will not add the closing tag for that message
  String _exportGemini({bool leaveLastAssistantOpen = false}) {
    final buffer = StringBuffer();

    for (int i = 0; i < messages.length; i++) {
      final message = messages[i];
      final isLastMessage = i == messages.length - 1;

      // Handle special case for the last assistant message
      final isEmptyAssistant =
          message.role == Role.assistant && message.content.isEmpty;
      final shouldLeaveOpen =
          leaveLastAssistantOpen && isLastMessage && isEmptyAssistant;

      switch (message.role) {
        case Role.user:
          buffer.write('<start_of_turn>user\n');
          buffer.write(message.content);
          buffer.writeln('<end_of_turn>');
        case Role.assistant:
          buffer.write('<start_of_turn>model\n');
          buffer.write(message.content);
          // Only add end tag if we're not leaving this message open
          if (!shouldLeaveOpen) {
            buffer.writeln('<end_of_turn>');
          }
        case Role.system:
          buffer.write('<start_of_turn>user\n');
          buffer.write('System instruction: ${message.content}');
          buffer.writeln('<end_of_turn>');
        case Role.unknown:
          break;
      }
    }

    return buffer.toString();
  }

  /// Exports chat history in Harmony format
  /// Tokens used:
  ///   <|system|>, <|user|>, <|assistant|>, and <|end|>
  /// If leaveLastAssistantOpen is true and the last message is an assistant,
  /// we omit the trailing <|end|> to let generation continue.
  String _exportHarmony({bool leaveLastAssistantOpen = false}) {
    final buffer = StringBuffer();

    for (int i = 0; i < messages.length; i++) {
      final msg = messages[i];
      final isLast = i == messages.length - 1;

      String roleTag;
      switch (msg.role) {
        case Role.system:
          roleTag = '<|system|>\n';
          break;
        case Role.user:
          roleTag = '<|user|>\n';
          break;
        case Role.assistant:
          roleTag = '<|assistant|>\n';
          break;
        case Role.unknown:
          // Skip unknowns in strict formats; you could also serialize as user.
          continue;
      }

      buffer.write(roleTag);
      buffer.write(msg.content);

      final isAssistantAndOpen =
          leaveLastAssistantOpen && isLast && msg.role == Role.assistant;

      if (!isAssistantAndOpen) {
        buffer.write('\n<|end|>\n');
      } else {
        // No <|end|> so the model can keep generating the assistant’s turn.
        buffer.write('\n');
      }
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
