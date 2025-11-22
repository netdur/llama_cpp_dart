import '../llama_cpp_dart.dart';

/// Represents supported chat formats for export
enum ChatFormat {
  chatml,
  alpaca,
  gemma,
  harmony;

  String get value => name;
}

/// Represents different roles in a chat conversation
enum Role {
  unknown,
  system,
  user,
  assistant;

  String get value => switch (this) {
        Role.unknown => 'unknown',
        Role.system => 'system',
        Role.user => 'user',
        Role.assistant => 'assistant',
      };

  static Role fromString(String value) => switch (value.toLowerCase()) {
        'unknown' => Role.unknown,
        'system' => Role.system,
        'user' => Role.user,
        'assistant' => Role.assistant,
        _ => Role.unknown,
      };
}

class Message {
  final Role role;
  final String content;

  const Message({
    required this.role,
    required this.content,
  });

  factory Message.fromJson(Map<String, dynamic> json) {
    return Message(
      role: Role.fromString(json['role'] as String),
      content: json['content'] as String,
    );
  }

  Map<String, dynamic> toJson() => {
        'role': role.value,
        'content': content,
      };

  @override
  String toString() => 'Message(role: ${role.value}, content: $content)';
}

class ChatHistory {
  final List<Message> messages;
  final List<Message> fullHistory;
  final int keepRecentPairs;

  ChatHistory({this.keepRecentPairs = 2})
      : messages = [],
        fullHistory = [];

  void addMessage({required Role role, required String content}) {
    final message = Message(role: role, content: content);
    messages.add(message);
    fullHistory.add(message);
  }

  /// --- STATEFUL API (New) ---
  
  /// Exports ONLY the latest turn (User + Optional Empty Assistant).
  /// Use this with `llama.setPrompt()` to append to existing context.
  String getLatestTurn(ChatFormat format) {
    if (messages.isEmpty) return "";

    // Logic: Get the last user message. 
    // If we added a placeholder assistant message (for streaming), include that too.
    int count = 1;
    if (messages.length >= 2 && 
        messages.last.role == Role.assistant && 
        messages.last.content.isEmpty) {
      count = 2;
    }

    final subset = messages.sublist(messages.length - count);
    return _formatList(subset, format, leaveLastAssistantOpen: true);
  }

  /// --- STATELESS API (Classic) ---

  /// Exports the ENTIRE history. 
  /// Use this when initializing context, or after calling `llama.clear()`.
  String exportFormat(ChatFormat format, {bool leaveLastAssistantOpen = false}) {
    return _formatList(messages, format, leaveLastAssistantOpen: leaveLastAssistantOpen);
  }

  // --- Internal Formatting Logic ---

  String _formatList(List<Message> msgs, ChatFormat format, {required bool leaveLastAssistantOpen}) {
    switch (format) {
      case ChatFormat.chatml:
        return _exportChatML(msgs);
      case ChatFormat.alpaca:
        return _exportAlpaca(msgs);
      case ChatFormat.gemma:
        return _exportGemma(msgs, leaveLastAssistantOpen: leaveLastAssistantOpen);
      case ChatFormat.harmony:
        return _exportHarmony(msgs, leaveLastAssistantOpen: leaveLastAssistantOpen);
    }
  }

  String _exportChatML(List<Message> msgs) {
    final buffer = StringBuffer();
    for (int i = 0; i < msgs.length; i++) {
      final message = msgs[i];
      buffer.writeln('<|im_start|>${message.role.value}');
      buffer.writeln(message.content);
      if (i < msgs.length - 1) {
        buffer.writeln('<|im_end|>');
      }
    }
    return buffer.toString();
  }

  String _exportAlpaca(List<Message> msgs) {
    final buffer = StringBuffer();
    for (final message in msgs) {
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

  String _exportGemma(List<Message> msgs, {bool leaveLastAssistantOpen = false}) {
    final buffer = StringBuffer();

    for (int i = 0; i < msgs.length; i++) {
      final message = msgs[i];
      final isLastMessage = i == msgs.length - 1;
      
      // If we are at the end, and it's an assistant message, we might want to leave it open
      final shouldLeaveOpen = leaveLastAssistantOpen && isLastMessage && message.role == Role.assistant;

      switch (message.role) {
        case Role.user:
          buffer.write('<start_of_turn>user\n${message.content}<end_of_turn>\n');
          break;
        case Role.assistant:
          buffer.write('<start_of_turn>model\n${message.content}');
          if (!shouldLeaveOpen) {
            buffer.write('<end_of_turn>\n');
          }
          break;
        case Role.system:
          // Gemma usually maps system to user
          buffer.write('<start_of_turn>user\nSystem: ${message.content}<end_of_turn>\n');
          break;
        case Role.unknown:
          break;
      }
    }
    
    // Critical: If we want to prompt the model, ensure the buffer ends with the model tag
    if (leaveLastAssistantOpen && msgs.isNotEmpty && msgs.last.role != Role.assistant) {
        buffer.write('<start_of_turn>model\n');
    }

    return buffer.toString();
  }

  String _exportHarmony(List<Message> msgs, {bool leaveLastAssistantOpen = false}) {
    final buffer = StringBuffer();

    for (int i = 0; i < msgs.length; i++) {
      final msg = msgs[i];
      final isLast = i == msgs.length - 1;

      String roleTag;
      switch (msg.role) {
        case Role.system: roleTag = '<|system|>\n'; break;
        case Role.user: roleTag = '<|user|>\n'; break;
        case Role.assistant: roleTag = '<|assistant|>\n'; break;
        default: continue;
      }

      buffer.write(roleTag);
      buffer.write(msg.content);

      final isAssistantAndOpen = leaveLastAssistantOpen && isLast && msg.role == Role.assistant;

      if (!isAssistantAndOpen) {
        buffer.write('\n<|end|>\n');
      } else {
        buffer.write('\n');
      }
    }
    return buffer.toString();
  }

  // --- Standard Methods ---

  bool autoTrimForSpace(Llama llama, {int reserveTokens = 100}) {
    int remainingSpace = llama.getRemainingContextSpace();
    if (remainingSpace > reserveTokens) return false;

    List<Message> systemMessages = messages.where((msg) => msg.role == Role.system).toList();
    List<Message> recentMessages = [];
    int pairsFound = 0;
    for (int i = messages.length - 1; i >= 0 && pairsFound < keepRecentPairs; i--) {
      recentMessages.insert(0, messages[i]);
      if (messages[i].role == Role.user) pairsFound++;
    }

    messages.clear();
    messages.addAll(systemMessages);
    messages.addAll(recentMessages);
    return true;
  }

  bool shouldTrimBeforePrompt(Llama llama, String newPrompt) {
    int estimatedTokens = (newPrompt.length / 4).ceil() + 50;
    int remainingSpace = llama.getRemainingContextSpace();
    return remainingSpace < estimatedTokens;
  }

  factory ChatHistory.fromJson(Map<String, dynamic> json) {
    final chatHistory = ChatHistory();
    final messagesList = json['messages'] as List<dynamic>;
    for (final message in messagesList) {
      chatHistory.messages.add(Message.fromJson(message as Map<String, dynamic>));
    }
    return chatHistory;
  }

  Map<String, dynamic> toJson() => {
    'messages': messages.map((message) => message.toJson()).toList(),
  };

  void clear() => messages.clear();
  int get length => messages.length;
  @override
  String toString() => 'ChatHistory(messages: $messages)';
}