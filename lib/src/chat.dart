enum Role {
  unknown,
  system,
  user,
  assistant,
}

extension RoleExtension on Role {
  static const values = {
    Role.unknown: "unknown",
    Role.system: "system",
    Role.user: "user",
    Role.assistant: "assistant",
  };

  String get value => values[this]!;
}

class Message {
  final Role role;
  final String content;

  Message(this.role, this.content);

  Map<String, dynamic> toJson() => {
        'role': role.value,
        'content': content,
      };

  static Message fromJson(Map<String, dynamic> json) => Message(
        Role.values[json['role'] as int],
        json['content'] as String,
      );
}

class ChatHistory {
  List<Message> messages = [];

  void addMessage(Role authorRole, String content) {
    messages.add(Message(authorRole, content));
  }

  String toJson() => {
        'messages': messages.map((message) => message.toJson()).toList(),
      }.toString();

  static ChatHistory fromJson(Map<String, dynamic> json) => ChatHistory()
    ..messages = (json['messages'] as List<dynamic>)
        .map((e) => Message.fromJson(e as Map<String, dynamic>))
        .toList();
}
