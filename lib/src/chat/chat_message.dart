import '../multimodal/media.dart';

/// One entry in a chat conversation. Immutable, isolate-sendable.
final class ChatMessage {
  /// Conversation role: typically `system`, `user`, `assistant`, or `tool`.
  /// The accepted set is determined by the model's chat template.
  final String role;

  /// Message body. May contain media markers (default `<__media__>`) that
  /// get replaced with image/audio chunks by the multimodal encoder when
  /// [media] is non-empty. The marker count must match `media.length`.
  final String content;

  /// Images and/or audio clips attached to this message, in marker order.
  /// Empty for text-only turns.
  final List<LlamaMedia> media;

  const ChatMessage({
    required this.role,
    required this.content,
    this.media = const <LlamaMedia>[],
  });

  const ChatMessage.system(this.content)
      : role = 'system',
        media = const <LlamaMedia>[];

  const ChatMessage.user(
    this.content, {
    this.media = const <LlamaMedia>[],
  }) : role = 'user';

  const ChatMessage.assistant(this.content)
      : role = 'assistant',
        media = const <LlamaMedia>[];

  const ChatMessage.tool(this.content)
      : role = 'tool',
        media = const <LlamaMedia>[];

  ChatMessage copyWith({
    String? role,
    String? content,
    List<LlamaMedia>? media,
  }) =>
      ChatMessage(
        role: role ?? this.role,
        content: content ?? this.content,
        media: media ?? this.media,
      );

  /// JSON friendly snapshot; the `media` field is reduced to a count, since
  /// raw bytes don't survive a JSON round-trip.
  Map<String, Object?> toJson() => {
        'role': role,
        'content': content,
        if (media.isNotEmpty) 'media_count': media.length,
      };

  factory ChatMessage.fromJson(Map<String, Object?> json) => ChatMessage(
        role: json['role']! as String,
        content: json['content']! as String,
      );

  @override
  String toString() {
    final mediaTag = media.isEmpty ? '' : ' +${media.length} media';
    return 'ChatMessage($role: ${_truncate(content)}$mediaTag)';
  }

  static String _truncate(String s, [int max = 60]) =>
      s.length <= max ? s : '${s.substring(0, max - 1)}…';
}
