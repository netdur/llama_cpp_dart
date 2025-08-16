import 'prompt_format.dart';

/// Implementation of the Harmony prompt format.
/// Default tokens:
///  systemSequence  = '<|system|>\n'
///  inputSequence   = '<|user|>\n'
///  outputSequence  = '<|assistant|>\n'
///  stopSequence    = '<|end|>\n'
class HarmonyFormat extends PromptFormat {
  /// When true, if the last message is from the assistant we omit the final <|end|>
  /// so a generation can continue seamlessly.
  final bool leaveLastAssistantOpen;

  HarmonyFormat({this.leaveLastAssistantOpen = false})
      : super(
          PromptFormatType.raw,
          systemSequence: '<|system|>\n',
          inputSequence: '<|user|>\n',
          outputSequence: '<|assistant|>\n',
          stopSequence: '<|end|>\n',
        );

  @override
  String formatPrompt(String prompt) {
    // Single-turn prompt -> user turn then open assistant turn
    // <|user|>prompt<|end|><|assistant|>
    final end = stopSequence;
    return '$inputSequence$prompt$end$outputSequence';
  }

  @override
  String formatMessages(List<Map<String, dynamic>> messages) {
    final buffer = StringBuffer();

    for (int i = 0; i < messages.length; i++) {
      final m = messages[i];
      final role = (m['role'] as String?)?.toLowerCase() ?? 'user';
      final content = m['content'] as String? ?? '';
      final isLast = i == messages.length - 1;

      switch (role) {
        case 'system':
          buffer.write(systemSequence);
          buffer.write(content);
          buffer.write(stopSequence);
          break;

        case 'user':
          buffer.write(inputSequence);
          buffer.write(content);
          buffer.write(stopSequence);
          break;

        case 'assistant':
          buffer.write(outputSequence);
          buffer.write(content);

          final shouldLeaveOpen =
              leaveLastAssistantOpen && isLast && content.isEmpty == false;

          if (!shouldLeaveOpen) {
            buffer.write(stopSequence);
          } else {
            // Deliberately omit <|end|> so the model can keep generating.
            buffer.write('\n');
          }
          break;

        default:
          // ignore unknown roles or map them to user if you prefer:
          // buffer.write(inputSequence); ...
          break;
      }
    }

    return buffer.toString();
  }
}
