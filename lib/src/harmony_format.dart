import 'prompt_format.dart';

/// Implementation of the Harmony prompt format.
/// Tokens:
///  systemSequence  = '<|system|>\n'
///  inputSequence   = '<|user|>\n'
///  outputSequence  = '<|assistant|>\n'
///  stopSequence    = '<|end|>\n'
class HarmonyFormat extends PromptFormat {
  HarmonyFormat()
      : super(
          PromptFormatType.raw,
          systemSequence: '<|system|>\n',
          inputSequence: '<|user|>\n',
          outputSequence: '<|assistant|>\n',
          stopSequence: '<|end|>\n',
        );

  @override
  String formatPrompt(String prompt) {
    // Single-turn prompt -> User turn + End + Open Assistant turn
    return '$inputSequence$prompt$stopSequence$outputSequence';
  }

  @override
  String formatMessages(List<Map<String, dynamic>> messages) {
    final buffer = StringBuffer();

    for (int i = 0; i < messages.length; i++) {
      final m = messages[i];
      final role = (m['role'] as String?)?.toLowerCase() ?? 'user';
      final content = m['content'] as String? ?? '';

      switch (role) {
        case 'system':
          buffer.write('$systemSequence$content$stopSequence');
          break;

        case 'user':
          buffer.write('$inputSequence$content$stopSequence');
          break;

        case 'assistant':
          // Check if this is an empty placeholder or a partial generation
          // If content is present, write it.
          if (content.isNotEmpty) {
            buffer.write('$outputSequence$content$stopSequence');
          }
          break;

        default:
          break;
      }
    }

    // CRITICAL: Trigger logic
    if (messages.isNotEmpty) {
      final lastRole = messages.last['role'];
      final lastContent = messages.last['content'];

      if (lastRole != 'assistant') {
        // If the last message was User/System, open the Assistant tag
        buffer.write(outputSequence);
      } else if (lastContent == null || lastContent.toString().isEmpty) {
        // If the last message was an empty Assistant placeholder, open the tag
        buffer.write(outputSequence);
      }
    }

    return buffer.toString();
  }
}