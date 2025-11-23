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
          if (content.isNotEmpty) {
            buffer.write('$outputSequence$content$stopSequence');
          }
          break;

        default:
          break;
      }
    }

    if (messages.isNotEmpty) {
      final lastRole = messages.last['role'];
      final lastContent = messages.last['content'];

      if (lastRole != 'assistant') {
        buffer.write(outputSequence);
      } else if (lastContent == null || lastContent.toString().isEmpty) {
        buffer.write(outputSequence);
      }
    }

    return buffer.toString();
  }
}
