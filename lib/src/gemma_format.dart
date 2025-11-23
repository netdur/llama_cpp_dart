import 'prompt_format.dart';

/// Implementation of the Gemma prompt format (used by Google Gemma/Gemini GGUFs).
// ignore: unintended_html_in_doc_comment
/// Format: <start_of_turn>role\ncontent<end_of_turn>
class GemmaFormat extends PromptFormat {
  final String systemPrefix;

  GemmaFormat({this.systemPrefix = 'System instruction: '})
      : super(PromptFormatType.raw,
            inputSequence: '<start_of_turn>user\n',
            outputSequence: '<start_of_turn>model\n',
            systemSequence: '<start_of_turn>user\n',
            stopSequence: '<end_of_turn>\n');

  @override
  String formatPrompt(String prompt) {
    return '$inputSequence$prompt$stopSequence$outputSequence';
  }

  @override
  String formatMessages(List<Map<String, dynamic>> messages) {
    final buffer = StringBuffer();

    for (var message in messages) {
      final role = message['role'];
      final content = message['content'];

      if (role == 'system') {
        buffer.write('$inputSequence$systemPrefix$content$stopSequence');
      } else if (role == 'user') {
        buffer.write('$inputSequence$content$stopSequence');
      } else if (role == 'assistant') {
        buffer.write('$outputSequence$content$stopSequence');
      }
    }

    buffer.write(outputSequence);

    return buffer.toString();
  }
}
