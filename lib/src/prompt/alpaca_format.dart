import 'prompt_format.dart';

/// Implementation of the Alpaca prompt format.
/// Structure:
/// ### Instruction:
/// {system}
///
/// ### Input:
/// {user}
///
/// ### Response:
/// {model}
class AlpacaFormat extends PromptFormat {
  /// Alpaca models often expect a specific default system prompt if none is provided.
  final String defaultSystemPrompt;

  AlpacaFormat({
    this.defaultSystemPrompt =
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
  }) : super(
          PromptFormatType.alpaca,
          inputSequence: '### Input:\n',
          outputSequence: '### Response:\n',
          systemSequence: '### Instruction:\n',
          stopSequence: '\n###',
        );

  @override
  String formatPrompt(String prompt) {
    return '$systemSequence$defaultSystemPrompt\n\n$inputSequence$prompt\n\n$outputSequence';
  }

  @override
  String formatMessages(List<Map<String, dynamic>> messages) {
    final buffer = StringBuffer();

    for (var message in messages) {
      final role = message['role'];
      final content = message['content'];

      if (role == 'system') {
        buffer.write('$systemSequence$content\n\n');
      } else if (role == 'user') {
        buffer.write('$inputSequence$content\n\n');
      } else if (role == 'assistant') {
        buffer.write('$outputSequence$content\n\n');
      }
    }

    buffer.write(outputSequence);

    return buffer.toString();
  }
}
