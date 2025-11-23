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
          stopSequence: '\n###', // Common stop token for Alpaca
        );

  @override
  String formatPrompt(String prompt) {
    // Single-shot usage: System + Input + Response Trigger
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
        // If we haven't seen a system prompt yet, some might prefer injecting the default here.
        // However, strict formatting usually respects only what's in the list.
        buffer.write('$inputSequence$content\n\n');
      } else if (role == 'assistant') {
        buffer.write('$outputSequence$content\n\n');
      }
    }

    // If the list didn't start with a system prompt, it's often good practice 
    // for Alpaca to prepend the default one, but we will respect the input list 
    // to keep the class logic pure.
    
    // Trigger generation
    buffer.write(outputSequence);

    return buffer.toString();
  }
}