import 'prompt_format.dart';

class AlpacaFormat extends PromptFormat {
  AlpacaFormat()
      : super(PromptFormatType.alpaca,
            inputSequence: '### Input:',
            outputSequence: '### Response:',
            systemSequence: '### Instruction:');

  String preparePrompt(String prompt) {
    return '### Input:\n\n$prompt\n\n### Response:\n\n';
  }
}
