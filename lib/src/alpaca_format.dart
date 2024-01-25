import 'prompt_format.dart';

class AlpacaFormat extends PromptFormat {
  AlpacaFormat()
    : super(PromptFormatType.alpaca,
      inputSequence: '### Input:',
      outputSequence: '### Response:',
      systemSequence: '### Instruction:'
    );
}
