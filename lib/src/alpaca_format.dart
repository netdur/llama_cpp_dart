import 'prompt_format.dart';
import 'sequence_filter.dart';

class AlpacaFormat extends PromptFormat {
  AlpacaFormat()
      : super(PromptFormatType.alpaca,
            [SequenceFilter('### Input:'), SequenceFilter('### Response:')]);

  String preparePrompt(String prompt) {
    return '### Input:\n\n$prompt\n\n### Response:\n\n';
  }
}
