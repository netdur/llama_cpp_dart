import 'prompt_format.dart';
import 'sequence_filter.dart';

class ChatMLFormat extends PromptFormat {
  ChatMLFormat()
      : super(PromptFormatType.chatml, [
          SequenceFilter('<|im_start|>user'),
          SequenceFilter('<|im_end|>'),
          SequenceFilter('<|im_start|>assistant')
        ]);
}
