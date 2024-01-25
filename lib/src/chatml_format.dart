import 'prompt_format.dart';
import 'sequence_filter.dart';

class ChatMLFormat extends PromptFormat {
  ChatMLFormat()
      : super(PromptFormatType.chatml, [
          SequenceFilter('<|im_start|>system'),
          SequenceFilter('<|im_start|>user'),
          SequenceFilter('<|im_end|>'),
          SequenceFilter('<|im_start|>assistant')
        ]);

  String preparePrompt(String prompt,
      [String role = "user", bool assistant = true]) {
    prompt = '<|im_start|>$role\n$prompt\n<|im_end|>\n';
    if (assistant) {
      prompt += '<|im_start|>assistant\n';
    }
    return prompt;
  }
}
