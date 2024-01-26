import 'prompt_format.dart';

class ChatMLFormat extends PromptFormat {
  ChatMLFormat()
    : super(PromptFormatType.chatml, 
      inputSequence: '<|im_start|>user',
      outputSequence: '<|im_start|>assistant',
      systemSequence: '<|im_start|>system',
      stopSequence: '<|im_end|>'
    );
}
