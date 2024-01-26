import 'prompt_format.dart';

class ChatMLFormat extends PromptFormat {
  ChatMLFormat()
      : super(PromptFormatType.chatml,
            inputSequence: '<|im_start|>user',
            outputSequence: '<|im_start|>assistant',
            systemSequence: '<|im_start|>system',
            stopSequence: '<|im_end|>');

  String preparePrompt(String prompt,
      [String role = "user", bool assistant = true]) {
    prompt = '<|im_start|>$role\n$prompt\n<|im_end|>\n';
    if (assistant) {
      prompt += '<|im_start|>assistant\n';
    }
    return prompt;
  }
}
