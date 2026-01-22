import 'prompt_format.dart';

/// Implementation of the ChatML prompt format (used by Qwen, Yi, etc.).
/// Structure:
/// <|im_start|>system
/// {content}<|im_end|>
/// <|im_start|>user
/// {content}<|im_end|>
/// <|im_start|>assistant
class ChatMLFormat extends PromptFormat {
  ChatMLFormat()
      : super(PromptFormatType.chatml,
            inputSequence: '<|im_start|>user\n',
            outputSequence: '<|im_start|>assistant\n',
            systemSequence: '<|im_start|>system\n',
            stopSequence: '<|im_end|>\n');

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
        buffer.write('$systemSequence$content$stopSequence');
      } else if (role == 'user') {
        buffer.write('$inputSequence$content$stopSequence');
      } else if (role == 'assistant') {
        if (content != null && content.toString().isNotEmpty) {
          buffer.write('$outputSequence$content$stopSequence');
        }
      }
    }

    if (messages.isNotEmpty && messages.last['role'] != 'assistant') {
      buffer.write(outputSequence);
    } else if (messages.isNotEmpty && messages.last['role'] == 'assistant') {
      final lastContent = messages.last['content'];
      if (lastContent == null || lastContent.toString().isEmpty) {
         buffer.write(outputSequence);
      }
    }

    return buffer.toString();
  }
}
