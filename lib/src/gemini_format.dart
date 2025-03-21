import 'prompt_format.dart';

/// Implementation of the Gemini prompt format with system instructions as user messages.
class GeminiFormat extends PromptFormat {
  final String systemPrefix;

  GeminiFormat({this.systemPrefix = 'System: '})
      : super(PromptFormatType.raw,
            inputSequence: '<start_of_turn>user\n',
            outputSequence: '<start_of_turn>model\n',
            systemSequence: '',
            stopSequence: '<end_of_turn>\n');

  @override
  String formatPrompt(String prompt) {
    return '$inputSequence$prompt$stopSequence$outputSequence';
  }

  @override
  String formatMessages(List<Map<String, dynamic>> messages) {
    String formattedMessages = '';

    // First, check for and handle system message
    // bool hasSystemMessage = false;
    for (var message in messages) {
      if (message['role'] == 'system') {
        formattedMessages +=
            '$inputSequence$systemPrefix${message['content']}$stopSequence';
        // hasSystemMessage = true;
        break;
      }
    }

    // Then process regular messages
    for (var message in messages) {
      if (message['role'] == 'user') {
        formattedMessages += '$inputSequence${message['content']}$stopSequence';
      } else if (message['role'] == 'assistant') {
        formattedMessages +=
            '$outputSequence${message['content']}$stopSequence';
      }
      // System messages were already handled above
    }

    return formattedMessages;
  }
}
