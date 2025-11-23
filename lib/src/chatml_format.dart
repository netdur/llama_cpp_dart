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
    // Single shot: User Input + Assistant Trigger
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
        // If content is empty (placeholder), we might skip writing the block 
        // here and let the trigger below handle opening it.
        // But usually, we write what we have.
        if (content != null && content.toString().isNotEmpty) {
          buffer.write('$outputSequence$content$stopSequence');
        }
      }
    }

    // CRITICAL: Append the open tag for the assistant to start generating
    // Check if the last message was already an assistant message (continuation)
    // If not, add the trigger.
    if (messages.isNotEmpty && messages.last['role'] != 'assistant') {
      buffer.write(outputSequence);
    } else if (messages.isNotEmpty && messages.last['role'] == 'assistant') {
      // If the last message was assistant but empty/partial, we might need to ensure 
      // we are not closing it if we want to continue generation. 
      // However, standard logic usually assumes list contains completed turns.
      // For a strict prompt trigger, we usually force the tag.
      
      // Special handling: if the last item was an empty assistant placeholder,
      // we treated it above. If it was written above as empty + stopSequence,
      // the model might stop.
      // Ideally, for the "generate next" logic, the list shouldn't contain the
      // answer we want to generate yet, OR it should be an empty string which we handle here.
      
      final lastContent = messages.last['content'];
      if (lastContent == null || lastContent.toString().isEmpty) {
         // It was skipped in the loop, so we write the header now
         buffer.write(outputSequence);
      }
    }

    return buffer.toString();
  }
}