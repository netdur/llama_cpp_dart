/// An enumeration representing different types of LLM Prompt Formats.
enum PromptFormatType {
  raw,
  chatml,
  alpaca,
  gemma,
  harmony,
}

/// A class representing a LLM Prompt Format.
/// Handles formatting of messages for specific model architectures.
abstract class PromptFormat {
  final PromptFormatType type;
  final String inputSequence;
  final String outputSequence;
  final String systemSequence;
  final String? stopSequence;

  PromptFormat(this.type,
      {required this.inputSequence,
      required this.outputSequence,
      required this.systemSequence,
      this.stopSequence});

  /// Formats a single prompt (User -> Assistant).
  String formatPrompt(String prompt) {
    if (stopSequence != null) {
      return '$inputSequence$prompt$stopSequence$outputSequence';
    }
    return '$inputSequence$prompt$outputSequence';
  }

  /// Formats a list of messages into a full conversation string.
  /// Subclasses should override this for specific logic (e.g. handling newlines or special tags).
  String formatMessages(List<Map<String, dynamic>> messages) {
    final buffer = StringBuffer();
    
    for (var message in messages) {
      final role = message['role'];
      final content = message['content'];

      if (role == 'user') {
        buffer.write('$inputSequence$content');
      } else if (role == 'assistant') {
        buffer.write('$outputSequence$content');
      } else if (role == 'system') {
        buffer.write('$systemSequence$content');
      }

      if (stopSequence != null) {
        buffer.write(stopSequence!);
      }
    }
    return buffer.toString();
  }
}