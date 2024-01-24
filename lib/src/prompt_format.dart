/// An enumeration representing different types of LLM Prompt Formats.
enum PromptFormat { raw, chatml, alpaca }

extension PromptFormatExtension on PromptFormat {
  static const values = {
    PromptFormat.raw: -1,
    PromptFormat.chatml: 0,
    PromptFormat.alpaca: 1,
  };

  int get value => values[this]!;
}

class PromptFormatter {
  static String formatPrompt(PromptFormat format, String prompt,
      [String role = "user", bool assistant = true]) {
    switch (format) {
      case PromptFormat.raw:
        return prompt;
      case PromptFormat.alpaca:
        return '### Input:\n\n$prompt\n\n### Response:\n\n';
      case PromptFormat.chatml:
        String formatted = '<|im_start|>$role\n$prompt\n<|im_end|>\n';
        if (assistant) {
          formatted += '<|im_start|>assistant\n';
        }
        return formatted;
      default:
        return prompt;
    }
  }
}
