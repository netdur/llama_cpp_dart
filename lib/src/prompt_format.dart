import 'package:llama_cpp_dart/src/sequence_filter.dart';

/// An enumeration representing different types of LLM Prompt Formats.
enum PromptFormatType { raw, chatml, alpaca }

extension PromptFormatExtension on PromptFormatType {
  static const values = {
    PromptFormatType.raw: -1,
    PromptFormatType.chatml: 0,
    PromptFormatType.alpaca: 1,
  };

  int get value => values[this]!;
}

class PromptFormat {
  final PromptFormatType type;
  final List<SequenceFilter> filters;

  PromptFormat(this.type, this.filters);
}
