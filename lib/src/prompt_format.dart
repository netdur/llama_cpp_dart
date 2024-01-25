import 'package:llama_cpp_dart/src/sequence_filter.dart';

/// An enumeration representing different types of LLM Prompt Formats.
enum PromptFormatType { raw, chatml, alpaca }

/// A class representing a LLM Prompt Format.
abstract class PromptFormat {
  final PromptFormatType type;
  final List<SequenceFilter> filters;

  PromptFormat(this.type, this.filters);

  String? filterResponse(String response) {
    // Iteratively process the response through each filter
    List<String?> chunks = [];
    for (var filter in filters) {
      chunks.add(filter.processChunk(response));
    }

    // If any of the filters return null, the response is incomplete
    for (var chunk in chunks) {
      if (chunk == null) return null;
    }

    // Return the longest chunk
    return chunks.reduce((a, b) => a!.length > b!.length ? a : b);
  }
}
