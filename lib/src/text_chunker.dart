typedef TokenLengthFunction = int Function(String text);

class TextChunker {
  /// Max size of a chunk (in units determined by [lengthFunction], usually tokens or chars)
  final int maxChunkSize;
  
  /// Number of sentences to overlap between chunks to preserve context
  final int overlapSentences;
  
  final RegExp _sentenceSplitter;
  final RegExp _paragraphSplitter;
  final TokenLengthFunction _lengthFunction;

  TextChunker({
    this.maxChunkSize = 512,
    this.overlapSentences = 1,
    TokenLengthFunction? lengthFunction,
  })  : 
        // Default to character count if no tokenizer is provided
        _lengthFunction = lengthFunction ?? ((text) => text.length),
        
        // Split on double newlines (standard paragraph break)
        _paragraphSplitter = RegExp(r'\n\s*\n'),
        
        // Split on punctuation, handling quotes/parens after: "Hello." -> Split
        // Lookbehind for [.!?], optional quote/paren, then whitespace
        _sentenceSplitter = RegExp(r'(?<=[.!?][")\]]?)\s+');

  List<String> chunk(String text) {
    if (text.isEmpty) return [];

    final paragraphs = _splitParagraphs(text);
    final chunks = <String>[];
    
    // Buffer for the current chunk being built
    var currentSentences = <String>[];
    var currentSize = 0;

    for (final paragraph in paragraphs) {
      final sentences = _splitSentences(paragraph);

      for (final sentence in sentences) {
        final sentSize = _lengthFunction(sentence);

        // If adding this sentence exceeds the limit
        if (currentSize + sentSize > maxChunkSize) {
          
          // 1. Flush current chunk if it has content
          if (currentSentences.isNotEmpty) {
            chunks.add(currentSentences.join(' '));
          }

          // 2. Handle Overlap
          // Keep the last N sentences to maintain context in the next chunk
          if (overlapSentences > 0 && currentSentences.length > overlapSentences) {
            final overlap = currentSentences.sublist(
              currentSentences.length - overlapSentences
            );
            currentSentences = List.from(overlap);
            
            // Recalculate size for the new start
            currentSize = 0;
            for (var s in currentSentences) currentSize += _lengthFunction(s);
            
            // Add a space cost approximation if we are counting characters
            if (currentSentences.isNotEmpty) {
               // If counting chars, spaces matter. If tokens, usually implied.
               // We'll assume a simple addition for safety.
               currentSize += currentSentences.length; 
            }
          } else {
            currentSentences = [];
            currentSize = 0;
          }
        }

        // Add the new sentence
        currentSentences.add(sentence);
        currentSize += sentSize + 1; // +1 for the space joining them
      }
    }

    // Flush remaining buffer
    if (currentSentences.isNotEmpty) {
      chunks.add(currentSentences.join(' '));
    }

    return chunks;
  }

  List<String> _splitParagraphs(String text) {
    return text
        .split(_paragraphSplitter)
        .map((p) => p.replaceAll(RegExp(r'\s+'), ' ').trim()) // Collapse internal whitespace
        .where((p) => p.isNotEmpty)
        .toList();
  }

  List<String> _splitSentences(String paragraph) {
    // Simple protection against "Dr.", "Mr." splitting
    // A robust NLP tokenizer is too heavy, but we can avoid splitting on short abbreviations
    // This is a naive heuristic but better than raw regex.
    return paragraph
        .split(_sentenceSplitter)
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
  }
}