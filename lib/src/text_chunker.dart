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
        _lengthFunction = lengthFunction ?? ((text) => text.length),
        
        _paragraphSplitter = RegExp(r'\n\s*\n'),
        
        _sentenceSplitter = RegExp(r'(?<=[.!?][")\]]?)\s+');

  List<String> chunk(String text) {
    if (text.isEmpty) return [];

    final paragraphs = _splitParagraphs(text);
    final chunks = <String>[];
    
    var currentSentences = <String>[];
    var currentSize = 0;

    for (final paragraph in paragraphs) {
      final sentences = _splitSentences(paragraph);

      for (final sentence in sentences) {
        final sentSize = _lengthFunction(sentence);

        if (currentSize + sentSize > maxChunkSize) {
          if (currentSentences.isNotEmpty) {
            chunks.add(currentSentences.join(' '));
          }

          if (overlapSentences > 0 && currentSentences.length > overlapSentences) {
            final overlap = currentSentences.sublist(
              currentSentences.length - overlapSentences
            );
            currentSentences = List.from(overlap);
            
            currentSize = 0;
            for (var s in currentSentences) {
              currentSize += _lengthFunction(s);
            }
            
            if (currentSentences.isNotEmpty) {
               currentSize += currentSentences.length; 
            }
          } else {
            currentSentences = [];
            currentSize = 0;
          }
        }

        currentSentences.add(sentence);
        currentSize += sentSize + 1;
      }
    }

    if (currentSentences.isNotEmpty) {
      chunks.add(currentSentences.join(' '));
    }

    return chunks;
  }

  List<String> _splitParagraphs(String text) {
    return text
        .split(_paragraphSplitter)
        .map((p) => p.replaceAll(RegExp(r'\s+'), ' ').trim())
        .where((p) => p.isNotEmpty)
        .toList();
  }

  List<String> _splitSentences(String paragraph) {
    return paragraph
        .split(_sentenceSplitter)
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
  }
}
