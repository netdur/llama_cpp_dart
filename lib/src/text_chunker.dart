class TextChunker {
  final int maxChunkSize;
  final int overlapSentences;
  final RegExp _sentenceSplitter;
  final RegExp _paragraphSplitter;

  TextChunker({
    this.maxChunkSize = 512,
    this.overlapSentences = 2,
  })  : _paragraphSplitter = RegExp(r'\n\s*\n'),
        _sentenceSplitter = RegExp(r'(?<=[.!?])\s+');

  List<String> chunk(String text) {
    final paragraphs = _splitParagraphs(text);
    final chunks = <String>[];

    for (final paragraph in paragraphs) {
      final sentences = _splitSentences(paragraph);
      var currentChunk = <String>[];
      var currentLength = 0;

      for (var i = 0; i < sentences.length; i++) {
        final sentence = sentences[i];
        final sentenceLength = sentence.length;

        if (currentLength + sentenceLength > maxChunkSize &&
            currentChunk.isNotEmpty) {
          chunks.add(currentChunk.join(' '));

          final overlapStart = currentChunk.length > overlapSentences
              ? currentChunk.length - overlapSentences
              : 0;
          currentChunk = currentChunk.sublist(overlapStart);
          currentLength = currentChunk.join(' ').length;
        }

        currentChunk.add(sentence);
        currentLength += sentenceLength + 1;
      }

      if (currentChunk.isNotEmpty) {
        chunks.add(currentChunk.join(' '));
      }
    }

    return _mergeSmallChunks(chunks);
  }

  List<String> _splitParagraphs(String text) {
    return text
        .split(_paragraphSplitter)
        .map((p) => p.replaceAll('\n', ' ').trim())
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

  List<String> _mergeSmallChunks(List<String> chunks) {
    final merged = <String>[];
    var currentChunk = StringBuffer();

    for (final chunk in chunks) {
      if (currentChunk.isEmpty) {
        currentChunk.write(chunk);
      } else if (currentChunk.length + chunk.length + 1 <= maxChunkSize) {
        currentChunk.write(' $chunk');
      } else {
        merged.add(currentChunk.toString());
        currentChunk = StringBuffer(chunk);
      }
    }

    if (currentChunk.isNotEmpty) {
      merged.add(currentChunk.toString());
    }

    return merged;
  }
}
