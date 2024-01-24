class SequenceFilter {
  final List<String> sequences;
  StringBuffer buffer = StringBuffer();

  SequenceFilter(this.sequences);

  String? processChunk(String chunk) {
    buffer.write(chunk); // Add new chunk to the buffer

    // Check if the buffer contains any of the sequences
    bool sequenceFound;
    do {
      sequenceFound = false;
      for (String sequence in sequences) {
        int index = buffer.toString().indexOf(sequence);
        if (index != -1) {
          // Remove the sequence from the buffer
          buffer = StringBuffer(buffer.toString().replaceFirst(sequence, ''));
          sequenceFound = true;
          break; // Exit the loop and start checking again
        }
      }
    } while (sequenceFound);

    // Return and clear the buffer if it's safe
    bool endsWithPartOfAnySequence = sequences.any(
      (seq) => buffer.toString().endsWith(seq.substring(0, 1))
    );
    if (!endsWithPartOfAnySequence && buffer.isNotEmpty) {
      String result = buffer.toString();
      buffer.clear();
      return result;
    }
    return null; // Return null if buffer is not safe to return
  }
}
