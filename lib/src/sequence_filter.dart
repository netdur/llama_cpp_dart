import 'dart:math';

class SequenceFilter {
  final String sequence;
  StringBuffer buffer = StringBuffer();
  bool firstChunk = true;

  SequenceFilter(this.sequence);

  String? processChunk(String chunk) {
    for (var i = 0; i < chunk.length; i++) {
      String sub;
      if (buffer.isEmpty) {
        sub = chunk.substring(i);
      } else {
        final budget = min(sequence.length - buffer.length, chunk.length);
        sub = buffer.toString() + chunk.substring(0, budget);
      }

      if (sequence.contains(sub)) {
        buffer.write(sub);
        if (buffer.length == sequence.length) {
          buffer.clear();
        }
      } else {
        if (buffer.isNotEmpty) {
          final result = buffer.toString() + chunk.substring(i);
          buffer.clear();
          return result;
        }
      }
    }
    return null;
  }
}
