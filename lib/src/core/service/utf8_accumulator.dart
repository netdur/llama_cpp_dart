import 'dart:convert';

/// Accumulates UTF-8 bytes until they decode cleanly.
class Utf8Accumulator {
  final List<int> buffer = [];
  static const int maxBufferLen = 8192;

  String process(List<int> newBytes) {
    buffer.addAll(newBytes);
    if (buffer.isEmpty) return "";

    try {
      final result = utf8.decode(buffer);
      buffer.clear();
      return result;
    } on FormatException {
      if (buffer.length > maxBufferLen) {
        final result = utf8.decode(buffer, allowMalformed: true);
        buffer.clear();
        return result;
      }
      return "";
    }
  }
}
