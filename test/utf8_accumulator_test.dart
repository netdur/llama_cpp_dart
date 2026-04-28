import 'dart:convert';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:test/test.dart';

void main() {
  group('Utf8Accumulator', () {
    test('passes through ASCII unchanged', () {
      final acc = Utf8Accumulator();
      expect(acc.accept('Hello'.codeUnits), 'Hello');
      expect(acc.hasPending, isFalse);
      expect(acc.flush(), '');
    });

    test('buffers an incomplete 2-byte sequence and emits on completion', () {
      final acc = Utf8Accumulator();
      // "é" is U+00E9 = 0xC3 0xA9 in UTF-8.
      expect(acc.accept(const [0xC3]), '');
      expect(acc.hasPending, isTrue);
      expect(acc.accept(const [0xA9]), 'é');
      expect(acc.hasPending, isFalse);
    });

    test('handles a 4-byte emoji split across three feeds', () {
      // U+1F600 = 0xF0 0x9F 0x98 0x80
      final acc = Utf8Accumulator();
      expect(acc.accept(const [0xF0, 0x9F]), '');
      expect(acc.accept(const [0x98]), '');
      expect(acc.accept(const [0x80]), '😀');
    });

    test('emits complete prefix and buffers tail across one call', () {
      final acc = Utf8Accumulator();
      // "abc" + first byte of "é"
      expect(acc.accept(const [0x61, 0x62, 0x63, 0xC3]), 'abc');
      expect(acc.hasPending, isTrue);
      expect(acc.accept(const [0xA9]), 'é');
    });

    test('flush emits buffered fragment as malformed UTF-8', () {
      final acc = Utf8Accumulator();
      acc.accept(const [0xC3]);
      final out = acc.flush();
      expect(out, isNotEmpty);
      expect(acc.hasPending, isFalse);
    });

    test('clear drops buffered bytes', () {
      final acc = Utf8Accumulator();
      acc.accept(const [0xC3]);
      acc.clear();
      expect(acc.hasPending, isFalse);
      expect(acc.flush(), '');
    });

    test('round-trips a longer mixed-script string fed byte-by-byte', () {
      final acc = Utf8Accumulator();
      const source = 'Hello, 世界! 😀 مرحبا';
      final bytes = utf8.encode(source);
      final buf = StringBuffer();
      for (final b in bytes) {
        buf.write(acc.accept([b]));
      }
      buf.write(acc.flush());
      expect(buf.toString(), source);
    });
  });
}
