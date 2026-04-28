import 'dart:convert';
import 'dart:typed_data';

/// Accumulates raw bytes from streamed token pieces and flushes only
/// complete UTF-8 codepoints.
///
/// Many BPE/SentencePiece pieces are sub-codepoint fragments (especially
/// for non-ASCII text or emoji). Decoding each piece independently corrupts
/// multi-byte characters. Feed bytes to [accept] and emit the returned
/// string; remaining bytes stay buffered until they form complete codepoints.
final class Utf8Accumulator {
  final BytesBuilder _buf = BytesBuilder(copy: false);

  /// Append [bytes] and return any complete UTF-8 prefix that can now be
  /// decoded. Incomplete trailing codepoints stay buffered for next call.
  String accept(List<int> bytes) {
    if (bytes.isEmpty && _buf.isEmpty) return '';
    _buf.add(bytes);
    final all = _buf.takeBytes();

    final boundary = _lastCompleteBoundary(all);
    if (boundary == 0) {
      _buf.add(all);
      return '';
    }
    final flushable = Uint8List.sublistView(all, 0, boundary);
    if (boundary < all.length) {
      _buf.add(Uint8List.sublistView(all, boundary));
    }
    return utf8.decode(flushable, allowMalformed: true);
  }

  /// Drain any buffered bytes, returning them as a (possibly malformed)
  /// UTF-8 string. Use at end-of-stream to surface trailing fragments.
  String flush() {
    if (_buf.isEmpty) return '';
    return utf8.decode(_buf.takeBytes(), allowMalformed: true);
  }

  /// True if any bytes are buffered awaiting completion.
  bool get hasPending => _buf.isNotEmpty;

  /// Drop any buffered bytes without returning them.
  void clear() {
    if (_buf.isNotEmpty) _buf.takeBytes();
  }

  /// Return the index just past the last complete UTF-8 codepoint in [bytes].
  /// `0` means no complete codepoint is present.
  static int _lastCompleteBoundary(Uint8List bytes) {
    var i = 0;
    final n = bytes.length;
    while (i < n) {
      final b = bytes[i];
      final width = b < 0x80
          ? 1
          : (b & 0xE0) == 0xC0
              ? 2
              : (b & 0xF0) == 0xE0
                  ? 3
                  : (b & 0xF8) == 0xF0
                      ? 4
                      : 1;
      if (i + width > n) break;
      i += width;
    }
    return i;
  }
}
