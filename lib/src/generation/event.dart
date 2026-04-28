import 'dart:typed_data';

import 'stop.dart';

/// One event in a generation stream from [Generator] or [LlamaSession.generate].
sealed class GenerationEvent {
  const GenerationEvent();
}

/// A token was sampled.
///
/// At the moment this event is emitted, the token has been *sampled* but
/// has not yet been fed back into the KV cache for the next step. If the
/// stream stops on this event (EOG or maxTokens), the token will not be
/// committed to the context's KV cache.
final class TokenEvent extends GenerationEvent {
  /// Token id.
  final int id;

  /// Raw bytes of the token's piece. May be a partial UTF-8 sequence.
  final Uint8List bytes;

  /// Best-effort UTF-8 decoded text emitted by this step. Empty if the
  /// piece was a sub-codepoint fragment that's still buffered in the
  /// streaming UTF-8 accumulator.
  final String text;

  /// Position the token occupies (or would occupy if committed) in the
  /// context's KV cache.
  final int position;

  const TokenEvent({
    required this.id,
    required this.bytes,
    required this.text,
    required this.position,
  });

  @override
  String toString() => 'TokenEvent(id=$id, pos=$position, text="$text")';
}

/// Emitted when the generator just performed a context-shift to make room
/// for the next decode. By the time this fires the KV cache has already
/// been mutated; consumers that mirror the token list (like `LlamaSession`)
/// should drop tokens at indices `[nKeep, nKeep + nDiscard)` from their
/// own copy.
final class ShiftEvent extends GenerationEvent {
  /// Number of leading tokens preserved.
  final int nKeep;

  /// Number of tokens dropped from the middle.
  final int nDiscard;

  /// Total tokens in the KV cache after the shift.
  final int newPosition;

  const ShiftEvent({
    required this.nKeep,
    required this.nDiscard,
    required this.newPosition,
  });

  @override
  String toString() =>
      'ShiftEvent(nKeep=$nKeep, nDiscard=$nDiscard, newPos=$newPosition)';
}

/// Terminal event of a generation stream.
final class DoneEvent extends GenerationEvent {
  /// Why the stream ended.
  final StopReason reason;

  /// Number of [TokenEvent]s emitted by this run.
  final int generatedCount;

  /// First KV-cache position past the last *committed* (decoded) token.
  /// Use this to update session state.
  final int committedPosition;

  /// Any buffered UTF-8 fragment the streaming accumulator could not
  /// flush mid-stream. Usually empty.
  final String trailingText;

  const DoneEvent({
    required this.reason,
    required this.generatedCount,
    required this.committedPosition,
    this.trailingText = '',
  });

  @override
  String toString() => 'DoneEvent(reason=$reason, generated=$generatedCount, '
      'committedPos=$committedPosition, trailing="$trailingText")';
}
