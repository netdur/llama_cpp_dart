import 'dart:collection';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../context/context.dart';
import '../ffi/library_loader.dart';
import '../generation/context_shift.dart';
import '../generation/event.dart';
import '../generation/generator.dart';
import '../generation/request.dart';
import '../sampling/sampler_params.dart';
import '../tokenizer/tokenizer.dart';
import '../types/exceptions.dart';

/// In-RAM conversation state on top of a [LlamaContext].
///
/// Owns the token history for one sequence id and the corresponding KV
/// cache cursor. Disk tiering and serialization come in M6.
final class LlamaSession {
  final LlamaContext context;
  final int seqId;
  final Tokenizer tokenizer;

  final List<int> _tokens = <int>[];
  int _kvHead = 0;

  LlamaSession(this.context, {this.seqId = 0})
      : tokenizer = Tokenizer(context.model.vocab);

  /// Full token history (prompt + generated).
  List<int> get tokens => UnmodifiableListView(_tokens);

  /// Number of tokens in the conversation history.
  int get tokenCount => _tokens.length;

  /// Position one past the last token committed to the KV cache.
  /// Tokens at index `>= kvHead` need prefill before sampling.
  int get kvHead => _kvHead;

  /// True when there are tokens in [tokens] that have not been decoded yet.
  bool get hasPendingPrompt => _kvHead < _tokens.length;

  /// Encode [text] and append the resulting tokens to the conversation.
  /// They become the next prefill input.
  void appendText(
    String text, {
    bool addSpecial = false,
    bool parseSpecial = true,
  }) {
    final ids = tokenizer.encode(
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
    _tokens.addAll(ids);
  }

  /// Append already-tokenized [ids] to the conversation.
  void appendTokens(List<int> ids) {
    _tokens.addAll(ids);
  }

  /// Generate tokens until stop. Performs prefill of any pending tokens.
  ///
  /// As tokens are emitted they are appended to [tokens]. The KV cache
  /// position is updated on the terminal [DoneEvent].
  Stream<GenerationEvent> generate({
    SamplerParams sampler = const SamplerParams(),
    int maxTokens = 256,
    ContextShiftPolicy shiftPolicy = ContextShiftPolicy.off,
    ContextShift shift = ContextShift.defaults,
  }) async* {
    if (!hasPendingPrompt) {
      throw StateError(
        'LlamaSession has no pending tokens; call appendText() before generate().',
      );
    }

    final pendingStart = _kvHead;
    final request = Request(
      promptTokens: _tokens.sublist(pendingStart),
      sampler: sampler,
      maxTokens: maxTokens,
      seqId: seqId,
      startPos: pendingStart,
      shiftPolicy: shiftPolicy,
      shift: shift,
    );

    final generator = Generator(context, tokenizer);
    try {
      await for (final event in generator.run(request)) {
        switch (event) {
          case TokenEvent():
            // event.position is the next free KV slot at the moment of
            // sampling — i.e., everything below has been decoded. Updating
            // here keeps _kvHead correct even if the stream is cancelled
            // mid-flight.
            _kvHead = event.position;
            _tokens.add(event.id);
            yield event;
          case ShiftEvent():
            // The Generator already mutated the underlying KV cache; mirror
            // the change in our token list so the next generate sees a
            // consistent prompt.
            if (event.nDiscard > 0 &&
                event.nKeep + event.nDiscard <= _tokens.length) {
              _tokens.removeRange(event.nKeep, event.nKeep + event.nDiscard);
            }
            _kvHead = event.newPosition;
            yield event;
          case DoneEvent():
            _kvHead = event.committedPosition;
            yield event;
            return;
        }
      }
    } finally {
      generator.dispose();
    }
  }

  /// Drop all tokens and reset the KV cache for this session's sequence.
  void clear() {
    final mem = LlamaLibrary.bindings.llama_get_memory(context.pointer);
    LlamaLibrary.bindings.llama_memory_seq_rm(mem, seqId, -1, -1);
    _tokens.clear();
    _kvHead = 0;
  }

  /// Drop the oldest non-keep window from the KV cache and slide the
  /// remainder left, matching `llama-server`'s `--context-shift` behaviour.
  ///
  /// Returns the number of tokens actually discarded (0 if there isn't
  /// enough room beyond [nKeep] to discard anything). Throws if the
  /// context's memory backend doesn't support shifting.
  ///
  /// [nKeep] is the absolute count of leading tokens to preserve; the
  /// caller is responsible for adjusting for BOS if needed. [nDiscard] is
  /// the number of tokens to drop; if null, defaults to half of `n_left`.
  int shiftContext({required int nKeep, int? nDiscard}) {
    final b = LlamaLibrary.bindings;
    final ctxPtr = context.pointer;
    if (!b.llama_memory_can_shift(b.llama_get_memory(ctxPtr))) {
      throw const LlamaContextException(
        "this context's memory backend does not support shifting "
        '(typically iSWA / recurrent / hybrid caches)',
      );
    }

    final nPast = _kvHead;
    final clampedKeep = nKeep < 0 ? 0 : (nKeep > nPast ? nPast : nKeep);
    // Always reserve at least 4 slots for forward progress, like server does.
    final keepCap = context.nCtx - 4;
    final effectiveKeep =
        clampedKeep > keepCap ? (keepCap < 0 ? 0 : keepCap) : clampedKeep;
    final nLeft = nPast - effectiveKeep;
    if (nLeft <= 0) return 0;

    final actualDiscard = nDiscard ?? (nLeft ~/ 2);
    if (actualDiscard <= 0) return 0;
    if (actualDiscard > nLeft) {
      throw ArgumentError.value(
        nDiscard,
        'nDiscard',
        'cannot exceed n_left=$nLeft (n_past=$nPast, n_keep=$effectiveKeep)',
      );
    }

    final mem = b.llama_get_memory(ctxPtr);
    b.llama_memory_seq_rm(
      mem,
      seqId,
      effectiveKeep,
      effectiveKeep + actualDiscard,
    );
    b.llama_memory_seq_add(
      mem,
      seqId,
      effectiveKeep + actualDiscard,
      nPast,
      -actualDiscard,
    );

    // Slide our token mirror to match.
    _tokens.removeRange(
      effectiveKeep,
      effectiveKeep + actualDiscard,
    );
    _kvHead = nPast - actualDiscard;
    return actualDiscard;
  }

  /// Snapshot the raw KV state for this sequence. The bytes are opaque to
  /// the wrapper — pass them to [restoreRawState] later (after model load)
  /// to resume.
  Uint8List captureRawState() {
    final b = LlamaLibrary.bindings;
    final ctxPtr = context.pointer;
    final size = b.llama_state_seq_get_size(ctxPtr, seqId);
    if (size == 0) return Uint8List(0);
    final buf = calloc<Uint8>(size);
    try {
      final written = b.llama_state_seq_get_data(ctxPtr, buf, size, seqId);
      if (written == 0) {
        throw const LlamaContextException(
          'llama_state_seq_get_data wrote 0 bytes',
        );
      }
      return Uint8List.fromList(buf.asTypedList(written));
    } finally {
      calloc.free(buf);
    }
  }

  /// Apply a previously captured KV blob to this session's sequence and
  /// replace the in-memory token history with [tokens]. The caller is
  /// responsible for verifying that [bytes] came from a compatible model.
  void restoreRawState(Uint8List bytes, List<int> tokens, int kvHead) {
    final b = LlamaLibrary.bindings;
    final ctxPtr = context.pointer;
    // Reset KV for this sequence first; otherwise set_data appends.
    final mem = b.llama_get_memory(ctxPtr);
    b.llama_memory_seq_rm(mem, seqId, -1, -1);

    if (bytes.isNotEmpty) {
      final buf = calloc<Uint8>(bytes.length);
      try {
        buf.asTypedList(bytes.length).setAll(0, bytes);
        final read = b.llama_state_seq_set_data(
          ctxPtr,
          buf,
          bytes.length,
          seqId,
        );
        if (read == 0) {
          throw const LlamaContextException(
            'llama_state_seq_set_data rejected the buffer',
          );
        }
      } finally {
        calloc.free(buf);
      }
    }

    _tokens
      ..clear()
      ..addAll(tokens);
    _kvHead = kvHead;
  }
}
