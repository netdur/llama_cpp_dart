import 'dart:ffi';

import '../batch/batch.dart';
import '../context/context.dart';
import '../ffi/library_loader.dart';
import '../sampling/sampler.dart';
import '../sampling/sampler_factory.dart';
import '../tokenizer/tokenizer.dart';
import '../tokenizer/utf8_accumulator.dart';
import '../types/exceptions.dart';
import 'context_shift.dart';
import 'event.dart';
import 'request.dart';
import 'stop.dart';

/// Drives the prefill + decode loop for one [Request].
///
/// One generator can be reused for multiple sequential requests on the same
/// context. Internally allocates a batch sized to `context.nBatch`.
final class Generator implements Finalizable {
  final LlamaContext context;
  final Tokenizer tokenizer;
  final LlamaBatch _batch;
  final bool _ownsBatch;
  bool _disposed = false;

  Generator(this.context, this.tokenizer, {LlamaBatch? batch})
      : _batch = batch ?? LlamaBatch(context.nBatch),
        _ownsBatch = batch == null;

  /// Run [request], yielding [TokenEvent]s as tokens are sampled and a final
  /// [DoneEvent] describing why the stream ended. May also yield
  /// [ShiftEvent]s when [Request.shiftPolicy] is `auto`.
  Stream<GenerationEvent> run(Request request) async* {
    _ensureAlive();
    if (request.promptTokens.isEmpty) {
      throw ArgumentError.value(
        request.promptTokens,
        'promptTokens',
        'Request.promptTokens must be non-empty',
      );
    }
    if (request.shiftPolicy == ContextShiftPolicy.auto && !context.canShift) {
      throw const LlamaContextException(
        'shiftPolicy=auto requested but the context cannot be shifted '
        '(recurrent / iSWA / hybrid cache). Use shiftPolicy.off and handle '
        'context overflow at the application level, or load a model whose '
        'memory backend reports llama_memory_can_shift = true.',
      );
    }

    final lib = LlamaLibrary.bindings;
    final vocab = context.model.vocab;
    final accumulator = Utf8Accumulator();

    final sampler = SamplerFactory.build(request.sampler);
    try {
      var pos = await _prefill(request, sampler);
      var generated = 0;

      while (true) {
        final token = sampler.sample(context);
        sampler.accept(token);

        final bytes = tokenizer.encodeToken(token);
        final isEog = vocab.isEog(token);

        if (isEog) {
          // Drop any pending UTF-8 fragments and suppress the EOG marker
          // from streaming text. Consumers can still inspect `bytes` or
          // re-decode `id` via the tokenizer if they need the marker.
          accumulator.clear();
          yield TokenEvent(
            id: token,
            bytes: bytes,
            text: '',
            position: pos,
          );
          yield DoneEvent(
            reason: const StopEog(),
            generatedCount: generated + 1,
            committedPosition: pos,
          );
          return;
        }

        final text = accumulator.accept(bytes);

        yield TokenEvent(
          id: token,
          bytes: bytes,
          text: text,
          position: pos,
        );
        generated++;

        if (generated >= request.maxTokens) {
          yield DoneEvent(
            reason: const StopMaxTokens(),
            generatedCount: generated,
            committedPosition: pos,
            trailingText: accumulator.flush(),
          );
          return;
        }

        // Make room in the KV cache before the next decode if needed.
        if (pos + 1 >= context.nCtx) {
          if (request.shiftPolicy != ContextShiftPolicy.auto) {
            throw LlamaDecodeException(
              0,
              'context full at pos=$pos / nCtx=${context.nCtx}; '
              'set Request.shiftPolicy = ContextShiftPolicy.auto to shift '
              'or stop earlier',
            );
          }
          final shifted = _applyShift(request, pos);
          pos -= shifted.nDiscard;
          yield ShiftEvent(
            nKeep: shifted.nKeep,
            nDiscard: shifted.nDiscard,
            newPosition: pos,
          );
        }

        // Commit token: feed back and decode so the next sample sees its logits.
        _batch.clear();
        _batch.add(token, pos, [request.seqId], wantLogits: true);
        final rc = lib.llama_decode(context.pointer, _batch.raw);
        if (rc != 0) {
          throw LlamaDecodeException(rc, 'decode failed at gen $generated');
        }
        pos++;
      }
    } finally {
      sampler.dispose();
    }
  }

  ({int nKeep, int nDiscard}) _applyShift(Request request, int pos) {
    final lib = LlamaLibrary.bindings;
    var nKeep = request.shift.nKeep < 0
        ? request.startPos + request.promptTokens.length
        : request.shift.nKeep;
    if (request.shift.keepBos) nKeep += 1;
    final keepCap = context.nCtx - 4;
    if (nKeep > keepCap) nKeep = keepCap < 0 ? 0 : keepCap;
    if (nKeep < 0) nKeep = 0;

    final nLeft = pos - nKeep;
    if (nLeft <= 0) {
      throw LlamaDecodeException(
        0,
        'cannot shift: n_left=$nLeft (n_past=$pos, n_keep=$nKeep)',
      );
    }
    final nDiscard = request.shift.nDiscard ?? (nLeft ~/ 2);
    if (nDiscard <= 0 || nDiscard > nLeft) {
      throw LlamaDecodeException(
        0,
        'invalid n_discard=$nDiscard (n_left=$nLeft)',
      );
    }

    final mem = lib.llama_get_memory(context.pointer);
    lib.llama_memory_seq_rm(
      mem,
      request.seqId,
      nKeep,
      nKeep + nDiscard,
    );
    lib.llama_memory_seq_add(
      mem,
      request.seqId,
      nKeep + nDiscard,
      pos,
      -nDiscard,
    );
    return (nKeep: nKeep, nDiscard: nDiscard);
  }

  /// Returns the position of the last decoded prefill token + 1, i.e., the
  /// next free KV slot. Logits are produced only at the last submitted token.
  Future<int> _prefill(Request request, Sampler sampler) async {
    final lib = LlamaLibrary.bindings;
    final prompt = request.promptTokens;
    final cap = _batch.capacity;
    var pos = request.startPos;

    for (var i = 0; i < prompt.length;) {
      final end = (i + cap) > prompt.length ? prompt.length : i + cap;
      final isFinalChunk = end == prompt.length;

      _batch.clear();
      for (var j = i; j < end; j++) {
        final isLast = isFinalChunk && j == end - 1;
        _batch.add(prompt[j], pos++, [request.seqId], wantLogits: isLast);
        // Inform stateful samplers (penalties etc) of the prompt context.
        sampler.accept(prompt[j]);
      }
      final rc = lib.llama_decode(context.pointer, _batch.raw);
      if (rc != 0) {
        throw LlamaDecodeException(
          rc,
          'prefill decode failed (chunk $i..${end - 1})',
        );
      }
      i = end;
    }
    return pos;
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    if (_ownsBatch) _batch.dispose();
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('Generator has been disposed.');
    }
  }
}
