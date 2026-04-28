import 'dart:ffi';

import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import '../types/exceptions.dart';

/// Owned wrapper around `llama_batch`. Reusable across decode calls.
///
/// Allocate with capacity equal to the largest single submission you will
/// ever make to `llama_decode`. Reuse the same batch by calling [clear] and
/// [add] again — the underlying native buffers are not reallocated.
final class LlamaBatch implements Finalizable {
  /// Maximum number of tokens this batch can hold.
  final int capacity;

  /// Maximum sequence-id list length per token.
  final int nSeqMax;

  /// Embedding dimension; `0` for token-id batches.
  final int embd;

  final llama_batch _batch;
  bool _disposed = false;

  LlamaBatch._(this.capacity, this.nSeqMax, this.embd, this._batch);

  /// Allocate a new batch.
  factory LlamaBatch(int capacity, {int nSeqMax = 1, int embd = 0}) {
    if (capacity <= 0) {
      throw ArgumentError.value(capacity, 'capacity', 'must be > 0');
    }
    if (nSeqMax <= 0) {
      throw ArgumentError.value(nSeqMax, 'nSeqMax', 'must be > 0');
    }
    final raw = LlamaLibrary.bindings.llama_batch_init(capacity, embd, nSeqMax);
    return LlamaBatch._(capacity, nSeqMax, embd, raw);
  }

  /// Underlying ffi struct. Pass to `llama_decode` / `llama_encode`.
  llama_batch get raw {
    _ensureAlive();
    return _batch;
  }

  int get nTokens => _batch.n_tokens;

  /// Reset the batch without freeing capacity.
  void clear() {
    _ensureAlive();
    _batch.n_tokens = 0;
  }

  /// Append one token to the batch.
  ///
  /// [seqIds] must contain at most [nSeqMax] entries.
  /// [wantLogits] requests that logits be produced for this position; usually
  /// only the final token of a prompt or each generation step needs it.
  void add(
    int token,
    int pos,
    List<int> seqIds, {
    bool wantLogits = false,
  }) {
    _ensureAlive();

    final n = _batch.n_tokens;
    if (n >= capacity) {
      throw LlamaDecodeException(
        -1,
        'LlamaBatch full ($capacity tokens)',
      );
    }
    if (seqIds.isEmpty) {
      throw ArgumentError.value(seqIds, 'seqIds', 'must be non-empty');
    }
    if (seqIds.length > nSeqMax) {
      throw ArgumentError.value(
        seqIds,
        'seqIds',
        'length ${seqIds.length} exceeds nSeqMax=$nSeqMax',
      );
    }

    _batch.token[n] = token;
    _batch.pos[n] = pos;

    final seqIdRow = _batch.seq_id[n];
    _batch.n_seq_id[n] = seqIds.length;
    for (var i = 0; i < seqIds.length; i++) {
      seqIdRow[i] = seqIds[i];
    }
    _batch.logits[n] = wantLogits ? 1 : 0;
    _batch.n_tokens = n + 1;
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    LlamaLibrary.bindings.llama_batch_free(_batch);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('LlamaBatch has been disposed.');
    }
  }
}
