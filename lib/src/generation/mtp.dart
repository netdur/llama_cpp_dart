import 'dart:ffi';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../context/context.dart';
import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import '../tokenizer/tokenizer.dart';
import '../types/exceptions.dart';
import 'speculative.dart' show SpeculativeResult;

/// FFI shim over llama.cpp's pre-norm hidden-state staging API
/// (`src/llama-ext.h`). These are **C++-mangled** (Itanium ABI) symbols, not
/// part of the public C header, so they are resolved by hand rather than
/// through the generated bindings. Stable on clang/gcc (iOS/Android/macOS);
/// would differ under MSVC.
final class _PreNorm {
  // On macOS/Linux `nm` shows a leading '_' that dlsym strips, so lookup
  // names begin at '_Z'.
  static const _setName = '_Z29llama_set_embeddings_pre_normP13llama_contextbb';
  static const _getIthName =
      '_Z33llama_get_embeddings_pre_norm_ithP13llama_contexti';

  final void Function(Pointer<Void>, bool, bool) _set;
  final Pointer<Float> Function(Pointer<Void>, int) _getIth;

  _PreNorm._(this._set, this._getIth);

  factory _PreNorm.resolve() {
    final path = LlamaLibrary.libraryPath;
    final lib = (path == null || path == '<process>')
        ? DynamicLibrary.process()
        : DynamicLibrary.open(path);
    try {
      final set = lib.lookupFunction<Void Function(Pointer<Void>, Bool, Bool),
          void Function(Pointer<Void>, bool, bool)>(_setName);
      final getIth = lib.lookupFunction<
          Pointer<Float> Function(Pointer<Void>, Int32),
          Pointer<Float> Function(Pointer<Void>, int)>(_getIthName);
      return _PreNorm._(set, getIth);
    } on ArgumentError catch (e) {
      throw LlamaLibraryException(
        'MTP requires the pre-norm embeddings staging API '
        '(llama_set_embeddings_pre_norm / llama_get_embeddings_pre_norm_ith) '
        'which is missing from this build of libllama: $e',
      );
    }
  }

  /// Enable/disable pre-norm hidden-state output for [ctx]. When [masked] is
  /// true, only positions with `logits != 0` emit hidden states; when false,
  /// every position in the batch does.
  void setEnabled(LlamaContext ctx, bool value, {required bool masked}) =>
      _set(ctx.pointer.cast(), value, masked);

  /// Snapshot the pre-norm hidden state of batch row [i] for [ctx] into a
  /// freshly-owned [Float32List] of length [nEmbd].
  Float32List rowCopy(LlamaContext ctx, int i, int nEmbd) {
    final p = _getIth(ctx.pointer.cast(), i);
    if (p == nullptr) {
      throw const LlamaContextException(
        'llama_get_embeddings_pre_norm_ith returned null '
        '(pre-norm output not enabled or row out of range)',
      );
    }
    return Float32List.fromList(p.asTypedList(nEmbd));
  }
}

/// A `llama_batch` carrying **both** token ids and per-row embeddings, as the
/// MTP draft path requires. `llama_batch_init` only allocates one of the two
/// buffers, so the token array is allocated and freed here by hand (mirroring
/// upstream's `common_speculative_impl_draft_mtp`).
final class _MtpBatch {
  final int capacity;
  final int nEmbd;
  final llama_batch _batch;
  final Pointer<llama_token> _token;
  final Float32List _embdView;
  bool _disposed = false;

  _MtpBatch._(
      this.capacity, this.nEmbd, this._batch, this._token, this._embdView);

  factory _MtpBatch(int capacity, int nEmbd) {
    final raw = LlamaLibrary.bindings.llama_batch_init(capacity, nEmbd, 1);
    // init allocated embd but not token; give it a token buffer.
    final token = calloc<llama_token>(capacity);
    raw.token = token;
    final embdView = raw.embd.asTypedList(capacity * nEmbd);
    return _MtpBatch._(capacity, nEmbd, raw, token, embdView);
  }

  llama_batch get raw => _batch;
  int get nTokens => _batch.n_tokens;

  void clear() => _batch.n_tokens = 0;

  /// Append one (token, position, embedding-row) triple.
  void add(int token, int pos, int seqId, Float32List embdRow,
      {required bool wantLogits}) {
    final n = _batch.n_tokens;
    if (n >= capacity) {
      throw LlamaDecodeException(-1, 'MTP batch full ($capacity)');
    }
    _batch.token[n] = token;
    _batch.pos[n] = pos;
    _batch.n_seq_id[n] = 1;
    _batch.seq_id[n][0] = seqId;
    _batch.logits[n] = wantLogits ? 1 : 0;
    _embdView.setRange(n * nEmbd, (n + 1) * nEmbd, embdRow);
    _batch.n_tokens = n + 1;
  }

  /// Overwrite the embedding row of an already-added position [n].
  void setEmbd(int n, Float32List embdRow) =>
      _embdView.setRange(n * nEmbd, (n + 1) * nEmbd, embdRow);

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    // Free our token buffer first, then null it so llama_batch_free does not
    // double-free a pointer it did not allocate.
    calloc.free(_token);
    _batch.token = nullptr;
    LlamaLibrary.bindings.llama_batch_free(_batch);
  }
}

/// Checkpoint of one sequence's context state, used to roll back rejected
/// drafts (partial `seq_rm` is forbidden on M-RoPE / recurrent models). Uses
/// the `_ext` API with [LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY] |
/// [LLAMA_STATE_SEQ_FLAGS_ON_DEVICE]: the snapshot stays in device memory and
/// captures only the minimal rollback delta (cost ~O(draft length) rather than
/// O(sequence length) per round), matching upstream's speculative loop.
/// [restore] re-applies it and trims the rejected tail.
final class _Checkpoint {
  static const _flags =
      LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

  Pointer<Uint8> _buf = nullptr;
  int _cap = 0;
  int _size = 0;
  int _trimFrom = 0; // first position to drop on restore (committed length)

  void save(LlamaContext ctx, int seqId) {
    final lib = LlamaLibrary.bindings;
    _trimFrom =
        lib.llama_memory_seq_pos_max(lib.llama_get_memory(ctx.pointer), seqId) +
            1;
    final need = lib.llama_state_seq_get_size_ext(ctx.pointer, seqId, _flags);
    if (need > _cap) {
      if (_buf != nullptr) calloc.free(_buf);
      _buf = calloc<Uint8>(need);
      _cap = need;
    }
    _size = lib.llama_state_seq_get_data_ext(
        ctx.pointer, _buf, need, seqId, _flags);
  }

  void restore(LlamaContext ctx, int seqId) {
    if (_buf == nullptr) return;
    final lib = LlamaLibrary.bindings;
    lib.llama_state_seq_set_data_ext(ctx.pointer, _buf, _size, seqId, _flags);
    // Drop anything decoded past the checkpoint. seq_rm may be a no-op/refused
    // on M-RoPE, but the partial restore already re-establishes the state.
    lib.llama_memory_seq_rm(
        lib.llama_get_memory(ctx.pointer), seqId, _trimFrom, -1);
  }

  void dispose() {
    if (_buf != nullptr) {
      calloc.free(_buf);
      _buf = nullptr;
      _cap = 0;
      _size = 0;
    }
  }
}

/// Greedy **MTP** (Multi-Token Prediction / NextN) speculative decoding over a
/// single model: a [target] context produces ground-truth logits and pre-norm
/// hidden states; a [draft] context built with [ContextType.mtp] runs the
/// model's NextN head, conditioned on those hidden states, to propose tokens.
///
/// This mirrors upstream llama.cpp's `common_speculative_impl_draft_mtp`
/// (PR #22673): the draft batch carries the target's pre-norm hidden state per
/// position (shifted by one), the NextN head predicts the next token, and the
/// target verifies in a single pass. Rejected drafts are rolled back via state
/// **checkpoints** (`llama_state_seq_*_data`) rather than partial `seq_rm`,
/// which M-RoPE / multimodal models forbid.
///
/// Output is identical to plain greedy decoding on [target] alone.
final class MtpSpeculativeDecoder {
  final LlamaContext target;
  final LlamaContext draft;

  final _PreNorm _pre;
  final int _nEmbd;
  final int _seqId;

  /// Cross-batch carryover: the target hidden state that conditions the first
  /// row of the next draft/mirror batch. Length [_nEmbd].
  Float32List _pendingH;

  /// Target hidden rows captured by the most recent [_process] call. Row `i`
  /// is the hidden state after the `i`-th token of the verify batch; used to
  /// pick the new [_pendingH] once acceptance is known.
  List<Float32List> _verifyH = const [];

  MtpSpeculativeDecoder._(
      this.target, this.draft, this._pre, this._nEmbd, this._seqId)
      : _pendingH = Float32List(_nEmbd);

  factory MtpSpeculativeDecoder({
    required LlamaContext target,
    required LlamaContext draft,
    int seqId = 0,
  }) {
    if (target.model.vocab.nTokens != draft.model.vocab.nTokens) {
      throw const LlamaContextException(
        'target and draft must share a vocabulary',
      );
    }
    final pre = _PreNorm.resolve();
    final nEmbd = target.model.nEmbd;
    final dec = MtpSpeculativeDecoder._(target, draft, pre, nEmbd, seqId);
    // Target emits hidden states for every position; draft only for the rows
    // it samples from (logits != 0).
    pre.setEnabled(target, true, masked: false);
    pre.setEnabled(draft, true, masked: true);
    return dec;
  }

  /// Generate greedily from [prompt]. Stops at [maxTokens] generated tokens,
  /// an end-of-generation token, or when the context window would overflow.
  ///
  /// [draftLength] is the maximum number of NextN draft tokens proposed per
  /// round; [pMin] gates them by confidence — a draft step whose top token is
  /// below this softmax probability ends that round's draft, keeping the
  /// verify batch small and acceptance high.
  SpeculativeResult generate({
    required String prompt,
    int maxTokens = 256,
    int draftLength = 3,
    double pMin = 0.5,
    bool addSpecial = true,
    bool parseSpecial = true,
  }) {
    if (prompt.isEmpty) {
      throw ArgumentError.value(prompt, 'prompt', 'must be non-empty');
    }
    if (draftLength < 1) {
      throw ArgumentError.value(draftLength, 'draftLength', 'must be >= 1');
    }

    final lib = LlamaLibrary.bindings;
    final vocab = target.model.vocab;
    final tokenizer = Tokenizer(vocab);

    final promptTokens = tokenizer.encode(prompt,
        addSpecial: addSpecial, parseSpecial: parseSpecial);
    if (promptTokens.isEmpty) {
      throw const LlamaContextException('prompt tokenized to 0 tokens');
    }

    final nCtx = target.nCtx < draft.nCtx ? target.nCtx : draft.nCtx;
    final maxBatch =
        target.nBatch < draft.nBatch ? target.nBatch : draft.nBatch;
    if (promptTokens.length > maxBatch) {
      throw LlamaContextException(
        'prompt (${promptTokens.length}) exceeds nBatch ($maxBatch); '
        'chunked prefill not implemented for MTP',
      );
    }

    final tgtMem = lib.llama_get_memory(target.pointer);
    final dftMem = lib.llama_get_memory(draft.pointer);
    lib.llama_memory_seq_rm(tgtMem, _seqId, -1, -1);
    lib.llama_memory_seq_rm(dftMem, _seqId, -1, -1);
    _pendingH = Float32List(_nEmbd);

    final draftBatch = _MtpBatch(maxBatch, _nEmbd);
    final tgtVerify = _TokenBatch(maxBatch);

    final ckptTgt = _Checkpoint();
    final ckptDft = _Checkpoint();

    // Native greedy sampler: argmax computed in llama.cpp over the on-host
    // logit buffer, avoiding a ~1 MB Dart copy of the 248k-token logit row on
    // every draft step and verify position.
    final smpl = lib.llama_sampler_init_greedy();

    final generated = <int>[];
    var drafted = 0;
    var accepted = 0;
    var stoppedOnEog = false;

    try {
      final n = promptTokens.length;

      // --- Prefill: target decodes prompt[0..n-2], last token kept aside. ---
      tgtVerify.clear();
      for (var i = 0; i < n - 1; i++) {
        tgtVerify.add(promptTokens[i], i, _seqId, wantLogits: i == n - 2);
      }
      if (n - 1 > 0) {
        final rc = lib.llama_decode(target.pointer, tgtVerify.raw);
        if (rc != 0) throw LlamaDecodeException(rc, 'MTP prefill (target)');
        // Mirror the prefill into the draft, capturing target hidden states.
        _process(
          draftBatch,
          tokens: promptTokens.sublist(0, n - 1),
          basePos: 0,
        );
      }

      var idLast = promptTokens[n - 1];
      var committed = n - 1; // KV covers [0, committed); idLast not yet decoded
      var reuseDraft = <int>[];

      while (!stoppedOnEog && generated.length < maxTokens) {
        if (committed + draftLength + 2 >= nCtx) break;

        List<int> draftTokens;
        if (reuseDraft.isNotEmpty) {
          // Re-verify a checkpoint-restored draft (all target-sampled, so it
          // is fully accepted and committed this round).
          draftTokens = reuseDraft;
          reuseDraft = <int>[];
        } else {
          // Checkpoint both contexts at the committed state before drafting.
          ckptTgt.save(target, _seqId);
          ckptDft.save(draft, _seqId);
          final savedPendingH = Float32List.fromList(_pendingH);

          draftTokens = _draft(
            draftBatch,
            idLast: idLast,
            basePos: committed,
            nMax: draftLength,
            pMin: pMin,
          );

          // Undo the draft's autoregressive pre-advancement of the draft KV
          // so [_process] can mirror the verify batch cleanly.
          ckptDft.restore(draft, _seqId);
          _pendingH = savedPendingH;
        }

        // --- Verify: target scores [idLast, draft...] in one pass. ---
        tgtVerify.clear();
        tgtVerify.add(idLast, committed, _seqId, wantLogits: true);
        for (var i = 0; i < draftTokens.length; i++) {
          tgtVerify.add(draftTokens[i], committed + 1 + i, _seqId,
              wantLogits: true);
        }
        final vrc = lib.llama_decode(target.pointer, tgtVerify.raw);
        if (vrc != 0) throw LlamaDecodeException(vrc, 'MTP verify (target)');

        // Mirror the verify batch into the draft with ground-truth target
        // hidden states; captures _verifyH and advances draft KV in lockstep.
        _process(
          draftBatch,
          tokens: [idLast, ...draftTokens],
          basePos: committed,
        );

        // --- Greedy accept: target argmax must match each draft token. ---
        final ids = <int>[];
        var acceptedThisRound = 0;
        for (var i = 0; i < draftTokens.length; i++) {
          final pred = lib.llama_sampler_sample(smpl, target.pointer, i);
          ids.add(pred);
          if (pred == draftTokens[i]) {
            acceptedThisRound++;
          } else {
            break; // pred is the bonus (divergence) token
          }
        }
        if (acceptedThisRound == draftTokens.length) {
          ids.add(lib.llama_sampler_sample(
              smpl, target.pointer, draftTokens.length));
        }

        drafted += draftTokens.length;
        accepted += acceptedThisRound;

        if (acceptedThisRound < draftTokens.length) {
          // Partial acceptance: roll both contexts back to the checkpoint and
          // re-verify the accepted prefix (+bonus) next round. idLast and
          // committed are unchanged.
          ckptTgt.restore(target, _seqId);
          ckptDft.restore(draft, _seqId);
          reuseDraft = ids;
          continue;
        }

        // Full acceptance: pick the carryover hidden state for the next round.
        final iH = acceptedThisRound < _verifyH.length
            ? acceptedThisRound
            : _verifyH.length - 1;
        if (iH >= 0) _pendingH = _verifyH[iH];

        committed += 1 + acceptedThisRound; // idLast + accepted drafts
        for (final id in ids) {
          idLast = id;
          generated.add(id);
          if (vocab.isEog(id)) {
            stoppedOnEog = true;
            break;
          }
          if (generated.length >= maxTokens) break;
        }
      }
    } finally {
      lib.llama_sampler_free(smpl);
      draftBatch.dispose();
      tgtVerify.dispose();
      ckptTgt.dispose();
      ckptDft.dispose();
    }

    final out = generated.length > maxTokens
        ? generated.sublist(0, maxTokens)
        : generated;
    return SpeculativeResult(
      tokens: out,
      text: tokenizer.decodeAll(out),
      draftedCount: drafted,
      acceptedCount: accepted,
      stoppedOnEog: stoppedOnEog,
    );
  }

  /// Mirror a target batch into the draft context, feeding each draft row the
  /// target's pre-norm hidden state from the *previous* position (row 0 gets
  /// [_pendingH]). Captures [_verifyH] and updates [_pendingH] to the last row.
  void _process(_MtpBatch batch,
      {required List<int> tokens, required int basePos}) {
    final n = tokens.length;
    if (n == 0) return;
    batch.clear();
    for (var k = 0; k < n; k++) {
      // logits=0: the draft does not sample here, it only updates KV state.
      final row = k == 0 ? _pendingH : _pre.rowCopy(target, k - 1, _nEmbd);
      batch.add(tokens[k], basePos + k, _seqId, row, wantLogits: false);
    }
    final rc = LlamaLibrary.bindings.llama_decode(draft.pointer, batch.raw);
    if (rc != 0) throw LlamaDecodeException(rc, 'MTP process (draft mirror)');

    _verifyH = List<Float32List>.generate(
        n, (i) => _pre.rowCopy(target, i, _nEmbd),
        growable: false);
    _pendingH = _verifyH[n - 1];
  }

  /// Autoregressively propose up to [nMax] draft tokens from the NextN head,
  /// each conditioned on the previous step's hidden state (the draft's own for
  /// subsequent steps, [_pendingH] for the seed). Advances the draft KV.
  List<int> _draft(_MtpBatch batch,
      {required int idLast,
      required int basePos,
      required int nMax,
      required double pMin}) {
    final lib = LlamaLibrary.bindings;
    final out = <int>[];

    // Seed: feed idLast at the committed position with the carryover hidden.
    batch.clear();
    batch.add(idLast, basePos, _seqId, _pendingH, wantLogits: true);
    var rc = lib.llama_decode(draft.pointer, batch.raw);
    if (rc != 0) throw LlamaDecodeException(rc, 'MTP draft seed');

    while (out.length < nMax) {
      // Confidence gate: only keep a draft token if the NextN head is
      // confident, else the extra (likely-rejected) token just inflates the
      // verify batch and triggers a costly partial-acceptance rollback.
      final logits = draft.logitsAt(0)!;
      final id = _argmaxWithProb(logits);
      if (id.prob < pMin) break;
      final hRow = _pre.rowCopy(draft, 0, _nEmbd);
      out.add(id.token);
      if (out.length >= nMax) break;
      if (basePos + out.length + 1 >= target.nCtx) break;
      batch.clear();
      batch.add(id.token, basePos + out.length, _seqId, hRow, wantLogits: true);
      rc = lib.llama_decode(draft.pointer, batch.raw);
      if (rc != 0) throw LlamaDecodeException(rc, 'MTP draft step');
    }
    return out;
  }

  /// Argmax token plus its softmax probability, in a single pass over the
  /// logit row (used for the draft confidence gate).
  static ({int token, double prob}) _argmaxWithProb(Float32List logits) {
    var best = 0;
    var bestVal = logits[0];
    for (var i = 1; i < logits.length; i++) {
      if (logits[i] > bestVal) {
        bestVal = logits[i];
        best = i;
      }
    }
    var sum = 0.0;
    for (var i = 0; i < logits.length; i++) {
      sum += math.exp(logits[i] - bestVal); // d <= 0, well-conditioned
    }
    return (token: best, prob: sum > 0 ? 1.0 / sum : 0.0);
  }
}

/// A plain token-only `llama_batch` wrapper (no embeddings) for the target
/// verify/prefill passes.
final class _TokenBatch {
  final int capacity;
  final llama_batch _batch;
  bool _disposed = false;

  _TokenBatch._(this.capacity, this._batch);

  factory _TokenBatch(int capacity) => _TokenBatch._(
      capacity, LlamaLibrary.bindings.llama_batch_init(capacity, 0, 1));

  llama_batch get raw => _batch;

  void clear() => _batch.n_tokens = 0;

  void add(int token, int pos, int seqId, {required bool wantLogits}) {
    final n = _batch.n_tokens;
    _batch.token[n] = token;
    _batch.pos[n] = pos;
    _batch.n_seq_id[n] = 1;
    _batch.seq_id[n][0] = seqId;
    _batch.logits[n] = wantLogits ? 1 : 0;
    _batch.n_tokens = n + 1;
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    LlamaLibrary.bindings.llama_batch_free(_batch);
  }
}
