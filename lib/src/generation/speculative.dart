import 'dart:math' as math;
import 'dart:typed_data';

import '../batch/batch.dart';
import '../context/context.dart';
import '../ffi/library_loader.dart';
import '../tokenizer/tokenizer.dart';
import '../types/exceptions.dart';

/// Outcome of a [SpeculativeDecoder.generate] run.
final class SpeculativeResult {
  /// Generated token ids, in order (excludes the prompt).
  final List<int> tokens;

  /// Detokenized text of [tokens].
  final String text;

  /// Total draft tokens proposed across all rounds.
  final int draftedCount;

  /// Total draft tokens the target accepted.
  final int acceptedCount;

  /// True if generation stopped on an end-of-generation token.
  final bool stoppedOnEog;

  const SpeculativeResult({
    required this.tokens,
    required this.text,
    required this.draftedCount,
    required this.acceptedCount,
    required this.stoppedOnEog,
  });

  /// Fraction of drafted tokens that were accepted (`0..1`). Higher means
  /// the draft tracks the target well and the speedup is larger. `0` when
  /// nothing was drafted.
  double get acceptanceRate =>
      draftedCount == 0 ? 0 : acceptedCount / draftedCount;

  @override
  String toString() => 'SpeculativeResult(${tokens.length} tokens, '
      'accept $acceptedCount/$draftedCount '
      '= ${(acceptanceRate * 100).toStringAsFixed(1)}%, '
      'eog=$stoppedOnEog)';
}

/// Synchronous, pure-Dart **greedy** speculative decoding over two
/// [LlamaContext]s that share a vocabulary.
///
/// [draft] proposes `draftLength` tokens per round; [target] verifies all of
/// them in a single `llama_decode`. The accepted prefix (every draft token
/// whose argmax matches the target's) plus one bonus token from the target
/// are committed each round, and the rejected tail is rolled back out of both
/// KV caches. The emitted sequence is **identical** to plain greedy decoding
/// on [target] alone — speculation only changes how many target forward
/// passes it takes to get there.
///
/// Requirements:
/// * [target] and [draft] must be built from models with the **same vocab**
///   (token ids must be comparable). A small + large model of the same family
///   is the usual pairing; passing two contexts of the *same* model is also
///   valid and yields ~100% acceptance (useful for correctness testing).
/// * [target] `nBatch >= draftLength + 1` (the verify batch).
/// * Both contexts have spare context room; generation stops early if a round
///   would overflow `nCtx`.
///
/// Supports both greedy ([temperature] == 0) and exact stochastic
/// speculative sampling ([temperature] > 0, full-vocab softmax). Top-k /
/// top-p truncation is not combined with the rejection rule yet. Runs on
/// the calling isolate and blocks until done.
final class SpeculativeDecoder {
  final LlamaContext target;
  final LlamaContext draft;

  SpeculativeDecoder({required this.target, required this.draft});

  /// Generate from [prompt]. Stops at [maxTokens] generated tokens, an
  /// end-of-generation token, or when the next round would overflow the
  /// smaller of the two contexts.
  ///
  /// [temperature] `== 0` runs greedy (deterministic; output is exactly
  /// plain greedy decoding on [target]). `> 0` runs exact stochastic
  /// speculative sampling: draft proposals are accepted with probability
  /// `min(1, p/q)` and rejections resample from the normalized residual
  /// `max(0, p - q)`, which is distributionally identical to sampling
  /// directly from [target] at that temperature. [seed] makes the
  /// stochastic path reproducible (`< 0` seeds from the clock).
  SpeculativeResult generate({
    required String prompt,
    int maxTokens = 256,
    int draftLength = 4,
    int seqId = 0,
    bool addSpecial = true,
    bool parseSpecial = true,
    double temperature = 0.0,
    int seed = -1,
  }) {
    if (prompt.isEmpty) {
      throw ArgumentError.value(prompt, 'prompt', 'must be non-empty');
    }
    if (draftLength < 1) {
      throw ArgumentError.value(draftLength, 'draftLength', 'must be >= 1');
    }
    if (temperature < 0) {
      throw ArgumentError.value(temperature, 'temperature', 'must be >= 0');
    }
    final stochastic = temperature > 0;
    final rng = stochastic ? math.Random(seed < 0 ? null : seed) : null;
    if (target.nBatch < draftLength + 1) {
      throw LlamaContextException(
        'target nBatch=${target.nBatch} is too small for draftLength='
        '$draftLength (need >= ${draftLength + 1})',
      );
    }
    if (target.model.vocab.nTokens != draft.model.vocab.nTokens) {
      throw LlamaContextException(
        'target and draft vocab sizes differ '
        '(${target.model.vocab.nTokens} vs ${draft.model.vocab.nTokens}); '
        'speculative decoding needs a shared vocabulary',
      );
    }

    final lib = LlamaLibrary.bindings;
    final vocab = target.model.vocab;
    final tokenizer = Tokenizer(vocab);

    final promptTokens = tokenizer.encode(
      prompt,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
    if (promptTokens.isEmpty) {
      throw const LlamaContextException('prompt tokenized to 0 tokens');
    }
    final maxPrefill =
        target.nBatch < draft.nBatch ? target.nBatch : draft.nBatch;
    if (promptTokens.length > maxPrefill) {
      throw LlamaContextException(
        'prompt has ${promptTokens.length} tokens which exceeds the smaller '
        'nBatch ($maxPrefill); chunked prefill is not implemented here',
      );
    }

    final nCtx = target.nCtx < draft.nCtx ? target.nCtx : draft.nCtx;

    // Fresh KV on both contexts for this sequence.
    final targetMem = lib.llama_get_memory(target.pointer);
    final draftMem = lib.llama_get_memory(draft.pointer);
    lib.llama_memory_seq_rm(targetMem, seqId, -1, -1);
    lib.llama_memory_seq_rm(draftMem, seqId, -1, -1);

    final generated = <int>[];
    var drafted = 0;
    var accepted = 0;
    var stoppedOnEog = false;

    final verifyBatch = LlamaBatch(target.nBatch);
    final draftBatch = LlamaBatch(draft.nBatch);

    try {
      // --- Prefill both contexts with the prompt. ---
      _prefill(
        context: target,
        batch: verifyBatch,
        tokens: promptTokens,
        seqId: seqId,
        wantLastLogits: true,
      );
      _prefill(
        context: draft,
        batch: draftBatch,
        tokens: promptTokens,
        seqId: seqId,
        wantLastLogits: false,
      );

      // First token comes straight from the target's prompt logits.
      final firstLogits = target.logitsAt(promptTokens.length - 1)!;
      var idLast = stochastic
          ? _sampleProbs(_softmax(firstLogits, temperature), rng!.nextDouble())
          : _argmax(firstLogits);
      var committedPos = promptTokens.length; // KV covers [0, committedPos)
      generated.add(idLast);
      if (vocab.isEog(idLast)) {
        stoppedOnEog = true;
      }

      while (!stoppedOnEog && generated.length < maxTokens) {
        // Cap the draft so we never run past the context window.
        var k = draftLength;
        if (committedPos + k + 1 >= nCtx) {
          k = nCtx - committedPos - 2;
        }
        if (k < 1) break;

        // --- Draft phase: autoregressively propose k tokens. ---
        final drafts = <int>[];
        // Stochastic mode keeps each step's full draft distribution so the
        // residual `max(0, p - q)` can be formed on rejection.
        final draftProbs = stochastic ? <Float64List>[] : null;
        var feed = idLast;
        var pos = committedPos;
        for (var i = 0; i < k; i++) {
          draftBatch.clear();
          draftBatch.add(feed, pos, [seqId], wantLogits: true);
          final rc = lib.llama_decode(draft.pointer, draftBatch.raw);
          if (rc != 0) {
            throw LlamaDecodeException(rc, 'draft decode failed');
          }
          final int next;
          if (stochastic) {
            final probs = _softmax(draft.lastLogits()!, temperature);
            next = _sampleProbs(probs, rng!.nextDouble());
            draftProbs!.add(probs);
          } else {
            next = _argmax(draft.lastLogits()!);
          }
          drafts.add(next);
          feed = next;
          pos++;
        }

        // --- Verify phase: target scores [idLast, d0..d{k-1}] in one pass. ---
        verifyBatch.clear();
        verifyBatch.add(idLast, committedPos, [seqId], wantLogits: true);
        for (var i = 0; i < k; i++) {
          verifyBatch.add(drafts[i], committedPos + 1 + i, [seqId],
              wantLogits: true);
        }
        final vrc = lib.llama_decode(target.pointer, verifyBatch.raw);
        if (vrc != 0) {
          throw LlamaDecodeException(vrc, 'target verify decode failed');
        }

        // Accept/reject each draft. Target logits at batch index j predict
        // draft token j.
        var acceptedThisRound = 0;
        var bonus = 0;
        var bonusSet = false;
        for (var j = 0; j < k; j++) {
          if (stochastic) {
            final p = _softmax(target.logitsAt(j)!, temperature);
            final q = draftProbs![j];
            final d = drafts[j];
            final qd = q[d];
            final ratio = qd <= 0 ? 1.0 : (p[d] / qd);
            if (rng!.nextDouble() < (ratio < 1.0 ? ratio : 1.0)) {
              acceptedThisRound++;
            } else {
              // Resample from the normalized residual max(0, p - q).
              bonus = _sampleResidual(p, q, rng.nextDouble());
              bonusSet = true;
              break;
            }
          } else {
            final pred = _argmax(target.logitsAt(j)!);
            if (pred == drafts[j]) {
              acceptedThisRound++;
            } else {
              bonus = pred; // target's own token at the divergence point
              bonusSet = true;
              break;
            }
          }
        }
        if (!bonusSet) {
          // All drafts accepted; the bonus is the target's own draw after
          // the last draft (batch index k).
          final lastLogits = target.logitsAt(k)!;
          bonus = stochastic
              ? _sampleProbs(_softmax(lastLogits, temperature), rng!.nextDouble())
              : _argmax(lastLogits);
        }

        drafted += k;
        accepted += acceptedThisRound;

        // --- Commit accepted drafts + bonus, honoring stop conditions. ---
        for (var j = 0; j < acceptedThisRound; j++) {
          if (generated.length >= maxTokens) break;
          generated.add(drafts[j]);
          if (vocab.isEog(drafts[j])) {
            stoppedOnEog = true;
            break;
          }
        }

        // New committed length: idLast + accepted drafts.
        final newCommitted = committedPos + 1 + acceptedThisRound;

        // Roll the rejected tail out of both KV caches.
        lib.llama_memory_seq_rm(targetMem, seqId, newCommitted, -1);
        lib.llama_memory_seq_rm(draftMem, seqId, newCommitted, -1);

        if (stoppedOnEog || generated.length >= maxTokens) {
          committedPos = newCommitted;
          break;
        }

        // Full-accept case: the draft proposed d[k-1] but only decoded
        // idLast..d[k-2] into its own KV, so the draft cache is one token
        // behind the committed sequence. Feed d[k-1] in (at its committed
        // position) to keep both caches in lockstep before the next round.
        if (acceptedThisRound == k) {
          draftBatch.clear();
          draftBatch.add(drafts[k - 1], committedPos + k, [seqId],
              wantLogits: false);
          final rc = lib.llama_decode(draft.pointer, draftBatch.raw);
          if (rc != 0) {
            throw LlamaDecodeException(rc, 'draft catch-up decode failed');
          }
        }

        committedPos = newCommitted;

        // The bonus token becomes the next idLast (a real target output).
        generated.add(bonus);
        idLast = bonus;
        if (vocab.isEog(bonus)) {
          stoppedOnEog = true;
        }
      }
    } finally {
      verifyBatch.dispose();
      draftBatch.dispose();
    }

    // Trim to maxTokens (a round may have appended the bonus past the cap).
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

  void _prefill({
    required LlamaContext context,
    required LlamaBatch batch,
    required List<int> tokens,
    required int seqId,
    required bool wantLastLogits,
  }) {
    batch.clear();
    for (var i = 0; i < tokens.length; i++) {
      batch.add(
        tokens[i],
        i,
        [seqId],
        wantLogits: wantLastLogits && i == tokens.length - 1,
      );
    }
    final rc = LlamaLibrary.bindings.llama_decode(context.pointer, batch.raw);
    if (rc != 0) {
      throw LlamaDecodeException(rc, 'speculative prefill decode failed');
    }
  }

  static int _argmax(Float32List logits) {
    var best = 0;
    var bestVal = logits[0];
    for (var i = 1; i < logits.length; i++) {
      if (logits[i] > bestVal) {
        bestVal = logits[i];
        best = i;
      }
    }
    return best;
  }

  /// Numerically-stable softmax of [logits] scaled by `1/temp`.
  static Float64List _softmax(Float32List logits, double temp) {
    final n = logits.length;
    final out = Float64List(n);
    var maxLogit = logits[0];
    for (var i = 1; i < n; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    final invT = 1.0 / temp;
    var sum = 0.0;
    for (var i = 0; i < n; i++) {
      final e = math.exp((logits[i] - maxLogit) * invT);
      out[i] = e;
      sum += e;
    }
    if (sum > 0) {
      final inv = 1.0 / sum;
      for (var i = 0; i < n; i++) {
        out[i] *= inv;
      }
    }
    return out;
  }

  /// Inverse-CDF sample from a normalized [probs] using uniform [r] in [0,1).
  static int _sampleProbs(Float64List probs, double r) {
    var cum = 0.0;
    for (var i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r < cum) return i;
    }
    return probs.length - 1; // floating-point slack
  }

  /// Sample from the normalized residual `max(0, p - q)`. Falls back to the
  /// argmax of [p] if the residual is all-zero (p and q identical).
  static int _sampleResidual(Float64List p, Float64List q, double r) {
    final n = p.length;
    var sum = 0.0;
    final res = Float64List(n);
    for (var i = 0; i < n; i++) {
      final d = p[i] - q[i];
      if (d > 0) {
        res[i] = d;
        sum += d;
      }
    }
    if (sum <= 0) {
      var best = 0;
      var bestVal = p[0];
      for (var i = 1; i < n; i++) {
        if (p[i] > bestVal) {
          bestVal = p[i];
          best = i;
        }
      }
      return best;
    }
    final target = r * sum;
    var cum = 0.0;
    for (var i = 0; i < n; i++) {
      cum += res[i];
      if (target < cum) return i;
    }
    return n - 1;
  }
}
