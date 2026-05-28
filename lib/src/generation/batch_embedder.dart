import 'dart:ffi';
import 'dart:math' as math;
import 'dart:typed_data';

import '../batch/batch.dart';
import '../context/context.dart';
import '../context/context_params.dart';
import '../ffi/library_loader.dart';
import '../tokenizer/tokenizer.dart';
import '../types/exceptions.dart';
import 'embedding.dart';

/// Synchronous, pure-Dart batched embedding over a [LlamaContext].
///
/// Embeds several texts in a **single** `llama_decode` pass by assigning
/// each text its own sequence id (`0 .. texts.length - 1`). On accelerated
/// backends the per-token compute is amortized across the batch, so this
/// is dramatically faster than embedding one text at a time — the typical
/// RAG-ingest pattern.
///
/// Runs on the calling isolate and blocks until the decode completes; for
/// a Flutter UI thread, drive it from your own worker isolate (or wait for
/// the `LlamaEngine` integration).
///
/// Requirements on the [context]:
/// * created with `ContextParams(embeddings: true)`,
/// * a pooled pooling type (`mean` / `cls` / `last` / `rank`) — unpooled
///   (`none`) batching is rejected; use `LlamaEngine.embed` per text,
/// * `nSeqMax >= texts.length` for any single [embed] call,
/// * `nBatch >=` the total token count of one [embed] call.
final class BatchEmbedder {
  final LlamaContext context;

  BatchEmbedder(this.context);

  /// Embed [texts] in one pass. Returns one [EmbeddingResult] per input,
  /// in the same order. Each result is a single pooled vector.
  ///
  /// [normalize] L2-normalizes each vector in place (cosine-ready).
  List<EmbeddingResult> embed(
    List<String> texts, {
    bool addSpecial = true,
    bool parseSpecial = true,
    bool normalize = true,
  }) {
    if (texts.isEmpty) return const <EmbeddingResult>[];

    if (!context.params.embeddings) {
      throw const LlamaContextException(
        'BatchEmbedder requires a context built with '
        'ContextParams(embeddings: true)',
      );
    }
    if (context.poolingType == PoolingType.none) {
      throw const LlamaContextException(
        'BatchEmbedder needs a pooled pooling type (mean/cls/last/rank); '
        'this context resolved to none. Use LlamaEngine.embed per text for '
        'unpooled embeddings.',
      );
    }
    if (texts.length > context.nSeqMax) {
      throw LlamaContextException(
        'BatchEmbedder got ${texts.length} texts but the context only '
        'allows nSeqMax=${context.nSeqMax} sequences',
      );
    }

    final lib = LlamaLibrary.bindings;
    final model = context.model;
    final tokenizer = Tokenizer(model.vocab);

    final perText = <List<int>>[];
    var total = 0;
    for (final t in texts) {
      final toks = tokenizer.encode(
        t,
        addSpecial: addSpecial,
        parseSpecial: parseSpecial,
      );
      if (toks.isEmpty) {
        throw LlamaContextException(
          'BatchEmbedder: an input tokenized to 0 tokens',
        );
      }
      perText.add(toks);
      total += toks.length;
    }

    if (total > context.nBatch) {
      throw LlamaContextException(
        'BatchEmbedder: $total total tokens exceed nBatch=${context.nBatch}; '
        'split the inputs into smaller groups or raise nBatch',
      );
    }

    // Fresh KV for every sequence slot we are about to write.
    final mem = lib.llama_get_memory(context.pointer);
    for (var i = 0; i < texts.length; i++) {
      lib.llama_memory_seq_rm(mem, i, -1, -1);
    }

    final batch = LlamaBatch(context.nBatch);
    try {
      for (var seq = 0; seq < perText.length; seq++) {
        final toks = perText[seq];
        for (var pos = 0; pos < toks.length; pos++) {
          batch.add(toks[pos], pos, [seq], wantLogits: true);
        }
      }

      final usesEncoder = model.hasEncoder && !model.hasDecoder;
      final rc = usesEncoder
          ? lib.llama_encode(context.pointer, batch.raw)
          : lib.llama_decode(context.pointer, batch.raw);
      if (rc != 0) {
        throw LlamaDecodeException(
          rc,
          '${usesEncoder ? 'llama_encode' : 'llama_decode'} '
          'failed during batched embed',
        );
      }

      final nEmbd = model.nEmbd;
      final poolingInt = _poolingTypeToInt(context.poolingType);

      final out = <EmbeddingResult>[];
      for (var seq = 0; seq < perText.length; seq++) {
        final ptr = lib.llama_get_embeddings_seq(context.pointer, seq);
        if (ptr == nullptr) {
          throw LlamaContextException(
            'llama_get_embeddings_seq returned null for seq $seq',
          );
        }
        final vec = Float32List.fromList(ptr.asTypedList(nEmbd));
        if (normalize) _l2NormInPlace(vec);
        out.add(EmbeddingResult(
          nEmbd: nEmbd,
          nTokens: perText[seq].length,
          pooled: true,
          poolingType: poolingInt,
          normalized: normalize,
          vector: vec,
        ));
      }
      return out;
    } finally {
      batch.dispose();
    }
  }
}

/// Map the Dart [PoolingType] to the `llama_pooling_type` integer convention
/// used by [EmbeddingResult.poolingType] (`-1 auto, 0 none, 1 mean, 2 cls,
/// 3 last, 4 rank`).
int _poolingTypeToInt(PoolingType p) => switch (p) {
      PoolingType.auto => -1,
      PoolingType.none => 0,
      PoolingType.mean => 1,
      PoolingType.cls => 2,
      PoolingType.last => 3,
      PoolingType.rank => 4,
    };

void _l2NormInPlace(Float32List v) {
  var sum = 0.0;
  for (final x in v) {
    sum += x * x;
  }
  if (sum <= 0) return;
  final norm = 1.0 / math.sqrt(sum);
  for (var i = 0; i < v.length; i++) {
    v[i] = v[i] * norm;
  }
}
