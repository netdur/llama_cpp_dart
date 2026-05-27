import 'dart:typed_data';

/// Result of an `LlamaEngine.embed` call.
final class EmbeddingResult {
  /// Embedding dimension produced by the model.
  final int nEmbd;

  /// Number of input tokens after tokenization.
  final int nTokens;

  /// True when the runtime returned a single pooled vector for the sequence
  /// (pooling type `mean`, `cls`, `last`, or `rank`). When false, [vector]
  /// holds one row of `nEmbd` floats per input token, in token order.
  final bool pooled;

  /// Effective pooling type the context reported at runtime. Matches the
  /// `llama_pooling_type` enum values (`0 = none, 1 = mean, 2 = cls, 3 = last,
  /// 4 = rank`). Useful when [PoolingType.auto] was requested and you want
  /// to know what the model picked.
  final int poolingType;

  /// Whether the values were L2-normalized before being returned.
  final bool normalized;

  /// Raw embedding values.
  ///
  /// * Pooled: a single `nEmbd`-length vector.
  /// * Unpooled: `nTokens * nEmbd` floats, row-major (token `i` starts at
  ///   offset `i * nEmbd`). Use [tokenEmbedding] for a per-token view.
  final Float32List vector;

  const EmbeddingResult({
    required this.nEmbd,
    required this.nTokens,
    required this.pooled,
    required this.poolingType,
    required this.normalized,
    required this.vector,
  });

  /// Pooled: returns [vector] (only valid for `i == 0`).
  ///
  /// Unpooled: returns a zero-copy view of the embedding for token [i].
  Float32List tokenEmbedding(int i) {
    if (pooled) {
      if (i != 0) {
        throw RangeError(
          'pooled embedding has a single vector (requested index $i)',
        );
      }
      return vector;
    }
    if (i < 0 || i >= nTokens) {
      throw RangeError.range(i, 0, nTokens - 1, 'i');
    }
    return Float32List.sublistView(vector, i * nEmbd, (i + 1) * nEmbd);
  }
}
