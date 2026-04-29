/// Three-state flag for FlashAttention.
enum FlashAttention {
  /// Use whatever the model/runtime decides (current default).
  auto,

  /// Force FlashAttention off.
  off,

  /// Force FlashAttention on. Errors if the backend cannot honour it.
  on,
}

/// Tensor types usable for the K/V cache. Mirrors a curated subset of
/// `ggml_type`. `f16` is the llama.cpp default; `q8_0` roughly halves
/// KV-cache memory at a small quality cost and is the common choice on
/// memory-constrained mobile devices.
enum KvCacheType {
  f32,
  f16,
  bf16,
  q8_0,
  q4_0,
  q4_1,
  q5_0,
  q5_1,
}

/// Declarative configuration for [LlamaContext.create].
///
/// Immutable, JSON-friendly. Holds no native memory.
final class ContextParams {
  /// Total context window in tokens.
  final int nCtx;

  /// Maximum logical batch size (prompt-time prefill batch).
  final int nBatch;

  /// Maximum physical batch size (per-decode submit).
  final int nUbatch;

  /// Maximum number of concurrent sequences.
  final int nSeqMax;

  /// Generation thread count. `0` lets llama.cpp pick.
  final int nThreads;

  /// Prompt-prefill thread count. `0` lets llama.cpp pick.
  final int nThreadsBatch;

  /// FlashAttention mode.
  final FlashAttention flashAttn;

  /// Offload K/Q/V matmul ops to GPU when applicable.
  final bool offloadKqv;

  /// Configure context for embedding extraction.
  final bool embeddings;

  /// K-cache tensor type. Defaults to `f16` (llama.cpp default).
  final KvCacheType typeK;

  /// V-cache tensor type. Defaults to `f16` (llama.cpp default).
  /// FlashAttention requires `typeK == typeV` on most backends.
  final KvCacheType typeV;

  /// Random seed used internally for stateful ops. -1 picks a runtime value.
  final int seed;

  const ContextParams({
    this.nCtx = 4096,
    this.nBatch = 2048,
    this.nUbatch = 512,
    this.nSeqMax = 1,
    this.nThreads = 0,
    this.nThreadsBatch = 0,
    this.flashAttn = FlashAttention.auto,
    this.offloadKqv = true,
    this.embeddings = false,
    this.typeK = KvCacheType.f16,
    this.typeV = KvCacheType.f16,
    this.seed = -1,
  });

  ContextParams copyWith({
    int? nCtx,
    int? nBatch,
    int? nUbatch,
    int? nSeqMax,
    int? nThreads,
    int? nThreadsBatch,
    FlashAttention? flashAttn,
    bool? offloadKqv,
    bool? embeddings,
    KvCacheType? typeK,
    KvCacheType? typeV,
    int? seed,
  }) {
    return ContextParams(
      nCtx: nCtx ?? this.nCtx,
      nBatch: nBatch ?? this.nBatch,
      nUbatch: nUbatch ?? this.nUbatch,
      nSeqMax: nSeqMax ?? this.nSeqMax,
      nThreads: nThreads ?? this.nThreads,
      nThreadsBatch: nThreadsBatch ?? this.nThreadsBatch,
      flashAttn: flashAttn ?? this.flashAttn,
      offloadKqv: offloadKqv ?? this.offloadKqv,
      embeddings: embeddings ?? this.embeddings,
      typeK: typeK ?? this.typeK,
      typeV: typeV ?? this.typeV,
      seed: seed ?? this.seed,
    );
  }

  Map<String, Object?> toJson() => {
        'nCtx': nCtx,
        'nBatch': nBatch,
        'nUbatch': nUbatch,
        'nSeqMax': nSeqMax,
        'nThreads': nThreads,
        'nThreadsBatch': nThreadsBatch,
        'flashAttn': flashAttn.name,
        'offloadKqv': offloadKqv,
        'embeddings': embeddings,
        'typeK': typeK.name,
        'typeV': typeV.name,
        'seed': seed,
      };

  factory ContextParams.fromJson(Map<String, Object?> json) => ContextParams(
        nCtx: (json['nCtx'] as int?) ?? 4096,
        nBatch: (json['nBatch'] as int?) ?? 2048,
        nUbatch: (json['nUbatch'] as int?) ?? 512,
        nSeqMax: (json['nSeqMax'] as int?) ?? 1,
        nThreads: (json['nThreads'] as int?) ?? 0,
        nThreadsBatch: (json['nThreadsBatch'] as int?) ?? 0,
        flashAttn: FlashAttention.values.firstWhere(
          (e) => e.name == json['flashAttn'],
          orElse: () => FlashAttention.auto,
        ),
        offloadKqv: (json['offloadKqv'] as bool?) ?? true,
        embeddings: (json['embeddings'] as bool?) ?? false,
        typeK: KvCacheType.values.firstWhere(
          (e) => e.name == json['typeK'],
          orElse: () => KvCacheType.f16,
        ),
        typeV: KvCacheType.values.firstWhere(
          (e) => e.name == json['typeV'],
          orElse: () => KvCacheType.f16,
        ),
        seed: (json['seed'] as int?) ?? -1,
      );
}
