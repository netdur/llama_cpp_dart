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

/// RoPE scaling strategy. `auto` (the runtime default) lets the model decide.
enum RopeScalingType {
  auto,
  none,
  linear,
  yarn,
  longrope,
}

/// Embedding pooling strategy. `auto` lets the runtime / model decide.
enum PoolingType {
  auto,
  none,
  mean,
  cls,
  last,
  rank,
}

/// Attention type for embedding workloads. `auto` keeps the runtime default.
enum AttentionType {
  auto,
  causal,
  nonCausal,
}

/// Declarative configuration for [LlamaContext.create].
///
/// Immutable, JSON-friendly. Holds no native memory.
final class ContextParams {
  /// Total context window in tokens. `0` = use the model's training value.
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

  /// RoPE scaling type. `auto` keeps the model default.
  final RopeScalingType ropeScalingType;

  /// Embedding pooling strategy.
  final PoolingType poolingType;

  /// Attention type (only meaningful for embedding models).
  final AttentionType attentionType;

  /// RoPE base frequency. `0.0` = read from the model.
  final double ropeFreqBase;

  /// RoPE frequency scaling factor. `0.0` = read from the model.
  final double ropeFreqScale;

  /// YaRN extrapolation mix factor. Negative = read from model.
  final double yarnExtFactor;

  /// YaRN magnitude scaling factor.
  final double yarnAttnFactor;

  /// YaRN low correction dim.
  final double yarnBetaFast;

  /// YaRN high correction dim.
  final double yarnBetaSlow;

  /// YaRN original context size.
  final int yarnOrigCtx;

  /// KV-cache defragmentation threshold (`holes / size`). `<= 0` disables.
  /// Marked deprecated upstream but still wired through.
  final double defragThreshold;

  /// Offload K/Q/V matmul ops to GPU when applicable.
  final bool offloadKqv;

  /// Configure context for embedding extraction.
  final bool embeddings;

  /// Disable internal performance timing collection.
  final bool noPerf;

  /// Offload host tensor operations to the device.
  final bool opOffload;

  /// Use a full-size sliding-window-attention cache (helps multi-sequence
  /// workloads at the cost of memory).
  final bool swaFull;

  /// Use a unified attention buffer across input sequences. Disable for
  /// long, divergent sequences.
  final bool kvUnified;

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
    this.ropeScalingType = RopeScalingType.auto,
    this.poolingType = PoolingType.auto,
    this.attentionType = AttentionType.auto,
    this.ropeFreqBase = 0.0,
    this.ropeFreqScale = 0.0,
    this.yarnExtFactor = -1.0,
    this.yarnAttnFactor = 1.0,
    this.yarnBetaFast = 32.0,
    this.yarnBetaSlow = 1.0,
    this.yarnOrigCtx = 0,
    this.defragThreshold = -1.0,
    this.offloadKqv = true,
    this.embeddings = false,
    this.noPerf = true,
    this.opOffload = true,
    this.swaFull = true,
    this.kvUnified = false,
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
    RopeScalingType? ropeScalingType,
    PoolingType? poolingType,
    AttentionType? attentionType,
    double? ropeFreqBase,
    double? ropeFreqScale,
    double? yarnExtFactor,
    double? yarnAttnFactor,
    double? yarnBetaFast,
    double? yarnBetaSlow,
    int? yarnOrigCtx,
    double? defragThreshold,
    bool? offloadKqv,
    bool? embeddings,
    bool? noPerf,
    bool? opOffload,
    bool? swaFull,
    bool? kvUnified,
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
      ropeScalingType: ropeScalingType ?? this.ropeScalingType,
      poolingType: poolingType ?? this.poolingType,
      attentionType: attentionType ?? this.attentionType,
      ropeFreqBase: ropeFreqBase ?? this.ropeFreqBase,
      ropeFreqScale: ropeFreqScale ?? this.ropeFreqScale,
      yarnExtFactor: yarnExtFactor ?? this.yarnExtFactor,
      yarnAttnFactor: yarnAttnFactor ?? this.yarnAttnFactor,
      yarnBetaFast: yarnBetaFast ?? this.yarnBetaFast,
      yarnBetaSlow: yarnBetaSlow ?? this.yarnBetaSlow,
      yarnOrigCtx: yarnOrigCtx ?? this.yarnOrigCtx,
      defragThreshold: defragThreshold ?? this.defragThreshold,
      offloadKqv: offloadKqv ?? this.offloadKqv,
      embeddings: embeddings ?? this.embeddings,
      noPerf: noPerf ?? this.noPerf,
      opOffload: opOffload ?? this.opOffload,
      swaFull: swaFull ?? this.swaFull,
      kvUnified: kvUnified ?? this.kvUnified,
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
        'ropeScalingType': ropeScalingType.name,
        'poolingType': poolingType.name,
        'attentionType': attentionType.name,
        'ropeFreqBase': ropeFreqBase,
        'ropeFreqScale': ropeFreqScale,
        'yarnExtFactor': yarnExtFactor,
        'yarnAttnFactor': yarnAttnFactor,
        'yarnBetaFast': yarnBetaFast,
        'yarnBetaSlow': yarnBetaSlow,
        'yarnOrigCtx': yarnOrigCtx,
        'defragThreshold': defragThreshold,
        'offloadKqv': offloadKqv,
        'embeddings': embeddings,
        'noPerf': noPerf,
        'opOffload': opOffload,
        'swaFull': swaFull,
        'kvUnified': kvUnified,
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
        ropeScalingType: RopeScalingType.values.firstWhere(
          (e) => e.name == json['ropeScalingType'],
          orElse: () => RopeScalingType.auto,
        ),
        poolingType: PoolingType.values.firstWhere(
          (e) => e.name == json['poolingType'],
          orElse: () => PoolingType.auto,
        ),
        attentionType: AttentionType.values.firstWhere(
          (e) => e.name == json['attentionType'],
          orElse: () => AttentionType.auto,
        ),
        ropeFreqBase: (json['ropeFreqBase'] as num?)?.toDouble() ?? 0.0,
        ropeFreqScale: (json['ropeFreqScale'] as num?)?.toDouble() ?? 0.0,
        yarnExtFactor: (json['yarnExtFactor'] as num?)?.toDouble() ?? -1.0,
        yarnAttnFactor: (json['yarnAttnFactor'] as num?)?.toDouble() ?? 1.0,
        yarnBetaFast: (json['yarnBetaFast'] as num?)?.toDouble() ?? 32.0,
        yarnBetaSlow: (json['yarnBetaSlow'] as num?)?.toDouble() ?? 1.0,
        yarnOrigCtx: (json['yarnOrigCtx'] as int?) ?? 0,
        defragThreshold: (json['defragThreshold'] as num?)?.toDouble() ?? -1.0,
        offloadKqv: (json['offloadKqv'] as bool?) ?? true,
        embeddings: (json['embeddings'] as bool?) ?? false,
        noPerf: (json['noPerf'] as bool?) ?? true,
        opOffload: (json['opOffload'] as bool?) ?? true,
        swaFull: (json['swaFull'] as bool?) ?? true,
        kvUnified: (json['kvUnified'] as bool?) ?? false,
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
