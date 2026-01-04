import 'dart:convert';
import '../llama_cpp_dart.dart';
import 'llama_cpp.dart';

/// GGML Types for KV Cache Quantization (ggml_type)
enum LlamaKvCacheType {
  f32(0),
  f16(1),
  q4_0(2),
  q4_1(3),
  q5_0(6),
  q5_1(7),
  q8_0(8);

  final int value;
  const LlamaKvCacheType(this.value);
}

/// Flash Attention Type (llama_flash_attn_type)
enum LlamaFlashAttnType {
  disabled(0),
  enabled(1);

  final int value;
  const LlamaFlashAttnType(this.value);
}

enum LlamaRopeScalingType {
  unspecified(-1),
  none(0),
  linear(1),
  yarn(2),
  longrope(3),
  maxValue(3);

  final int value;
  const LlamaRopeScalingType(this.value);
}

enum LlamaPoolingType {
  unspecified(-1),
  none(0),
  mean(1),
  cls(2),
  last(3),
  rank(4);

  final int value;
  const LlamaPoolingType(this.value);
}

enum LlamaAttentionType {
  unspecified(-1),
  causal(0),
  nonCausal(1);

  final int value;
  const LlamaAttentionType(this.value);
}

class ContextParams {
  /// Generation Config (App level)
  int nPredict = -1;

  int nCtx = 512;
  int nBatch = 512;
  int nUbatch = 512;
  int nSeqMax = 1;
  int nThreads = 8;
  int nThreadsBatch = 8;

  LlamaRopeScalingType ropeScalingType = LlamaRopeScalingType.unspecified;
  LlamaPoolingType poolingType = LlamaPoolingType.unspecified;
  LlamaAttentionType attentionType = LlamaAttentionType.unspecified;
  
  LlamaFlashAttnType flashAttention = LlamaFlashAttnType.disabled;

  LlamaKvCacheType typeK = LlamaKvCacheType.f16;
  LlamaKvCacheType typeV = LlamaKvCacheType.f16;

  bool embeddings = false;
  bool offloadKqv = true;
  bool noPerfTimings = false;
  
  bool opOffload = true;
  bool swaFull = false;
  bool kvUnified = false;
  bool autoTrimContext = false;
  int trimKeepTokens = 256;

  double ropeFreqBase = 0.0;
  double ropeFreqScale = 0.0;
  double yarnExtFactor = -1.0;
  double yarnAttnFactor = 1.0;
  double yarnBetaFast = 32.0;
  double yarnBetaSlow = 1.0;
  int yarnOrigCtx = 0;
  double defragThold = -1.0;

  ContextParams();

  llama_context_params get() {
    final params = Llama.lib.llama_context_default_params();

    params.n_ctx = nCtx;
    params.n_batch = nBatch;
    params.n_ubatch = nUbatch;
    params.n_seq_max = nSeqMax;
    params.n_threads = nThreads;
    params.n_threads_batch = nThreadsBatch;

    params.rope_scaling_typeAsInt = ropeScalingType.value;
    params.pooling_typeAsInt = poolingType.value;
    params.attention_typeAsInt = attentionType.value;
    
    params.flash_attn_typeAsInt = flashAttention.value;

    params.rope_freq_base = ropeFreqBase;
    params.rope_freq_scale = ropeFreqScale;
    params.yarn_ext_factor = yarnExtFactor;
    params.yarn_attn_factor = yarnAttnFactor;
    params.yarn_beta_fast = yarnBetaFast;
    params.yarn_beta_slow = yarnBetaSlow;
    params.yarn_orig_ctx = yarnOrigCtx;
    params.defrag_thold = defragThold;

    params.embeddings = embeddings;
    params.offload_kqv = offloadKqv;
    params.no_perf = noPerfTimings;
    
    params.op_offload = opOffload;
    params.swa_full = swaFull;
    params.kv_unified = kvUnified;

    params.type_kAsInt = typeK.value;
    params.type_vAsInt = typeV.value;

    return params;
  }

  factory ContextParams.fromJson(Map<String, dynamic> json) {
    return ContextParams()
      ..nPredict = json['nPredict'] ?? -1
      ..nCtx = json['nCtx'] ?? 512
      ..nBatch = json['nBatch'] ?? 512
      ..nUbatch = json['nUbatch'] ?? 512
      ..nSeqMax = json['nSeqMax'] ?? 1
      ..nThreads = json['nThreads'] ?? 8
      ..nThreadsBatch = json['nThreadsBatch'] ?? 8
      
      ..ropeScalingType = LlamaRopeScalingType.values.firstWhere(
          (e) => e.value == (json['ropeScalingType'] ?? -1),
          orElse: () => LlamaRopeScalingType.unspecified)
      ..poolingType = LlamaPoolingType.values.firstWhere(
          (e) => e.value == (json['poolingType'] ?? -1),
          orElse: () => LlamaPoolingType.unspecified)
      ..attentionType = LlamaAttentionType.values.firstWhere(
          (e) => e.value == (json['attentionType'] ?? -1),
          orElse: () => LlamaAttentionType.unspecified)
      
      ..flashAttention = LlamaFlashAttnType.values.firstWhere(
          (e) => e.value == (json['flashAttention'] ?? 0),
          orElse: () => LlamaFlashAttnType.disabled)
      ..typeK = LlamaKvCacheType.values.firstWhere(
          (e) => e.value == (json['typeK'] ?? 1),
          orElse: () => LlamaKvCacheType.f16)
      ..typeV = LlamaKvCacheType.values.firstWhere(
          (e) => e.value == (json['typeV'] ?? 1),
          orElse: () => LlamaKvCacheType.f16)

      ..ropeFreqBase = json['ropeFreqBase']?.toDouble() ?? 0.0
      ..ropeFreqScale = json['ropeFreqScale']?.toDouble() ?? 0.0
      ..yarnExtFactor = json['yarnExtFactor']?.toDouble() ?? -1.0
      ..yarnAttnFactor = json['yarnAttnFactor']?.toDouble() ?? 1.0
      ..yarnBetaFast = json['yarnBetaFast']?.toDouble() ?? 32.0
      ..yarnBetaSlow = json['yarnBetaSlow']?.toDouble() ?? 1.0
      ..yarnOrigCtx = json['yarnOrigCtx'] ?? 0
      ..defragThold = json['defragThold']?.toDouble() ?? -1.0
      
      ..embeddings = json['embeddings'] ?? false
      ..offloadKqv = json['offloadKqv'] ?? true
      ..noPerfTimings = json['noPerfTimings'] ?? false
      ..opOffload = json['opOffload'] ?? true
      ..swaFull = json['swaFull'] ?? false
      ..kvUnified = json['kvUnified'] ?? false
      ..autoTrimContext = json['autoTrimContext'] ?? false
      ..trimKeepTokens = json['trimKeepTokens'] ?? 256;
  }

  Map<String, dynamic> toJson() => {
        'nPredict': nPredict,
        'nCtx': nCtx,
        'nBatch': nBatch,
        'nUbatch': nUbatch,
        'nSeqMax': nSeqMax,
        'nThreads': nThreads,
        'nThreadsBatch': nThreadsBatch,
        'ropeScalingType': ropeScalingType.value,
        'poolingType': poolingType.value,
        'attentionType': attentionType.value,
        
        'flashAttention': flashAttention.value,
        'typeK': typeK.value,
        'typeV': typeV.value,
        'opOffload': opOffload,
        'swaFull': swaFull,
        'kvUnified': kvUnified,
        
        'ropeFreqBase': ropeFreqBase,
        'ropeFreqScale': ropeFreqScale,
        'yarnExtFactor': yarnExtFactor,
        'yarnAttnFactor': yarnAttnFactor,
        'yarnBetaFast': yarnBetaFast,
        'yarnBetaSlow': yarnBetaSlow,
        'yarnOrigCtx': yarnOrigCtx,
        'defragThold': defragThold,
        'embeddings': embeddings,
        'offloadKqv': offloadKqv,
        'noPerfTimings': noPerfTimings,
        'autoTrimContext': autoTrimContext,
        'trimKeepTokens': trimKeepTokens,
      };

  @override
  String toString() => jsonEncode(toJson());
}
