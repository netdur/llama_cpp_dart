import 'dart:math';
import 'dart:convert';

import 'llama_cpp.dart';
import 'llama.dart';

/// An enumeration representing different types of Llama Rope Scaling.
enum LlamaRopeScalingType { unspecified, none, linear, yarn, maxValue }

/// ContextParams holds configuration settings for the Llama model.
///
/// This class allows for specifying various parameters that affect
/// the operation of the Llama model such as random number generation,
/// batch sizes, thread usage, and specific algorithmic settings.
class ContextParams {
  /// Seed for random number generation. Defaults to a random integer.
  /// A value of -1 indicates a random seed should be used.
  int seed = Random().nextInt(1000000);

  /// Text context size. Defaults to 512.
  /// A value of 0 will use the model's default setting.
  int context = 512;

  /// Maximum batch size for prompt processing. Defaults to 512.
  int batch = 512;

  /// Number of threads to use for generation. Defaults to 8.
  int threads = 8;

  /// Number of threads to use for batch processing. Defaults to 8.
  int threadsBatch = 8;

  /// Type of RoPE (Rotary Positional Embedding) scaling to use.
  /// Defaults to `LlamaRopeScalingType.unspecified`.
  LlamaRopeScalingType ropeScalingType = LlamaRopeScalingType.unspecified;

  /// Base frequency for RoPE (Rotary Positional Embedding).
  /// Defaults to 0.0, which means the model's default is used.
  double ropeFreqBase = 0.0;

  /// Frequency scaling factor for RoPE.
  /// A value of 0.0 indicates the model's default scaling is used.
  double ropeFreqScale = 0.0;

  /// YaRN (Yet another Representation Network) extrapolation mix factor.
  /// Negative values indicate the model's default setting is used.
  /// Defaults to -1.0.
  double yarnExtFactor = -1.0;

  /// YaRN attention magnitude scaling factor. Defaults to 1.0.
  double yarnAttnFactor = 1.0;

  /// YaRN low correction dimension. Defaults to 32.0.
  double yarnBetaFast = 32.0;

  /// YaRN high correction dimension. Defaults to 1.0.
  double yarnBetaSlow = 1.0;

  /// Original context size for YaRN. Defaults to 0.
  int yarnOrigCtx = 0;

  /// Flag to use experimental 'mul_mat_q' kernels.
  /// Deprecated and always true.
  bool mulMatQ = true;

  /// If true, `llama_eval()` computes all logits, not just the last one.
  /// Deprecated - set `llama_batch.logits` instead.
  bool logitsAll = false;

  /// If true, operates in embedding mode only.
  bool embedding = false;

  /// Determines whether to offload the KQV operations to GPU.
  /// Defaults to true.
  bool offloadKqv = true;

  ContextParams();

  /// Constructs and returns a `llama_context_params` object
  /// with current settings of this `ContextParams` instance.
  llama_context_params get() {
    llama_context_params contextParams =
        Llama.lib.llama_context_default_params();
    contextParams.seed = seed;
    contextParams.n_ctx = context;
    contextParams.n_batch = batch;
    contextParams.n_threads = threads;
    contextParams.n_threads_batch = threadsBatch;
    contextParams.rope_scaling_type = ropeScalingType.index;
    contextParams.rope_freq_base = ropeFreqBase;
    contextParams.rope_freq_scale = ropeFreqScale;
    contextParams.yarn_ext_factor = yarnExtFactor;
    contextParams.yarn_attn_factor = yarnAttnFactor;
    contextParams.yarn_beta_fast = yarnBetaFast;
    contextParams.yarn_beta_slow = yarnBetaSlow;
    contextParams.yarn_orig_ctx = yarnOrigCtx;
    contextParams.mul_mat_q = mulMatQ;
    contextParams.logits_all = logitsAll;
    contextParams.embedding = embedding;
    contextParams.offload_kqv = offloadKqv;
    return contextParams;
  }

  /// Factory constructor to create a `ContextParams` instance from JSON.
  factory ContextParams.fromJson(Map<String, dynamic> json) {
    ContextParams contextParams = ContextParams();
    contextParams.seed = json['seed'] ?? -1;
    contextParams.context = json['context'] ?? 512;
    contextParams.batch = json['batch'] ?? 512;
    contextParams.threads = json['threads'] ?? -1;
    contextParams.threadsBatch = json['threadsBatch'] ?? -1;
    contextParams.ropeScalingType =
        LlamaRopeScalingType.values[json['ropeScalingType'] ?? 0];
    contextParams.ropeFreqBase = json['ropeFreqBase']?.toDouble() ?? 0.0;
    contextParams.ropeFreqScale = json['ropeFreqScale']?.toDouble() ?? 0.0;
    contextParams.yarnExtFactor = json['yarnExtFactor']?.toDouble() ?? -1.0;
    contextParams.yarnAttnFactor = json['yarnAttnFactor']?.toDouble() ?? 1.0;
    contextParams.yarnBetaFast = json['yarnBetaFast']?.toDouble() ?? 32.0;
    contextParams.yarnBetaSlow = json['yarnBetaSlow']?.toDouble() ?? 1.0;
    contextParams.yarnOrigCtx = json['yarnOrigCtx'] ?? 0;
    contextParams.mulMatQ = json['mulMatQ'] ?? true;
    contextParams.logitsAll = json['logitsAll'] ?? false;
    contextParams.embedding = json['embedding'] ?? false;
    contextParams.offloadKqv = json['offloadKqv'] ?? true;
    return contextParams;
  }

  /// Converts the `ContextParams` instance to a JSON map.
  Map<String, dynamic> toJson() => {
        'seed': seed,
        'context': context,
        'batch': batch,
        'threads': threads,
        'threadsBatch': threadsBatch,
        'ropeScalingType': ropeScalingType.index,
        'ropeFreqBase': ropeFreqBase,
        'ropeFreqScale': ropeFreqScale,
        'yarnExtFactor': yarnExtFactor,
        'yarnAttnFactor': yarnAttnFactor,
        'yarnBetaFast': yarnBetaFast,
        'yarnBetaSlow': yarnBetaSlow,
        'yarnOrigCtx': yarnOrigCtx,
        'mulMatQ': mulMatQ,
        'logitsAll': logitsAll,
        'embedding': embedding,
        'offloadKqv': offloadKqv,
      };

  /// Returns a string representation of the `ContextParams` instance.
  ///
  /// Useful for debugging and logging.
  @override
  String toString() {
    return jsonEncode(toJson());
  }
}
