/// Performance counters reported by llama.cpp's internal timers.
///
/// Mirrors `struct llama_perf_context_data`. Times are in milliseconds and
/// counts are token-level. Sampled via [LlamaContext.perf]; reset via
/// [LlamaContext.resetPerf].
final class ContextPerf {
  /// Microsecond clock reading when the context started timing.
  final double startMs;

  /// Time spent in `llama_model_load_from_file`. Only populated when the
  /// model was loaded by this context's library lifetime.
  final double loadMs;

  /// Cumulative wall time spent prefilling prompts (large batches).
  final double promptEvalMs;

  /// Cumulative wall time spent decoding generated tokens (small batches).
  final double evalMs;

  /// Number of prompt tokens evaluated. Pairs with [promptEvalMs].
  final int promptTokens;

  /// Number of generated tokens evaluated. Pairs with [evalMs].
  final int generatedTokens;

  /// Number of tokens served from the reused KV cache prefix (prompt cache).
  final int reusedTokens;

  const ContextPerf({
    required this.startMs,
    required this.loadMs,
    required this.promptEvalMs,
    required this.evalMs,
    required this.promptTokens,
    required this.generatedTokens,
    required this.reusedTokens,
  });

  /// Convenience: prompt tokens per second. `0` when no prompt was eval'd.
  double get promptTokensPerSecond =>
      promptEvalMs <= 0 ? 0 : promptTokens * 1000.0 / promptEvalMs;

  /// Convenience: generated tokens per second. `0` when nothing has been
  /// generated yet.
  double get generatedTokensPerSecond =>
      evalMs <= 0 ? 0 : generatedTokens * 1000.0 / evalMs;

  @override
  String toString() =>
      'ContextPerf(prompt: $promptTokens tok / ${promptEvalMs.toStringAsFixed(2)}ms '
      '(${promptTokensPerSecond.toStringAsFixed(1)} tok/s), '
      'gen: $generatedTokens tok / ${evalMs.toStringAsFixed(2)}ms '
      '(${generatedTokensPerSecond.toStringAsFixed(1)} tok/s), '
      'reused: $reusedTokens)';
}

/// Performance counters reported by a sampler chain. Mirrors
/// `struct llama_perf_sampler_data`.
final class SamplerPerf {
  /// Cumulative wall time spent sampling.
  final double sampleMs;

  /// Number of sampling calls made.
  final int nSample;

  const SamplerPerf({
    required this.sampleMs,
    required this.nSample,
  });

  /// Convenience: average milliseconds per sample call.
  double get msPerSample => nSample == 0 ? 0 : sampleMs / nSample;

  @override
  String toString() =>
      'SamplerPerf(nSample: $nSample, sampleMs: ${sampleMs.toStringAsFixed(2)}, '
      '${msPerSample.toStringAsFixed(3)} ms/sample)';
}
