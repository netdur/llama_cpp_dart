import '../sampling/sampler_params.dart';
import 'context_shift.dart';

/// A single generation request. Immutable.
final class Request {
  /// Tokens to prefill before sampling begins.
  final List<int> promptTokens;

  /// Sampling configuration.
  final SamplerParams sampler;

  /// Maximum number of tokens to emit.
  final int maxTokens;

  /// Sequence id within the parent context.
  final int seqId;

  /// First KV position to use for prefill. `0` for fresh sequences;
  /// continuations supply the post-decode position from the previous run.
  final int startPos;

  /// What the generator should do when the KV cache is one slot from full.
  final ContextShiftPolicy shiftPolicy;

  /// Tunables for [ContextShiftPolicy.auto].
  final ContextShift shift;

  const Request({
    required this.promptTokens,
    this.sampler = const SamplerParams(),
    this.maxTokens = 256,
    this.seqId = 0,
    this.startPos = 0,
    this.shiftPolicy = ContextShiftPolicy.off,
    this.shift = ContextShift.defaults,
  });
}
