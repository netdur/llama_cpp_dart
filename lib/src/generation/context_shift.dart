/// Policy for what `Generator` does when the next decode would push past
/// the context window.
enum ContextShiftPolicy {
  /// Throw [LlamaDecodeException] when the context fills. The default —
  /// callers must handle overflow themselves.
  off,

  /// Automatically discard the oldest non-keep tokens and shift the KV
  /// cache left when the next sample would overflow. Mirrors
  /// `llama-server`'s `--context-shift` behaviour.
  ///
  /// Requires `LlamaContext.canShift` to be true and no multimodal
  /// projector to be active. Throws on first overflow if either is false.
  auto,
}

/// Shift configuration. Mirrors the `n_keep` / `n_discard` knobs from
/// llama-server. Used by [ContextShiftPolicy.auto].
final class ContextShift {
  /// Number of leading tokens to preserve. `-1` keeps the whole prompt
  /// (history captured before generation started). `0` keeps nothing
  /// beyond a possible BOS. The runtime always reserves at least 4 ctx
  /// slots for forward progress, capping `nKeep <= nCtx - 4`.
  final int nKeep;

  /// Tokens to discard per shift. `null` discards half of `n_left`
  /// (= `nTokens - nKeep`), matching server defaults.
  final int? nDiscard;

  /// Add 1 to `nKeep` to preserve the BOS token. Set to `false` for
  /// models that don't add a BOS at tokenize time.
  final bool keepBos;

  const ContextShift({
    this.nKeep = -1,
    this.nDiscard,
    this.keepBos = true,
  });

  static const ContextShift defaults = ContextShift();
}
