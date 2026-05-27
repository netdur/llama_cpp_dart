/// Bit flags for the `_ext` family of `llama_state_seq_*` functions.
///
/// These tune what a per-sequence KV snapshot contains and where it
/// lives. Combine with bitwise OR (`a.value | b.value`).
final class StateSeqFlags {
  /// The exact bit pattern to pass to llama.cpp.
  final int value;

  const StateSeqFlags._(this.value);

  /// No flags. Default `get_data` / `set_data` behavior.
  static const none = StateSeqFlags._(0);

  /// Only capture sliding-window-attention (SWA) cache state. Useful when
  /// you only need to round-trip the SWA portion across sessions. Kept for
  /// backwards-compat with older callers; the same bit means
  /// "partial only" upstream.
  static const swaOnly = StateSeqFlags._(1);

  /// Only capture the partial state (SWA KV cache or recurrent cache like
  /// Mamba). Skips the full attention KV.
  static const partialOnly = StateSeqFlags._(1);

  /// Keep the captured snapshot on device buffers (e.g. Metal / CUDA)
  /// rather than copying through host memory. Faster save/load but the
  /// resulting handle is only valid against the same context, and any
  /// subsequent on-device save invalidates earlier ones for the same seq.
  static const onDevice = StateSeqFlags._(2);

  /// Combine two flag values.
  StateSeqFlags operator |(StateSeqFlags other) =>
      StateSeqFlags._(value | other.value);

  /// True if every bit set in [other] is also set here.
  bool contains(StateSeqFlags other) => (value & other.value) == other.value;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is StateSeqFlags && other.value == value);

  @override
  int get hashCode => value.hashCode;

  @override
  String toString() => 'StateSeqFlags(0x${value.toRadixString(16)})';
}
