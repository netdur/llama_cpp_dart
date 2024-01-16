/// Enumerates the modes for splitting a Llama model across multiple GPUs.
///
/// This enum is used to define how the model's computational load
/// is distributed across available GPU resources.
enum LlamaSplitMode {
  /// Indicates that the model should use a single GPU.
  none,

  /// Splits layers and Key-Value (KV) pairs across multiple GPUs.
  layer,

  /// Splits rows of the model across GPUs.
  row,
}

/// Extension on `LlamaSplitMode` to provide additional functionality.
extension LlamaSplitModeExtension on LlamaSplitMode {
  /// Returns an integer representation of the `LlamaSplitMode`.
  ///
  /// This is useful for scenarios where an enum needs to be
  /// converted to a primitive type, for example, when interacting
  /// with lower-level APIs or external libraries.
  int get value {
    switch (this) {
      case LlamaSplitMode.none:
        return 0;
      case LlamaSplitMode.layer:
        return 1;
      case LlamaSplitMode.row:
        return 2;
      default:
        return 0;
    }
  }
}
