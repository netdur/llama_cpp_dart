/// Declarative configuration for [LlamaModel.load].
///
/// Immutable, JSON-friendly. Holds no native memory.
final class ModelParams {
  /// Absolute path to a GGUF file.
  final String path;

  /// Number of layers offloaded to a GPU/NPU backend (Metal, Hexagon, ...).
  /// `0` keeps everything on CPU.
  final int gpuLayers;

  /// `mmap` the GGUF file instead of reading it into RAM.
  final bool useMmap;

  /// Lock memory pages to prevent the OS from swapping.
  final bool useMlock;

  /// Load only the vocabulary (skip tensor data). Useful for tokenizer-only
  /// workflows.
  final bool vocabOnly;

  /// Verify tensor data on load. Slower; off by default.
  final bool checkTensors;

  const ModelParams({
    required this.path,
    this.gpuLayers = 0,
    this.useMmap = true,
    this.useMlock = false,
    this.vocabOnly = false,
    this.checkTensors = false,
  });

  ModelParams copyWith({
    String? path,
    int? gpuLayers,
    bool? useMmap,
    bool? useMlock,
    bool? vocabOnly,
    bool? checkTensors,
  }) {
    return ModelParams(
      path: path ?? this.path,
      gpuLayers: gpuLayers ?? this.gpuLayers,
      useMmap: useMmap ?? this.useMmap,
      useMlock: useMlock ?? this.useMlock,
      vocabOnly: vocabOnly ?? this.vocabOnly,
      checkTensors: checkTensors ?? this.checkTensors,
    );
  }

  Map<String, Object?> toJson() => {
        'path': path,
        'gpuLayers': gpuLayers,
        'useMmap': useMmap,
        'useMlock': useMlock,
        'vocabOnly': vocabOnly,
        'checkTensors': checkTensors,
      };

  factory ModelParams.fromJson(Map<String, Object?> json) => ModelParams(
        path: json['path']! as String,
        gpuLayers: (json['gpuLayers'] as int?) ?? 0,
        useMmap: (json['useMmap'] as bool?) ?? true,
        useMlock: (json['useMlock'] as bool?) ?? false,
        vocabOnly: (json['vocabOnly'] as bool?) ?? false,
        checkTensors: (json['checkTensors'] as bool?) ?? false,
      );
}
