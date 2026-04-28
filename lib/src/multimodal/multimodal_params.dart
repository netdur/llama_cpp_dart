/// Declarative configuration for [MultimodalContext].
///
/// Immutable, JSON-friendly. Holds no native memory.
final class MultimodalParams {
  /// Absolute path to the multimodal projector file (e.g. `mmproj-F16.gguf`).
  final String mmprojPath;

  /// Offload the projector to the GPU/NPU backend when supported.
  final bool useGpu;

  /// Print backend timings to stderr after each encode pass.
  final bool printTimings;

  /// Threads used by the projector encoder. `0` lets the runtime pick.
  final int nThreads;

  /// Marker substituted in the prompt for each media item, in order.
  /// Defaults to llama.cpp's `<__media__>`.
  final String mediaMarker;

  /// Run a warmup encode pass after init. Slower startup, more reliable
  /// first-token latency.
  final bool warmup;

  /// Lower bound on the number of image tokens emitted (vision models with
  /// dynamic resolution). `0` means use the model's default.
  final int imageMinTokens;

  /// Upper bound on the number of image tokens emitted. `0` means use the
  /// model's default.
  final int imageMaxTokens;

  const MultimodalParams({
    required this.mmprojPath,
    this.useGpu = true,
    this.printTimings = false,
    this.nThreads = 0,
    this.mediaMarker = '<__media__>',
    this.warmup = false,
    this.imageMinTokens = 0,
    this.imageMaxTokens = 0,
  });

  MultimodalParams copyWith({
    String? mmprojPath,
    bool? useGpu,
    bool? printTimings,
    int? nThreads,
    String? mediaMarker,
    bool? warmup,
    int? imageMinTokens,
    int? imageMaxTokens,
  }) {
    return MultimodalParams(
      mmprojPath: mmprojPath ?? this.mmprojPath,
      useGpu: useGpu ?? this.useGpu,
      printTimings: printTimings ?? this.printTimings,
      nThreads: nThreads ?? this.nThreads,
      mediaMarker: mediaMarker ?? this.mediaMarker,
      warmup: warmup ?? this.warmup,
      imageMinTokens: imageMinTokens ?? this.imageMinTokens,
      imageMaxTokens: imageMaxTokens ?? this.imageMaxTokens,
    );
  }

  Map<String, Object?> toJson() => {
        'mmprojPath': mmprojPath,
        'useGpu': useGpu,
        'printTimings': printTimings,
        'nThreads': nThreads,
        'mediaMarker': mediaMarker,
        'warmup': warmup,
        'imageMinTokens': imageMinTokens,
        'imageMaxTokens': imageMaxTokens,
      };

  factory MultimodalParams.fromJson(Map<String, Object?> json) =>
      MultimodalParams(
        mmprojPath: json['mmprojPath']! as String,
        useGpu: (json['useGpu'] as bool?) ?? true,
        printTimings: (json['printTimings'] as bool?) ?? false,
        nThreads: (json['nThreads'] as int?) ?? 0,
        mediaMarker: (json['mediaMarker'] as String?) ?? '<__media__>',
        warmup: (json['warmup'] as bool?) ?? false,
        imageMinTokens: (json['imageMinTokens'] as int?) ?? 0,
        imageMaxTokens: (json['imageMaxTokens'] as int?) ?? 0,
      );
}
