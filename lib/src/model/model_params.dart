/// How a multi-GPU model is split across devices.
enum SplitMode {
  /// Single device (use [ModelParams.mainGpu]).
  none,

  /// Split layers across devices.
  layer,

  /// Split rows across devices (matrix-parallel).
  row,

  /// Split tensors across devices.
  tensor,
}

/// Type of a [KvOverride] entry.
enum KvOverrideType {
  intValue,
  floatValue,
  boolValue,
  string,
}

/// Override for a single GGUF metadata key. Used to patch model metadata at
/// load time without re-quantizing the file.
final class KvOverride {
  /// Metadata key (max 127 chars; longer values are truncated).
  final String key;
  final KvOverrideType type;
  final int? intValue;
  final double? floatValue;
  final bool? boolValue;
  final String? stringValue;

  const KvOverride._({
    required this.key,
    required this.type,
    this.intValue,
    this.floatValue,
    this.boolValue,
    this.stringValue,
  });

  factory KvOverride.int(String key, int value) => KvOverride._(
        key: key,
        type: KvOverrideType.intValue,
        intValue: value,
      );

  factory KvOverride.float(String key, double value) => KvOverride._(
        key: key,
        type: KvOverrideType.floatValue,
        floatValue: value,
      );

  factory KvOverride.bool(String key, bool value) => KvOverride._(
        key: key,
        type: KvOverrideType.boolValue,
        boolValue: value,
      );

  factory KvOverride.string(String key, String value) => KvOverride._(
        key: key,
        type: KvOverrideType.string,
        stringValue: value,
      );

  Map<String, Object?> toJson() => {
        'key': key,
        'type': type.name,
        if (intValue != null) 'int': intValue,
        if (floatValue != null) 'float': floatValue,
        if (boolValue != null) 'bool': boolValue,
        if (stringValue != null) 'string': stringValue,
      };

  factory KvOverride.fromJson(Map<String, Object?> json) {
    final key = json['key']! as String;
    final type = KvOverrideType.values.firstWhere(
      (e) => e.name == json['type'],
      orElse: () => KvOverrideType.intValue,
    );
    return switch (type) {
      KvOverrideType.intValue => KvOverride.int(key, json['int']! as int),
      KvOverrideType.floatValue =>
        KvOverride.float(key, (json['float']! as num).toDouble()),
      KvOverrideType.boolValue => KvOverride.bool(key, json['bool']! as bool),
      KvOverrideType.string => KvOverride.string(key, json['string']! as String),
    };
  }
}

/// Declarative configuration for [LlamaModel.load].
///
/// Immutable, JSON-friendly. Holds no native memory.
final class ModelParams {
  /// Absolute path to a GGUF file.
  final String path;

  /// Number of layers offloaded to a GPU/NPU backend (Metal, Hexagon, ...).
  /// `0` keeps everything on CPU. Negative = all layers.
  final int gpuLayers;

  /// Strategy for splitting the model across multiple GPUs.
  final SplitMode splitMode;

  /// Index of the GPU used for the entire model when [splitMode] is
  /// [SplitMode.none]. Indexing matches the order returned by
  /// `LlamaBackends.list()` filtered to GPU devices.
  final int mainGpu;

  /// Per-device split proportions (layers or rows). When non-empty the list
  /// is padded / truncated to `llama_max_devices()` at load time. Sum should
  /// be `1.0` but llama.cpp normalizes regardless.
  final List<double> tensorSplit;

  /// Names of devices (matching `BackendDevice.name`) to make eligible for
  /// offload. Empty = use all available devices (the default).
  final List<String> devices;

  /// GGUF metadata key/value overrides applied at load time.
  final List<KvOverride> kvOverrides;

  /// `mmap` the GGUF file instead of reading it into RAM.
  final bool useMmap;

  /// Use direct I/O when supported. Takes precedence over [useMmap].
  final bool useDirectIo;

  /// Lock memory pages to prevent the OS from swapping.
  final bool useMlock;

  /// Load only the vocabulary (skip tensor data). Useful for tokenizer-only
  /// workflows.
  final bool vocabOnly;

  /// Verify tensor data on load. Slower; off by default.
  final bool checkTensors;

  /// Allow extra buffer types for weight repacking.
  final bool useExtraBufts;

  /// Bypass the host buffer so that extra buffer types can be used.
  final bool noHost;

  /// Load metadata only and simulate memory allocations (useful for sizing /
  /// planning without actually populating tensors).
  final bool noAlloc;

  const ModelParams({
    required this.path,
    this.gpuLayers = 0,
    this.splitMode = SplitMode.layer,
    this.mainGpu = 0,
    this.tensorSplit = const [],
    this.devices = const [],
    this.kvOverrides = const [],
    this.useMmap = true,
    this.useDirectIo = false,
    this.useMlock = false,
    this.vocabOnly = false,
    this.checkTensors = false,
    this.useExtraBufts = true,
    this.noHost = false,
    this.noAlloc = false,
  });

  ModelParams copyWith({
    String? path,
    int? gpuLayers,
    SplitMode? splitMode,
    int? mainGpu,
    List<double>? tensorSplit,
    List<String>? devices,
    List<KvOverride>? kvOverrides,
    bool? useMmap,
    bool? useDirectIo,
    bool? useMlock,
    bool? vocabOnly,
    bool? checkTensors,
    bool? useExtraBufts,
    bool? noHost,
    bool? noAlloc,
  }) {
    return ModelParams(
      path: path ?? this.path,
      gpuLayers: gpuLayers ?? this.gpuLayers,
      splitMode: splitMode ?? this.splitMode,
      mainGpu: mainGpu ?? this.mainGpu,
      tensorSplit: tensorSplit ?? this.tensorSplit,
      devices: devices ?? this.devices,
      kvOverrides: kvOverrides ?? this.kvOverrides,
      useMmap: useMmap ?? this.useMmap,
      useDirectIo: useDirectIo ?? this.useDirectIo,
      useMlock: useMlock ?? this.useMlock,
      vocabOnly: vocabOnly ?? this.vocabOnly,
      checkTensors: checkTensors ?? this.checkTensors,
      useExtraBufts: useExtraBufts ?? this.useExtraBufts,
      noHost: noHost ?? this.noHost,
      noAlloc: noAlloc ?? this.noAlloc,
    );
  }

  Map<String, Object?> toJson() => {
        'path': path,
        'gpuLayers': gpuLayers,
        'splitMode': splitMode.name,
        'mainGpu': mainGpu,
        'tensorSplit': tensorSplit,
        'devices': devices,
        'kvOverrides': kvOverrides.map((e) => e.toJson()).toList(),
        'useMmap': useMmap,
        'useDirectIo': useDirectIo,
        'useMlock': useMlock,
        'vocabOnly': vocabOnly,
        'checkTensors': checkTensors,
        'useExtraBufts': useExtraBufts,
        'noHost': noHost,
        'noAlloc': noAlloc,
      };

  factory ModelParams.fromJson(Map<String, Object?> json) => ModelParams(
        path: json['path']! as String,
        gpuLayers: (json['gpuLayers'] as int?) ?? 0,
        splitMode: SplitMode.values.firstWhere(
          (e) => e.name == json['splitMode'],
          orElse: () => SplitMode.layer,
        ),
        mainGpu: (json['mainGpu'] as int?) ?? 0,
        tensorSplit: ((json['tensorSplit'] as List?) ?? const [])
            .map((e) => (e as num).toDouble())
            .toList(growable: false),
        devices:
            ((json['devices'] as List?)?.cast<String>()) ?? const <String>[],
        kvOverrides: ((json['kvOverrides'] as List?) ?? const [])
            .cast<Map<String, Object?>>()
            .map(KvOverride.fromJson)
            .toList(growable: false),
        useMmap: (json['useMmap'] as bool?) ?? true,
        useDirectIo: (json['useDirectIo'] as bool?) ?? false,
        useMlock: (json['useMlock'] as bool?) ?? false,
        vocabOnly: (json['vocabOnly'] as bool?) ?? false,
        checkTensors: (json['checkTensors'] as bool?) ?? false,
        useExtraBufts: (json['useExtraBufts'] as bool?) ?? true,
        noHost: (json['noHost'] as bool?) ?? false,
        noAlloc: (json['noAlloc'] as bool?) ?? false,
      );
}
