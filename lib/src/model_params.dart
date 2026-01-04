import 'dart:convert';
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'llama_cpp.dart';

/// Enum representing how to split the model across multiple GPUs
enum LlamaSplitMode {
  none(0),
  layer(1),
  row(2);

  final int value;
  const LlamaSplitMode(this.value);
}

class ModelParams {
  PromptFormat? formatter;

  /// Number of layers to store in VRAM
  int nGpuLayers = 99;

  /// How to split the model across multiple GPUs
  LlamaSplitMode splitMode = LlamaSplitMode.none;

  /// The GPU that is used for the entire model when split_mode is none
  int mainGpu = 0;

  /// Proportion of the model (layers or rows) to offload to each GPU
  List<double> tensorSplit = [];

  /// Override key-value pairs of the model metadata
  /// (Note: Currently not fully wired to C struct in this wrapper)
  Map<String, dynamic> kvOverrides = {};

  /// Only load the vocabulary, no weights
  bool vocabOnly = false;

  /// Use mmap if possible
  bool useMemorymap = true;

  /// Force system to keep model in RAM
  bool useMemoryLock = false;

  /// Validate model tensor data
  bool checkTensors = false;

  /// Use extra buffer types (used for weight repacking)
  bool useExtraBufts = false;

  /// Bypass host buffer allowing extra buffers to be used
  bool noHost = false;

  Pointer<Float>? _tensorSplitPtr;

  ModelParams();

  /// Constructs and returns a `llama_model_params` object with current settings
  llama_model_params get() {
    final modelParams = Llama.lib.llama_model_default_params();

    modelParams.n_gpu_layers = nGpuLayers;
    modelParams.split_modeAsInt = splitMode.value;
    modelParams.main_gpu = mainGpu;
    modelParams.vocab_only = vocabOnly;
    modelParams.use_mmap = useMemorymap;
    modelParams.use_mlock = useMemoryLock;
    modelParams.check_tensors = checkTensors;

    modelParams.use_extra_bufts = useExtraBufts;
    modelParams.no_host = noHost;

    if (tensorSplit.isNotEmpty) {
      _tensorSplitPtr = malloc<Float>(tensorSplit.length);
      for (var i = 0; i < tensorSplit.length; i++) {
        _tensorSplitPtr![i] = tensorSplit[i];
      }
      modelParams.tensor_split = _tensorSplitPtr!;
    }

    modelParams.progress_callback = nullptr;
    modelParams.progress_callback_user_data = nullptr;
    modelParams.kv_overrides = nullptr;

    return modelParams;
  }

  /// Free allocated memory
  void dispose() {
    if (_tensorSplitPtr != null) {
      malloc.free(_tensorSplitPtr!);
      _tensorSplitPtr = null;
    }
  }

  /// Constructs a ModelParams instance from a JSON map
  factory ModelParams.fromJson(Map<String, dynamic> json) {
    final params = ModelParams();
    params.nGpuLayers = json['nGpuLayers'] ?? 99;

    params.splitMode = LlamaSplitMode.values.firstWhere(
        (e) => e.value == (json['splitMode'] ?? 0),
        orElse: () => LlamaSplitMode.none);

    params.mainGpu = json['mainGpu'] ?? 0;
    if (json['tensorSplit'] != null) {
      params.tensorSplit = (json['tensorSplit'] as List)
          .map((e) => (e as num).toDouble())
          .toList();
    }
    params.kvOverrides = Map<String, dynamic>.from(json['kvOverrides'] ?? {});
    params.vocabOnly = json['vocabOnly'] ?? false;
    params.useMemorymap = json['useMemorymap'] ?? true;
    params.useMemoryLock = json['useMemoryLock'] ?? false;
    params.checkTensors = json['checkTensors'] ?? false;
    params.useExtraBufts = json['useExtraBufts'] ?? false;
    params.noHost = json['noHost'] ?? false;
    return params;
  }

  /// Converts the ModelParams instance to a JSON map
  Map<String, dynamic> toJson() {
    return {
      'nGpuLayers': nGpuLayers,
      'splitMode': splitMode.value,
      'mainGpu': mainGpu,
      'tensorSplit': tensorSplit,
      'kvOverrides': kvOverrides,
      'vocabOnly': vocabOnly,
      'useMemorymap': useMemorymap,
      'useMemoryLock': useMemoryLock,
      'checkTensors': checkTensors,
      'useExtraBufts': useExtraBufts,
      'noHost': noHost,
    };
  }

  @override
  String toString() => jsonEncode(toJson());
}
