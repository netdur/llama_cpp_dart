import 'dart:convert';
// import 'dart:ffi';
// import 'package:ffi/ffi.dart';

import 'llama.dart';
import 'llama_cpp.dart';
import 'prompt_format.dart';
// import 'llama_split_mode.dart';

/// ModelParams configures how the model is split and operated across multiple GPUs.
///
/// It includes settings for tensor splitting, metadata overrides, GPU layer storage,
/// and memory management options.
class ModelParams {
  // how to split the model across multiple GPUs
  // LlamaSplitMode splitsMode = LlamaSplitMode.layer;

  /// Proportion of the model (layers or rows) to offload to each GPU.
  /// Size is defined by LLAMA_MAX_DEVICES.
  // List<int> tensorSplit = [];

  /// Override key-value pairs of the model metadata.
  // Map<String, dynamic> metadataOverride = {};

  /// Prompt format. Defaults to `PromptFormat.raw`.
  PromptFormatType format = PromptFormatType.raw;

  /// Number of layers to store in VRAM. Default is 99.
  int gpuLayerLayer = 99;

  /// Interpretation of main_gpu depends on split_mode:
  /// LLAMA_SPLIT_NONE: the GPU used for the entire model.
  /// LLAMA_SPLIT_ROW: the GPU used for small tensors and intermediate results.
  /// LLAMA_SPLIT_LAYER: ignored.
  int mainGpu = 0;

  /// If true, only load the vocabulary without weights.
  bool vocabOnly = false;

  /// If true, use memory mapping if possible.
  bool useMemorymap = true;

  /// If true, force the system to keep the model in RAM.
  bool useMemoryLock = false;

  // late Pointer<Float> _tensorSplitPointer;

  ModelParams() {
    // _tensorSplitPointer = calloc<Float>(tensorSplit.length);
    // _finalizer.attach(this, _tensorSplitPointer, detach: this);
  }

  /*
  static final _finalizer = Finalizer<Pointer<Float>>((pointer) {
    if (pointer.address != 0) {
      calloc.free(pointer);
    }
  });
  */

  /// Releases allocated resources.
  void dispose() {
    // _finalizer.detach(this);
    // calloc.free(_tensorSplitPointer);
  }

  /// Constructs and returns a `llama_model_params` object with current settings.
  llama_model_params get() {
    llama_model_params modelParams = Llama.lib.llama_model_default_params();
    /*
    _tensorSplitPointer = calloc<Float>(tensorSplit.length);
    for (int i = 0; i < tensorSplit.length; i++) {
      _tensorSplitPointer[i] = tensorSplit[i].toDouble();
    }
    modelParams.tensor_split = _tensorSplitPointer;
    */

    /*
    final kvOverridesPointer =
        calloc<llama_model_kv_override>(metadataOverride.length);

    int i = 0;
    metadataOverride.forEach((key, value) {
      llama_model_kv_override override = kvOverridesPointer[i];

      Pointer<Char> overrideKey = key.toNativeUtf8().cast<Char>();
      override.key[overrideKey.value] = 0;

      if (value is int) {
        override.tag = 0;
        override.unnamed.int_value = value;
      } else if (value is double) {
        override.tag = 1;
        override.unnamed.float_value = value;
      } else if (value is bool) {
        override.tag = 2;
        override.unnamed.bool_value = value;
      }

      malloc.free(overrideKey);
      i++;
    });
    modelParams.kv_overrides = kvOverridesPointer;
    */

    modelParams.main_gpu = mainGpu;
    modelParams.n_gpu_layers = gpuLayerLayer;
    modelParams.vocab_only = vocabOnly;
    modelParams.use_mmap = useMemorymap;
    modelParams.use_mlock = useMemoryLock;
    return modelParams;
  }

  /// Constructs a `ModelParams` instance from a JSON map.
  ///
  /// The JSON map should contain key-value pairs corresponding to the
  /// properties of this class.
  ModelParams.fromJson(Map<String, dynamic> json) {
    // splitsMode = LlamaSplitMode.values[json['splitsMode'] ?? 0];
    // tensorSplit = List<int>.from(json['tensorSplit'] ?? []);
    // metadataOverride = Map<String, dynamic>.from(json['metadataOverride'] ?? {});
    gpuLayerLayer = json['gpuLayerLayer'] ?? 0;
    mainGpu = json['mainGpu'] ?? 0;
    vocabOnly = json['vocabOnly'] ?? false;
    useMemorymap = json['useMemorymap'] ?? true;
    useMemoryLock = json['useMemoryLock'] ?? false;
  }

  /// Converts the `ModelParams` instance to a JSON map.
  ///
  /// Useful for serialization and debugging.
  Map<String, dynamic> toJson() {
    return {
      // 'splitsMode': splitsMode.index,
      // 'tensorSplit': tensorSplit,
      // 'metadataOverride': metadataOverride,
      'gpuLayerLayer': gpuLayerLayer,
      'mainGpu': mainGpu,
      'vocabOnly': vocabOnly,
      'useMemorymap': useMemorymap,
      'useMemoryLock': useMemoryLock
    };
  }

  /// Returns a string representation of the `ModelParams` instance.
  ///
  /// Useful for debugging and logging.
  @override
  String toString() {
    return jsonEncode(toJson());
  }
}
