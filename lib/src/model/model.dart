import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import '../types/exceptions.dart';
import 'model_params.dart';
import 'vocab.dart';

/// A loaded GGUF model. Owns the underlying `llama_model*`.
///
/// One model can back multiple [LlamaContext]s. Dispose contexts before
/// disposing the model.
final class LlamaModel implements Finalizable {
  final ModelParams params;
  final Pointer<llama_model> _ptr;
  late final LlamaVocab vocab = LlamaVocab.fromModel(_ptr);

  bool _disposed = false;

  LlamaModel._(this._ptr, this.params);

  /// Load a model from disk. Synchronous and blocks the calling isolate;
  /// callers that need async behavior should drive this from a worker
  /// isolate (`LlamaEngine` in M3).
  static LlamaModel load(ModelParams params) {
    final b = LlamaLibrary.bindings;

    final cParams = b.llama_model_default_params()
      ..n_gpu_layers = params.gpuLayers
      ..split_modeAsInt = _splitModeInt(params.splitMode)
      ..main_gpu = params.mainGpu
      ..use_mmap = params.useMmap
      ..use_direct_io = params.useDirectIo
      ..use_mlock = params.useMlock
      ..vocab_only = params.vocabOnly
      ..check_tensors = params.checkTensors
      ..use_extra_bufts = params.useExtraBufts
      ..no_host = params.noHost
      ..no_alloc = params.noAlloc;

    final allocations = <_Allocation>[];

    if (params.tensorSplit.isNotEmpty) {
      final maxDevices = b.llama_max_devices();
      final ptr = calloc<Float>(maxDevices);
      for (var i = 0; i < params.tensorSplit.length && i < maxDevices; i++) {
        ptr[i] = params.tensorSplit[i];
      }
      cParams.tensor_split = ptr;
      allocations.add(_Allocation(ptr.cast()));
    }

    if (params.devices.isNotEmpty) {
      final ptr = _resolveDevices(params.devices);
      cParams.devices = ptr;
      allocations.add(_Allocation(ptr.cast()));
    }

    if (params.kvOverrides.isNotEmpty) {
      final ptr = _buildKvOverrides(params.kvOverrides);
      cParams.kv_overrides = ptr;
      allocations.add(_Allocation(ptr.cast()));
    }

    final pathPtr = params.path.toNativeUtf8(allocator: calloc);
    try {
      final ptr = b.llama_model_load_from_file(pathPtr.cast(), cParams);
      if (ptr == nullptr) {
        throw LlamaModelLoadException('failed to load model: ${params.path}');
      }
      return LlamaModel._(ptr, params);
    } finally {
      calloc.free(pathPtr);
      for (final a in allocations) {
        calloc.free(a.ptr);
      }
    }
  }

  static Pointer<ggml_backend_dev_t> _resolveDevices(List<String> names) {
    final b = LlamaLibrary.bindings;
    // NULL-terminated; allocate one extra slot.
    final ptr = calloc<ggml_backend_dev_t>(names.length + 1);
    final available = b.ggml_backend_dev_count();

    for (var i = 0; i < names.length; i++) {
      var found = false;
      for (var j = 0; j < available; j++) {
        final dev = b.ggml_backend_dev_get(j);
        if (dev == nullptr) continue;
        final namePtr = b.ggml_backend_dev_name(dev);
        if (namePtr == nullptr) continue;
        final n = namePtr.cast<Utf8>().toDartString();
        if (n == names[i]) {
          ptr[i] = dev;
          found = true;
          break;
        }
      }
      if (!found) {
        calloc.free(ptr);
        throw LlamaModelLoadException(
          'unknown backend device "${names[i]}" — available: '
          '${_listDeviceNames()}',
        );
      }
    }
    return ptr;
  }

  static String _listDeviceNames() {
    final b = LlamaLibrary.bindings;
    final names = <String>[];
    for (var j = 0; j < b.ggml_backend_dev_count(); j++) {
      final dev = b.ggml_backend_dev_get(j);
      if (dev == nullptr) continue;
      final namePtr = b.ggml_backend_dev_name(dev);
      if (namePtr == nullptr) continue;
      names.add(namePtr.cast<Utf8>().toDartString());
    }
    return names.join(', ');
  }

  static Pointer<llama_model_kv_override> _buildKvOverrides(
    List<KvOverride> overrides,
  ) {
    // llama.cpp iterates until key[0] == 0 — append one zero-keyed entry as
    // the terminator.
    final count = overrides.length + 1;
    final arr = calloc<llama_model_kv_override>(count);
    for (var i = 0; i < overrides.length; i++) {
      final o = overrides[i];
      _writeFixedString(arr[i].key, o.key, 128);
      switch (o.type) {
        case KvOverrideType.intValue:
          arr[i].tagAsInt =
              llama_model_kv_override_type.LLAMA_KV_OVERRIDE_TYPE_INT.value;
          arr[i].unnamed.val_i64 = o.intValue!;
          break;
        case KvOverrideType.floatValue:
          arr[i].tagAsInt =
              llama_model_kv_override_type.LLAMA_KV_OVERRIDE_TYPE_FLOAT.value;
          arr[i].unnamed.val_f64 = o.floatValue!;
          break;
        case KvOverrideType.boolValue:
          arr[i].tagAsInt =
              llama_model_kv_override_type.LLAMA_KV_OVERRIDE_TYPE_BOOL.value;
          arr[i].unnamed.val_bool = o.boolValue!;
          break;
        case KvOverrideType.string:
          arr[i].tagAsInt =
              llama_model_kv_override_type.LLAMA_KV_OVERRIDE_TYPE_STR.value;
          _writeFixedString(arr[i].unnamed.val_str, o.stringValue!, 128);
          break;
      }
    }
    // Terminator (key[0] = 0 already from calloc's zero fill).
    return arr;
  }

  static void _writeFixedString(Array<Char> dst, String src, int max) {
    final bytes = src.codeUnits;
    final n = bytes.length < max - 1 ? bytes.length : max - 1;
    for (var i = 0; i < n; i++) {
      dst[i] = bytes[i] & 0xff;
    }
    dst[n] = 0;
  }

  static int _splitModeInt(SplitMode v) => switch (v) {
        SplitMode.none => llama_split_mode.LLAMA_SPLIT_MODE_NONE.value,
        SplitMode.layer => llama_split_mode.LLAMA_SPLIT_MODE_LAYER.value,
        SplitMode.row => llama_split_mode.LLAMA_SPLIT_MODE_ROW.value,
        SplitMode.tensor => llama_split_mode.LLAMA_SPLIT_MODE_TENSOR.value,
      };

  Pointer<llama_model> get pointer {
    _ensureAlive();
    return _ptr;
  }

  int get nParams => LlamaLibrary.bindings.llama_model_n_params(pointer);
  int get nEmbd => LlamaLibrary.bindings.llama_model_n_embd(pointer);
  int get nLayer => LlamaLibrary.bindings.llama_model_n_layer(pointer);
  int get nHead => LlamaLibrary.bindings.llama_model_n_head(pointer);
  int get nHeadKv => LlamaLibrary.bindings.llama_model_n_head_kv(pointer);

  /// Native context length the model was trained with.
  int get trainCtx => LlamaLibrary.bindings.llama_model_n_ctx_train(pointer);

  /// Total size of the model on disk, in bytes.
  int get sizeBytes => LlamaLibrary.bindings.llama_model_size(pointer);

  bool get hasEncoder => LlamaLibrary.bindings.llama_model_has_encoder(pointer);
  bool get hasDecoder => LlamaLibrary.bindings.llama_model_has_decoder(pointer);
  bool get isRecurrent =>
      LlamaLibrary.bindings.llama_model_is_recurrent(pointer);
  bool get isDiffusion =>
      LlamaLibrary.bindings.llama_model_is_diffusion(pointer);
  bool get isHybrid => LlamaLibrary.bindings.llama_model_is_hybrid(pointer);

  /// Token the model expects the decoder to start with in encoder-decoder
  /// setups (e.g. T5). Returns `-1` for decoder-only models.
  int get decoderStartToken =>
      LlamaLibrary.bindings.llama_model_decoder_start_token(pointer);

  /// Embedding dimension for *input* tokens — may differ from [nEmbd] for
  /// models with a tied input/output but different projection sizes.
  int get nEmbdInp =>
      LlamaLibrary.bindings.llama_model_n_embd_inp(pointer);

  /// Embedding dimension for *output* tokens.
  int get nEmbdOut =>
      LlamaLibrary.bindings.llama_model_n_embd_out(pointer);

  /// Sliding-window-attention span the model was trained with. `0` for
  /// models that use full attention.
  int get nSwa => LlamaLibrary.bindings.llama_model_n_swa(pointer);

  /// Number of classification outputs on the head. `0` for non-classifier
  /// models. Pairs with [classifierLabel] to read each label.
  int get nClassifierOut =>
      LlamaLibrary.bindings.llama_model_n_cls_out(pointer);

  /// Human-readable label for classifier output [index]. Returns `null`
  /// when the model has no label table for that index.
  String? classifierLabel(int index) {
    final ptr =
        LlamaLibrary.bindings.llama_model_cls_label(pointer, index);
    if (ptr == nullptr) return null;
    return ptr.cast<Utf8>().toDartString();
  }

  /// RoPE frequency-scaling factor the model was trained with. `1.0` when
  /// the model does not declare one.
  double get ropeFreqScaleTrain =>
      LlamaLibrary.bindings.llama_model_rope_freq_scale_train(pointer);

  /// RoPE variant the model uses.
  RopeType get ropeType {
    final raw = LlamaLibrary.bindings.llama_model_rope_type(pointer);
    return switch (raw) {
      llama_rope_type.LLAMA_ROPE_TYPE_NONE => RopeType.none,
      llama_rope_type.LLAMA_ROPE_TYPE_NORM => RopeType.norm,
      llama_rope_type.LLAMA_ROPE_TYPE_NEOX => RopeType.neox,
      llama_rope_type.LLAMA_ROPE_TYPE_MROPE => RopeType.mrope,
      llama_rope_type.LLAMA_ROPE_TYPE_IMROPE => RopeType.imrope,
      llama_rope_type.LLAMA_ROPE_TYPE_VISION => RopeType.vision,
    };
  }

  /// Number of GGUF metadata key/value pairs embedded in the model.
  int get metaCount => LlamaLibrary.bindings.llama_model_meta_count(pointer);

  /// Read the metadata key at [index] (`0 <= index < metaCount`). Returns
  /// `null` if the index is out of range.
  String? metaKeyAt(int index) =>
      _readModelString(maxLen: 256, (buf, size) {
        return LlamaLibrary.bindings.llama_model_meta_key_by_index(
          pointer,
          index,
          buf,
          size,
        );
      });

  /// Read the metadata value at [index] (`0 <= index < metaCount`).
  /// Returns `null` if the index is out of range.
  String? metaValueAt(int index) =>
      _readModelString(maxLen: 1024, (buf, size) {
        return LlamaLibrary.bindings.llama_model_meta_val_str_by_index(
          pointer,
          index,
          buf,
          size,
        );
      });

  /// Read the metadata value for [key]. Returns `null` when the key is
  /// missing. Keys follow GGUF convention, e.g. `general.architecture`.
  String? metaValue(String key) {
    final keyPtr = key.toNativeUtf8();
    try {
      return _readModelString(maxLen: 1024, (buf, size) {
        return LlamaLibrary.bindings.llama_model_meta_val_str(
          pointer,
          keyPtr.cast<Char>(),
          buf,
          size,
        );
      });
    } finally {
      calloc.free(keyPtr);
    }
  }

  /// All metadata entries as a `key -> value` map. Convenience for
  /// debugging and UI inspection — for large models this materializes
  /// every entry. Stream via [metaKeyAt] / [metaValueAt] if that matters.
  Map<String, String> metaEntries() {
    final n = metaCount;
    final out = <String, String>{};
    for (var i = 0; i < n; i++) {
      final k = metaKeyAt(i);
      if (k == null) continue;
      out[k] = metaValueAt(i) ?? '';
    }
    return out;
  }

  /// Short, model-provided description (architecture + size hint).
  String describe({int maxLen = 256}) {
    final buf = calloc<Char>(maxLen);
    try {
      final n = LlamaLibrary.bindings.llama_model_desc(pointer, buf, maxLen);
      if (n <= 0) return '';
      return buf.cast<Utf8>().toDartString(length: n);
    } finally {
      calloc.free(buf);
    }
  }

  String? _readModelString(
    int Function(Pointer<Char> buf, int size) call, {
    required int maxLen,
  }) {
    final buf = calloc<Char>(maxLen);
    try {
      final n = call(buf, maxLen);
      if (n < 0) return null;
      if (n == 0) return '';
      return buf.cast<Utf8>().toDartString(length: n);
    } finally {
      calloc.free(buf);
    }
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    LlamaLibrary.bindings.llama_model_free(_ptr);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('LlamaModel has been disposed.');
    }
  }
}

class _Allocation {
  final Pointer<NativeType> ptr;
  _Allocation(this.ptr);
}
