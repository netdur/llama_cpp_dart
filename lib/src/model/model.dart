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
