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
      ..use_mmap = params.useMmap
      ..use_mlock = params.useMlock
      ..vocab_only = params.vocabOnly
      ..check_tensors = params.checkTensors;

    final pathPtr = params.path.toNativeUtf8(allocator: calloc);
    try {
      final ptr = b.llama_model_load_from_file(pathPtr.cast(), cParams);
      if (ptr == nullptr) {
        throw LlamaModelLoadException('failed to load model: ${params.path}');
      }
      return LlamaModel._(ptr, params);
    } finally {
      calloc.free(pathPtr);
    }
  }

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
