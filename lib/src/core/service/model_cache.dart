import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../llama.dart';
import '../llama_cpp.dart';
import '../model_params.dart';

class SharedModelHandle {
  final Pointer<llama_model> model;
  final Pointer<llama_vocab> vocab;
  final ModelCacheKey? key;
  int refs = 1;

  SharedModelHandle({required this.model, required this.vocab, this.key});
}

class ModelCacheKey {
  final String path;
  final String paramsSignature;

  ModelCacheKey(this.path, ModelParams params)
      : paramsSignature = params.toString();

  @override
  bool operator ==(Object other) =>
      other is ModelCacheKey &&
      other.path == path &&
      other.paramsSignature == paramsSignature;

  @override
  int get hashCode => Object.hash(path, paramsSignature);
}

class ModelCache {
  static final Map<ModelCacheKey, SharedModelHandle> _cache = {};

  static bool get isEmpty => _cache.isEmpty;

  static SharedModelHandle acquire({
    required llama_cpp lib,
    required String path,
    required ModelParams params,
  }) {
    final key = ModelCacheKey(path, params);
    final cached = _cache[key];
    if (cached != null) {
      cached.refs++;
      return cached;
    }

    final mp = params.get();
    final modelPathPtr = path.toNativeUtf8().cast<Char>();

    Pointer<llama_model> loadedModel = nullptr;
    Pointer<llama_vocab> vocab = nullptr;

    try {
      loadedModel = lib.llama_load_model_from_file(modelPathPtr, mp);
      if (loadedModel == nullptr) {
        throw LlamaException("Could not load model at $path");
      }
      vocab = lib.llama_model_get_vocab(loadedModel);
    } finally {
      malloc.free(modelPathPtr);
    }

    final handle =
        SharedModelHandle(model: loadedModel, vocab: vocab, key: key);
    _cache[key] = handle;
    return handle;
  }

  static void release({
    required llama_cpp lib,
    required SharedModelHandle handle,
  }) {
    handle.refs--;
    if (handle.refs > 0) return;

    if (handle.key != null) {
      _cache.remove(handle.key);
    } else {
      _cache.removeWhere((_, h) => identical(h, handle));
    }

    if (handle.model.address != 0) {
      lib.llama_free_model(handle.model);
    }
  }
}
