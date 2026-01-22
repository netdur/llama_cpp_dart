import 'dart:ffi';

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
