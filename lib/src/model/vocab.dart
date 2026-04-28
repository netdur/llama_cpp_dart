import 'dart:ffi';

import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';

/// Read-only view over a model's vocabulary.
///
/// The vocabulary pointer is owned by the parent [LlamaModel]; this object
/// becomes invalid when the model is disposed.
final class LlamaVocab {
  final Pointer<llama_vocab> _ptr;
  const LlamaVocab._(this._ptr);

  factory LlamaVocab.fromModel(Pointer<llama_model> model) {
    final ptr = LlamaLibrary.bindings.llama_model_get_vocab(model);
    if (ptr == nullptr) {
      throw StateError('llama_model_get_vocab returned null');
    }
    return LlamaVocab._(ptr);
  }

  Pointer<llama_vocab> get pointer => _ptr;

  int get nTokens => LlamaLibrary.bindings.llama_vocab_n_tokens(_ptr);

  int get bos => LlamaLibrary.bindings.llama_vocab_bos(_ptr);
  int get eos => LlamaLibrary.bindings.llama_vocab_eos(_ptr);
  int get eot => LlamaLibrary.bindings.llama_vocab_eot(_ptr);
  int get sep => LlamaLibrary.bindings.llama_vocab_sep(_ptr);
  int get nl => LlamaLibrary.bindings.llama_vocab_nl(_ptr);
  int get pad => LlamaLibrary.bindings.llama_vocab_pad(_ptr);
  int get mask => LlamaLibrary.bindings.llama_vocab_mask(_ptr);

  /// True if [token] should end generation (EOS or alternate end-of-generation).
  bool isEog(int token) =>
      LlamaLibrary.bindings.llama_vocab_is_eog(_ptr, token);

  /// True if [token] is a control / special token.
  bool isControl(int token) =>
      LlamaLibrary.bindings.llama_vocab_is_control(_ptr, token);
}
