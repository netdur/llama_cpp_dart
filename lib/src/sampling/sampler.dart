import 'dart:ffi';

import '../context/context.dart';
import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import 'sampler_params.dart';

/// Owned wrapper around a `llama_sampler` chain.
///
/// One sampler per generation request. Disposable.
final class Sampler implements Finalizable {
  final SamplerParams params;
  final Pointer<llama_sampler> _ptr;
  bool _disposed = false;

  Sampler(this._ptr, this.params);

  Pointer<llama_sampler> get pointer {
    _ensureAlive();
    return _ptr;
  }

  /// Sample a token from the logits at index [idx] of the last decode.
  /// `idx = -1` (default) takes the logits at the last submitted position.
  int sample(LlamaContext context, {int idx = -1}) {
    return LlamaLibrary.bindings
        .llama_sampler_sample(pointer, context.pointer, idx);
  }

  /// Inform the chain about the chosen [token]. Required for stateful samplers
  /// (penalties, mirostat, grammar) to update their internal state.
  void accept(int token) {
    LlamaLibrary.bindings.llama_sampler_accept(pointer, token);
  }

  /// Reset stateful samplers (penalties, mirostat) to their initial state.
  void reset() {
    LlamaLibrary.bindings.llama_sampler_reset(pointer);
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    LlamaLibrary.bindings.llama_sampler_free(_ptr);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('Sampler has been disposed.');
    }
  }
}
