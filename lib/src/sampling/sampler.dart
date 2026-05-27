import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../context/context.dart';
import '../diagnostics/perf.dart';
import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import '../types/exceptions.dart';
import 'sampler_params.dart';

/// Owned wrapper around a `llama_sampler` chain.
///
/// One sampler per generation request. Disposable.
final class Sampler implements Finalizable {
  final SamplerParams params;
  final Pointer<llama_sampler> _ptr;
  final bool _borrowed;
  bool _disposed = false;

  Sampler(this._ptr, this.params) : _borrowed = false;

  /// Wrap a sampler pointer we don't own (e.g. a chain stage). [dispose]
  /// on this handle is a no-op so we don't double-free the parent's slot.
  Sampler._borrowed(this._ptr)
      : params = const SamplerParams(),
        _borrowed = true;

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

  /// Snapshot the chain's performance counters (cumulative sample time and
  /// number of sample calls). Counters reset only via [resetPerf].
  SamplerPerf perf() {
    final d = LlamaLibrary.bindings.llama_perf_sampler(pointer);
    return SamplerPerf(sampleMs: d.t_sample_ms, nSample: d.n_sample);
  }

  /// Reset the sampler's performance counters.
  void resetPerf() {
    LlamaLibrary.bindings.llama_perf_sampler_reset(pointer);
  }

  /// Print sampler perf counters via llama.cpp's logger.
  void printPerf() {
    LlamaLibrary.bindings.llama_perf_sampler_print(pointer);
  }

  /// Display name reported by the sampler (e.g. `chain`, `top-k`,
  /// `temperature`). Empty string when llama.cpp returns null.
  String get name {
    final ptr = LlamaLibrary.bindings.llama_sampler_name(pointer);
    if (ptr == nullptr) return '';
    return ptr.cast<Utf8>().toDartString();
  }

  /// Seed currently in use by the random stage of this sampler (top of the
  /// chain). Returns `0` for samplers that don't have a seed concept.
  int get seed => LlamaLibrary.bindings.llama_sampler_get_seed(pointer);

  /// Number of stages in the chain. Returns `0` for samplers that aren't
  /// chains (a single greedy / dist / penalty sampler).
  int get chainCount => LlamaLibrary.bindings.llama_sampler_chain_n(pointer);

  /// Borrow the [index]-th stage of the chain. The returned sampler is
  /// owned by this chain — do NOT call `dispose` on it. Useful for reading
  /// `.name` or perf counters on a specific stage.
  Sampler chainGet(int index) {
    final raw =
        LlamaLibrary.bindings.llama_sampler_chain_get(pointer, index);
    if (raw == nullptr) {
      throw RangeError.range(index, 0, chainCount - 1, 'index');
    }
    return Sampler._borrowed(raw);
  }

  /// Detach the [index]-th stage from the chain and return it as an owned
  /// [Sampler]. The caller must dispose the returned handle.
  Sampler chainRemove(int index) {
    final raw =
        LlamaLibrary.bindings.llama_sampler_chain_remove(pointer, index);
    if (raw == nullptr) {
      throw RangeError.range(index, 0, chainCount - 1, 'index');
    }
    return Sampler(raw, params);
  }

  /// Apply this sampler to a [llama_token_data_array] in place. Advanced;
  /// most callers should use [sample] instead. The array must be allocated
  /// and populated by the caller (e.g. from `lastLogits`).
  void apply(Pointer<llama_token_data_array> array) {
    LlamaLibrary.bindings.llama_sampler_apply(pointer, array);
  }

  /// Deep-copy this sampler (including any stateful sub-stages). The clone
  /// is independent — disposing it does not affect the original.
  Sampler clone() {
    final raw = LlamaLibrary.bindings.llama_sampler_clone(pointer);
    if (raw == nullptr) {
      throw const LlamaLibraryException('llama_sampler_clone returned null');
    }
    return Sampler(raw, params);
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    if (_borrowed) return;
    LlamaLibrary.bindings.llama_sampler_free(_ptr);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('Sampler has been disposed.');
    }
  }
}
