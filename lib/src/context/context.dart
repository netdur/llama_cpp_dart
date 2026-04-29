import 'dart:ffi';

import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import '../model/model.dart';
import '../types/exceptions.dart';
import 'context_params.dart';

/// Inference context bound to a [LlamaModel].
///
/// Owns the underlying `llama_context*`. Holds KV cache state.
final class LlamaContext implements Finalizable {
  final LlamaModel model;
  final ContextParams params;
  final Pointer<llama_context> _ptr;
  bool _disposed = false;

  LlamaContext._(this.model, this.params, this._ptr);

  static LlamaContext create(LlamaModel model, ContextParams params) {
    final b = LlamaLibrary.bindings;

    final cp = b.llama_context_default_params()
      ..n_ctx = params.nCtx
      ..n_batch = params.nBatch
      ..n_ubatch = params.nUbatch
      ..n_seq_max = params.nSeqMax
      ..n_threads = params.nThreads
      ..n_threads_batch = params.nThreadsBatch
      ..flash_attn_typeAsInt = _flashAttnInt(params.flashAttn)
      ..offload_kqv = params.offloadKqv
      ..embeddings = params.embeddings
      ..type_kAsInt = _kvCacheTypeInt(params.typeK)
      ..type_vAsInt = _kvCacheTypeInt(params.typeV);

    final ptr = b.llama_init_from_model(model.pointer, cp);
    if (ptr == nullptr) {
      throw const LlamaContextException(
        'llama_init_from_model returned null',
      );
    }
    return LlamaContext._(model, params, ptr);
  }

  Pointer<llama_context> get pointer {
    _ensureAlive();
    return _ptr;
  }

  int get nCtx => LlamaLibrary.bindings.llama_n_ctx(pointer);
  int get nBatch => LlamaLibrary.bindings.llama_n_batch(pointer);
  int get nUbatch => LlamaLibrary.bindings.llama_n_ubatch(pointer);
  int get nSeqMax => LlamaLibrary.bindings.llama_n_seq_max(pointer);
  int get nThreads => LlamaLibrary.bindings.llama_n_threads(pointer);
  int get nThreadsBatch => LlamaLibrary.bindings.llama_n_threads_batch(pointer);

  /// True if this context's memory backend supports `llama_memory_seq_add`
  /// (i.e., position shifting). False for recurrent and most iSWA caches —
  /// shifting them is mathematically unsound. Used to gate context-shift
  /// policies; mirrors `llama_memory_can_shift` exactly.
  bool get canShift {
    final lib = LlamaLibrary.bindings;
    return lib.llama_memory_can_shift(lib.llama_get_memory(pointer));
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    LlamaLibrary.bindings.llama_free(_ptr);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('LlamaContext has been disposed.');
    }
  }

  static int _flashAttnInt(FlashAttention v) => switch (v) {
        FlashAttention.auto =>
          llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_AUTO.value,
        FlashAttention.off =>
          llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_DISABLED.value,
        FlashAttention.on =>
          llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED.value,
      };

  static int _kvCacheTypeInt(KvCacheType v) => switch (v) {
        KvCacheType.f32 => ggml_type.GGML_TYPE_F32.value,
        KvCacheType.f16 => ggml_type.GGML_TYPE_F16.value,
        KvCacheType.bf16 => ggml_type.GGML_TYPE_BF16.value,
        KvCacheType.q8_0 => ggml_type.GGML_TYPE_Q8_0.value,
        KvCacheType.q4_0 => ggml_type.GGML_TYPE_Q4_0.value,
        KvCacheType.q4_1 => ggml_type.GGML_TYPE_Q4_1.value,
        KvCacheType.q5_0 => ggml_type.GGML_TYPE_Q5_0.value,
        KvCacheType.q5_1 => ggml_type.GGML_TYPE_Q5_1.value,
      };
}
