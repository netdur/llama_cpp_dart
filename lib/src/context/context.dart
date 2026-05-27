import 'dart:ffi';
import 'dart:typed_data';

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
      ..n_rs_seq = params.nRsSeq
      ..n_threads = params.nThreads
      ..n_threads_batch = params.nThreadsBatch
      ..ctx_typeAsInt = _ctxTypeInt(params.ctxType)
      ..flash_attn_typeAsInt = _flashAttnInt(params.flashAttn)
      ..rope_scaling_typeAsInt = _ropeScalingInt(params.ropeScalingType)
      ..pooling_typeAsInt = _poolingInt(params.poolingType)
      ..attention_typeAsInt = _attentionInt(params.attentionType)
      ..rope_freq_base = params.ropeFreqBase
      ..rope_freq_scale = params.ropeFreqScale
      ..yarn_ext_factor = params.yarnExtFactor
      ..yarn_attn_factor = params.yarnAttnFactor
      ..yarn_beta_fast = params.yarnBetaFast
      ..yarn_beta_slow = params.yarnBetaSlow
      ..yarn_orig_ctx = params.yarnOrigCtx
      ..defrag_thold = params.defragThreshold
      ..offload_kqv = params.offloadKqv
      ..embeddings = params.embeddings
      ..no_perf = params.noPerf
      ..op_offload = params.opOffload
      ..swa_full = params.swaFull
      ..kv_unified = params.kvUnified
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

  /// Per-sequence context window. Equals [nCtx] when the unified KV cache is
  /// disabled and divides it when [ContextParams.kvUnified] is true.
  int get nCtxSeq => LlamaLibrary.bindings.llama_n_ctx_seq(pointer);

  /// Number of recurrent-state rollback snapshots active per sequence.
  /// Mirrors [ContextParams.nRsSeq] after the runtime resolved it.
  int get nRsSeq => LlamaLibrary.bindings.llama_n_rs_seq(pointer);

  /// Effective pooling type the context settled on. When
  /// [ContextParams.poolingType] is `auto` the model decides; this reports
  /// the result.
  PoolingType get poolingType {
    final raw = LlamaLibrary.bindings.llama_pooling_type$1(pointer);
    return switch (raw) {
      llama_pooling_type.LLAMA_POOLING_TYPE_NONE => PoolingType.none,
      llama_pooling_type.LLAMA_POOLING_TYPE_MEAN => PoolingType.mean,
      llama_pooling_type.LLAMA_POOLING_TYPE_CLS => PoolingType.cls,
      llama_pooling_type.LLAMA_POOLING_TYPE_LAST => PoolingType.last,
      llama_pooling_type.LLAMA_POOLING_TYPE_RANK => PoolingType.rank,
      _ => PoolingType.auto,
    };
  }

  /// True if this context's memory backend supports `llama_memory_seq_add`
  /// (i.e., position shifting). False for recurrent and most iSWA caches —
  /// shifting them is mathematically unsound. Used to gate context-shift
  /// policies; mirrors `llama_memory_can_shift` exactly.
  bool get canShift {
    final lib = LlamaLibrary.bindings;
    return lib.llama_memory_can_shift(lib.llama_get_memory(pointer));
  }

  /// Switch the thread counts at runtime.
  ///
  /// [n] is the generation thread count, [nBatch] the prefill thread count.
  /// `0` for either lets llama.cpp pick.
  void setThreads({required int n, required int nBatch}) {
    LlamaLibrary.bindings.llama_set_n_threads(pointer, n, nBatch);
  }

  /// Toggle embedding-output mode. Lets a single context flip between
  /// generation and embedding without rebuild. The pooling type is fixed
  /// at create time, so this is mostly useful for embedding models that
  /// also support causal generation.
  void setEmbeddings(bool value) {
    LlamaLibrary.bindings.llama_set_embeddings(pointer, value);
  }

  /// Toggle causal attention at runtime. Most chat / completion workloads
  /// want `true`; switch off for bidirectional / encoder-style passes.
  void setCausalAttn(bool value) {
    LlamaLibrary.bindings.llama_set_causal_attn(pointer, value);
  }

  /// Mark the context as in a warmup phase (skips some bookkeeping for
  /// faster first decode). Call with `false` once warmup completes.
  void setWarmup(bool value) {
    LlamaLibrary.bindings.llama_set_warmup(pointer, value);
  }

  /// Block until pending async work on the context's compute backends has
  /// completed. Mostly relevant on GPU backends; a no-op on pure CPU.
  void synchronize() {
    LlamaLibrary.bindings.llama_synchronize(pointer);
  }

  /// Copy the logits for the last position with `wantLogits = true` in the
  /// most recently decoded batch. Length matches [LlamaModel.nVocab].
  /// Returns `null` if the runtime did not produce logits for that position.
  Float32List? lastLogits() {
    final ptr = LlamaLibrary.bindings.llama_get_logits(pointer);
    if (ptr == nullptr) return null;
    return Float32List.fromList(ptr.asTypedList(model.vocab.nTokens));
  }

  /// Copy the logits for batch position [i]. Same caveats as [lastLogits].
  /// Use `-1` to mean "the last position with logits enabled."
  Float32List? logitsAt(int i) {
    final ptr = LlamaLibrary.bindings.llama_get_logits_ith(pointer, i);
    if (ptr == nullptr) return null;
    return Float32List.fromList(ptr.asTypedList(model.vocab.nTokens));
  }

  /// Token id picked by the sampler for batch position [i], assuming a
  /// `llama_sampler_decode_with_*` path produced it. Returns `-1` when no
  /// sampled token is available for that position.
  int sampledTokenAt(int i) =>
      LlamaLibrary.bindings.llama_get_sampled_token_ith(pointer, i);

  /// Top-k sampled probabilities for batch position [i]. Length matches
  /// the runtime's top-k snapshot for that position.
  Float32List? sampledProbsAt(int i) {
    final lib = LlamaLibrary.bindings;
    final n = lib.llama_get_sampled_probs_count_ith(pointer, i);
    if (n == 0) return null;
    final ptr = lib.llama_get_sampled_probs_ith(pointer, i);
    if (ptr == nullptr) return null;
    return Float32List.fromList(ptr.asTypedList(n));
  }

  /// Top-k sampled candidate token ids for batch position [i]. Pairs
  /// index-for-index with [sampledProbsAt] / [sampledLogitsAt].
  Int32List? sampledCandidatesAt(int i) {
    final lib = LlamaLibrary.bindings;
    final n = lib.llama_get_sampled_candidates_count_ith(pointer, i);
    if (n == 0) return null;
    final ptr = lib.llama_get_sampled_candidates_ith(pointer, i);
    if (ptr == nullptr) return null;
    return Int32List.fromList(ptr.asTypedList(n));
  }

  /// Top-k sampled logits for batch position [i].
  Float32List? sampledLogitsAt(int i) {
    final lib = LlamaLibrary.bindings;
    final n = lib.llama_get_sampled_logits_count_ith(pointer, i);
    if (n == 0) return null;
    final ptr = lib.llama_get_sampled_logits_ith(pointer, i);
    if (ptr == nullptr) return null;
    return Float32List.fromList(ptr.asTypedList(n));
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

  static int _ctxTypeInt(ContextType v) => switch (v) {
        ContextType.defaultCtx =>
          llama_context_type.LLAMA_CONTEXT_TYPE_DEFAULT.value,
        ContextType.mtp => llama_context_type.LLAMA_CONTEXT_TYPE_MTP.value,
      };

  static int _flashAttnInt(FlashAttention v) => switch (v) {
        FlashAttention.auto =>
          llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_AUTO.value,
        FlashAttention.off =>
          llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_DISABLED.value,
        FlashAttention.on =>
          llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED.value,
      };

  static int _ropeScalingInt(RopeScalingType v) => switch (v) {
        RopeScalingType.auto =>
          llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED.value,
        RopeScalingType.none =>
          llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_NONE.value,
        RopeScalingType.linear =>
          llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_LINEAR.value,
        RopeScalingType.yarn =>
          llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_YARN.value,
        RopeScalingType.longrope =>
          llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_LONGROPE.value,
      };

  static int _poolingInt(PoolingType v) => switch (v) {
        PoolingType.auto =>
          llama_pooling_type.LLAMA_POOLING_TYPE_UNSPECIFIED.value,
        PoolingType.none => llama_pooling_type.LLAMA_POOLING_TYPE_NONE.value,
        PoolingType.mean => llama_pooling_type.LLAMA_POOLING_TYPE_MEAN.value,
        PoolingType.cls => llama_pooling_type.LLAMA_POOLING_TYPE_CLS.value,
        PoolingType.last => llama_pooling_type.LLAMA_POOLING_TYPE_LAST.value,
        PoolingType.rank => llama_pooling_type.LLAMA_POOLING_TYPE_RANK.value,
      };

  static int _attentionInt(AttentionType v) => switch (v) {
        AttentionType.auto =>
          llama_attention_type.LLAMA_ATTENTION_TYPE_UNSPECIFIED.value,
        AttentionType.causal =>
          llama_attention_type.LLAMA_ATTENTION_TYPE_CAUSAL.value,
        AttentionType.nonCausal =>
          llama_attention_type.LLAMA_ATTENTION_TYPE_NON_CAUSAL.value,
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
