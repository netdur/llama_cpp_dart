import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' show max;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'context_params.dart';
import 'llama.dart' show Llama, LlamaException, LlamaStatus;
import 'llama_cpp.dart';
import 'llama_input.dart';
import 'model_params.dart';
import 'sampler_params.dart';

/// Multi-user friendly wrapper around llama.cpp.
/// - Reuses loaded weights across service instances when possible.
/// - Uses a single shared context (`llama_context`) to support continuous batching.
/// - Manages multiple sessions via `seq_id`.
class LlamaService {
  static final Map<_ModelCacheKey, _SharedModelHandle> _modelCache = {};

  static llama_cpp get lib => Llama.lib;

  static bool _backendInitialized = false;

  final String modelPath;
  final ModelParams modelParams;
  final ContextParams defaultContextParams;
  final SamplerParams defaultSamplerParams;
  final bool _verbose;

  final Map<String, _ServiceSession> _sessions = {};

  /// Active sessions that have a helper-managed sequence ID.
  final Map<int, _ServiceSession> _sessionsBySeqId = {};

  late final _SharedModelHandle _sharedModel;

  // Shared context and batch
  Pointer<llama_context> _context = nullptr;
  late llama_batch _batch;
  int _nBatch = 0;

  // Multimodal (Vision)
  Pointer<mtmd_context> _mctx = nullptr;
  bool _visionEnabled = false;

  // Management
  bool _disposed = false;
  final _stopSignal = Completer<void>();

  // Optimization: Event Signal
  Completer<void> _workSignal = Completer<void>();

  // Dispose Race Condition Fix
  Completer<void>? _activeDecodeSync;

  // Sequence ID management
  final List<int> _freeSeqIds = [];
  int _maxParallel = 1;

  /// Creates a service that can host multiple sessions on the same weights.
  ///
  /// Note: [contextParams.nSeqMax] controls the maximum number of parallel sequences.
  LlamaService(
    this.modelPath, {
    String? mmprojPath,
    ModelParams? modelParams,
    ContextParams? contextParams,
    SamplerParams? samplerParams,
    bool verbose = false,
  })  : modelParams = modelParams ?? ModelParams(),
        defaultContextParams = contextParams ?? ContextParams(),
        defaultSamplerParams = samplerParams ?? SamplerParams(),
        _verbose = verbose {
    _ensureBackend();
    _sharedModel = _acquireModel(modelPath, this.modelParams);

    if (_verbose) {
      final ptr = lib.llama_print_system_info();
      final sysInfo = ptr.cast<Utf8>().toDartString();
      // ignore: avoid_print
      print(sysInfo);
    }

    // Initialize Vision if provided
    if (mmprojPath != null && mmprojPath.isNotEmpty) {
      final mprojPathPtr = mmprojPath.toNativeUtf8().cast<Char>();
      try {
        var mparam = lib.mtmd_context_params_default();
        mparam.use_gpu = this.modelParams.nGpuLayers != 0;
        _mctx =
            lib.mtmd_init_from_file(mprojPathPtr, _sharedModel.model, mparam);
        if (_mctx == nullptr) {
          throw LlamaException(
              "Failed to create multimodal projector context from $mmprojPath");
        }
        _visionEnabled = true;
      } finally {
        malloc.free(mprojPathPtr);
      }
    }

    // Initialize Shared Context
    final ctxParams = defaultContextParams.get();
    _maxParallel = max(1, ctxParams.n_seq_max);

    // Ensure sufficient slots
    for (int i = 0; i < _maxParallel; i++) {
      _freeSeqIds.add(i);
    }

    try {
      _context =
          lib.llama_new_context_with_model(_sharedModel.model, ctxParams);
      if (_context == nullptr) {
        throw LlamaException("Could not create shared context");
      }

      _nBatch = ctxParams.n_batch;
      // Fix: Initialize batch for Tokens only (n_embd = 0).
      // llama.cpp forbids both token and embd pointers being non-null.
      // For text generation, we need tokens. Vision/Embeddings will require a separate batch configuration or swapping pointers.
      _batch = lib.llama_batch_init(_nBatch, 0, _maxParallel);

      if (_batch.token.address == 0 ||
          _batch.pos.address == 0 ||
          _batch.seq_id.address == 0) {
        throw LlamaException("llama_batch_init failed to allocate memory");
      }
    } catch (e) {
      _releaseModel(_sharedModel);
      rethrow;
    }

    // Start the scheduling loop
    _runLoop();
  }

  /// Returns true when the service or all sessions have been disposed.
  bool get isDisposed => _disposed;

  /// Current status for a specific session.
  LlamaStatus status(String sessionId) =>
      _sessions[sessionId]?.status ?? LlamaStatus.uninitialized;

  /// Creates a new logical user session.
  ///
  /// Assigns a sequence ID from the pool. Throws if max parallel sessions reached.
  void createSession(
    String sessionId, {
    ContextParams?
        contextParams, // Note: Only nPredict/nCtx are relevant per-session now
    SamplerParams? samplerParams,
  }) {
    _checkDisposed();
    if (_sessions.containsKey(sessionId)) return;

    if (_freeSeqIds.isEmpty) {
      throw LlamaException(
          "Max parallel sessions ($_maxParallel) reached. Cannot create session $sessionId.");
    }

    final seqId = _freeSeqIds.removeLast();
    final ctxParams = contextParams ?? defaultContextParams;
    final sampler = _initializeSampler(samplerParams ?? defaultSamplerParams);

    // Clear KV cache for this slot to be safe
    final mem = lib.llama_get_memory(_context);
    lib.llama_memory_seq_rm(mem, seqId, -1, -1);

    final session = _ServiceSession(
      id: sessionId,
      seqId: seqId,
      sampler: sampler,
      contextParams: ctxParams,
    );

    session.decodeBuf = malloc<Char>(256);

    _sessions[sessionId] = session;
    _sessionsBySeqId[seqId] = session;
    session.status = LlamaStatus.ready;
  }

  /// Removes a session and frees its sequence ID.
  Future<void> freeSession(String sessionId) async {
    final session = _sessions.remove(sessionId);
    if (session == null) return;

    _sessionsBySeqId.remove(session.seqId);

    // Clear data associated with this sequence
    if (_context != nullptr) {
      final mem = lib.llama_get_memory(_context);
      // Wait for decode to finish if accidental collision?
      // Unlikely since we removed from map.
      lib.llama_memory_seq_rm(mem, session.seqId, -1, -1);
    }

    session.dispose(lib);
    _freeSeqIds.add(session.seqId);
  }

  /// Sets the prompt for a specific session (starts generation).
  /// [clearHistory]: If true (default), wipes memory for this session before prompting.
  /// If false, appending to existing context.
  Future<void> setPrompt(
    String sessionId,
    String prompt, {
    bool clearHistory = true,
    void Function(int current, int total)? onProgress,
  }) async {
    _checkDisposed();
    final session = _requireSession(sessionId);
    if (prompt.isEmpty) throw ArgumentError('Prompt cannot be empty');

    // Reset session state
    session.pendingItems.clear();
    session.nGenerated = 0;
    session.nPromptTokens = 0;
    session.pendingBytes.clear(); // Resource fix
    session.status = LlamaStatus.generating;

    if (clearHistory) {
      // Clear KV cache for new prompt
      final mem = lib.llama_get_memory(_context);
      lib.llama_memory_seq_rm(mem, session.seqId, -1, -1);
      session.nPos = 0;
      if (session.sampler != nullptr) lib.llama_sampler_reset(session.sampler);
    }

    // Tokenize
    final promptUtf8 = prompt.toNativeUtf8();
    try {
      final nCtx = session.contextParams.nCtx;
      final length = promptUtf8.length;
      final addBos = clearHistory; // Only add BOS if starting fresh

      // Estimate
      int nTokens = -lib.llama_tokenize(_sharedModel.vocab, promptUtf8.cast(),
          length, nullptr, 0, addBos, true);

      if (nTokens < 0)
        nTokens = -nTokens; // Handle negative return code as size

      final tokenPtr = malloc<llama_token>(nTokens);
      try {
        nTokens = lib.llama_tokenize(_sharedModel.vocab, promptUtf8.cast(),
            length, tokenPtr, nTokens, addBos, true);

        if (session.nPos + nTokens > nCtx) {
          throw LlamaException(
              "Prompt too long or context full ($nTokens tokens, pos ${session.nPos} > $nCtx)");
        }

        // Queue tokens
        // Create PromptItems
        for (int i = 0; i < nTokens; i++) {
          session.pendingItems.add(_PendingItem.token(tokenPtr[i]));
        }
        session.nPromptTokens = nTokens;
      } finally {
        malloc.free(tokenPtr);
      }
    } finally {
      malloc.free(promptUtf8);
    }

    _notifyWork();
  }

  /// Same as [generateText] but for CLI compatibility, waits for completion if needed.
  Future<String> generateCompleteText(String sessionId,
      {int? maxTokens}) async {
    final stream = generateText(sessionId);
    final buf = StringBuffer();
    await for (final chunk in stream) {
      buf.write(chunk);
      if (maxTokens != null && buf.length > maxTokens * 4) {
        // crude approx
      }
    }
    return buf.toString();
  }

  /// Stream generation.
  Stream<String> generateText(String sessionId) {
    _checkDisposed();
    final session = _requireSession(sessionId);
    return session.outputStream.stream;
  }

  /// Supports vision. Queues images + text.
  Stream<String> generateWithMedia(
    String sessionId,
    String prompt, {
    required List<LlamaInput> inputs,
    bool clearHistory = true,
  }) async* {
    _checkDisposed();
    if (!_visionEnabled || _mctx == nullptr) {
      throw LlamaException('Vision disabled – construct with mmprojPath.');
    }
    if (inputs.isEmpty) throw ArgumentError('No images given');

    final session = _requireSession(sessionId);

    // We will behave like setPrompt but with mixed content
    session.pendingItems.clear();
    session.pendingItems.clear();
    session.nGenerated = 0;
    session.pendingBytes.clear(); // Resource fix
    session.status = LlamaStatus.generating;

    if (clearHistory) {
      // Clear KV
      final mem = lib.llama_get_memory(_context);
      lib.llama_memory_seq_rm(mem, session.seqId, -1, -1);
      session.nPos = 0;
      if (session.sampler != nullptr) lib.llama_sampler_reset(session.sampler);
    }

    // 1. Convert inputs to chunks
    final images = inputs.whereType<LlamaImage>().toList();
    const marker = '<image>';

    // Fix: Validate markers
    if (marker.allMatches(prompt).length != images.length) {
      throw ArgumentError(
          "Mismatch between <image> markers impacting prompt and provided inputs");
    }

    // Prepare images
    final bitmapRefs = <BitmapPointers>[];
    Pointer<Pointer<mtmd_bitmap>>? bmpArr; // nullable for safely freeing

    Pointer<mtmd_input_chunks> chunks = nullptr;
    Pointer<mtmd_input_text> txtPtr = nullptr;
    Pointer<Char> fullPtr = nullptr;

    try {
      for (final img in images) {
        bitmapRefs.add(img.toBitmap(lib, malloc));
      }
      bmpArr = malloc<Pointer<mtmd_bitmap>>(bitmapRefs.length);
      for (var i = 0; i < bitmapRefs.length; ++i) {
        bmpArr[i] = bitmapRefs[i].bitmap;
      }

      final modelMark = lib.mtmd_default_marker().cast<Utf8>().toDartString();
      final fullPrompt = prompt.replaceAll(marker, modelMark);
      fullPtr = fullPrompt.toNativeUtf8().cast<Char>();

      txtPtr = calloc<mtmd_input_text>();
      txtPtr.ref
        ..text = fullPtr
        ..add_special = true
        ..parse_special = true;

      chunks = lib.mtmd_input_chunks_init();

      final tk =
          lib.mtmd_tokenize(_mctx, chunks, txtPtr, bmpArr, bitmapRefs.length);
      if (tk != 0) throw LlamaException('mtmd_tokenize failed ($tk)');

      final nChunks = lib.mtmd_input_chunks_size(chunks);

      for (var i = 0; i < nChunks; ++i) {
        final chunk = lib.mtmd_input_chunks_get(chunks, i);
        final type = lib.mtmd_input_chunk_get_type(chunk);

        if (type == mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_IMAGE) {
          if (lib.mtmd_encode_chunk(_mctx, chunk) != 0) {
            throw LlamaException('encode image failed');
          }
          // Get embeddings
          final embdPtr = lib.mtmd_get_output_embd(_mctx);
          final nTok = lib.mtmd_input_chunk_get_n_tokens(chunk);
          final nEmbd = lib.llama_n_embd(_sharedModel.model);

          final totalFloats = nTok * nEmbd;
          final floatList = embdPtr.asTypedList(totalFloats);
          final copy = Float32List.fromList(floatList);

          session.pendingItems.add(_PendingItem.embedding(copy, nTok));
        } else {
          // Text
          final nPtr = malloc<Size>();
          final tokPt = lib.mtmd_input_chunk_get_tokens_text(chunk, nPtr);
          final nTok = nPtr.value;
          malloc.free(nPtr);

          for (int k = 0; k < nTok; k++) {
            session.pendingItems.add(_PendingItem.token(tokPt[k]));
          }
        }
      }
    } finally {
      if (chunks != nullptr) lib.mtmd_input_chunks_free(chunks);
      if (bmpArr != null) malloc.free(bmpArr);
      for (final r in bitmapRefs) {
        if (r.bitmap != nullptr) lib.mtmd_bitmap_free(r.bitmap);
      }
      if (txtPtr != nullptr) calloc.free(txtPtr);
      if (fullPtr != nullptr) malloc.free(fullPtr);
    }

    _notifyWork();

    // Return stream
    yield* session.outputStream.stream;
  }

  void _notifyWork() {
    if (!_workSignal.isCompleted) {
      _workSignal.complete();
    }
  }

  void saveSession(String sessionId, String path) {
    if (_verbose)
      print(
          "Warning: saveSession not fully supported in continuous batching mode");
  }

  bool loadSession(String sessionId, String path) {
    if (_verbose)
      print(
          "Warning: loadSession not fully supported in continuous batching mode");
    return false;
  }

  // --- Internal Scheduling Loop ---

  Future<void> _runLoop() async {
    while (!_disposed) {
      // 1. Identify work
      bool anyWork = false;
      int batchIdx = 0;
      _batch.n_tokens = 0;

      bool? batchIsEmbedding;

      final activeSessions = _sessions.values.toList();
      for (final session in activeSessions) {
        if (session.status != LlamaStatus.generating) continue;

        // Check context limits
        if (session.nPos >= session.contextParams.nCtx - 2) {
          session.status = LlamaStatus.ready;
          session.outputStream.add("\n[Context Limit]");
          // Fix: Clear context and reset pos to avoid loop
          // Ideally we should implement rolling context, but for now just stop/reset safe.
          // Or just stop. But user said "Prevent Context Stalling".
          // "If session.nPos exceeds context, specifically clear the KV cache... reset nPos to 0... clear session.pendingItems"
          final mem = lib.llama_get_memory(_context);
          lib.llama_memory_seq_rm(mem, session.seqId, -1, -1);
          session.nPos = 0;
          session.pendingItems.clear();

          continue;
        }

        int available = _nBatch - batchIdx;
        if (available <= 0) break;

        if (session.pendingItems.isNotEmpty) {
          // PROCESSING INPUT
          var item = session.pendingItems.first;
          bool isEmb = item.isEmbedding;

          if (batchIsEmbedding == null) batchIsEmbedding = isEmb;
          if (batchIsEmbedding != isEmb) continue; // Skip mixed types

          if (isEmb) {
            if (item.nTokens > available) {
              if (batchIdx == 0) {
                throw LlamaException(
                    "Embedding chunk size ${item.nTokens} > batch size $_nBatch");
              }
              continue; // Wait for next tick
            }

            final nEmbd = lib.llama_n_embd(_sharedModel.model);
            final destPtr = _batch.embd.elementAt(batchIdx * nEmbd);

            final floatCount = item.values!.length;
            final destList = destPtr.asTypedList(floatCount);
            destList.setAll(0, item.values!);

            for (int k = 0; k < item.nTokens; k++) {
              _batch.pos[batchIdx + k] = session.nPos + k;
              _batch.n_seq_id[batchIdx + k] = 1;
              _batch.seq_id[batchIdx + k][0] = session.seqId;
              _batch.logits[batchIdx + k] = 0;
            }

            session.nPos += item.nTokens;

            bool isLast = (session.pendingItems.length == 1);
            if (isLast) {
              _batch.logits[batchIdx + item.nTokens - 1] = 1;
            }

            batchIdx += item.nTokens;
            session.pendingItems.removeAt(0);
            anyWork = true;
          } else {
            // Token input
            int count = 0;
            while (count < available && session.pendingItems.isNotEmpty) {
              if (session.pendingItems.first.isEmbedding) break;
              final tItem = session.pendingItems.removeAt(0);

              _batch.token[batchIdx] = tItem.token!;
              _batch.pos[batchIdx] = session.nPos;
              _batch.n_seq_id[batchIdx] = 1;
              _batch.seq_id[batchIdx][0] = session.seqId;
              _batch.logits[batchIdx] = 0;

              session.nPos++;
              batchIdx++;
              count++;

              // If this was the last pending item for this session, we need logits to sample the next token
              if (session.pendingItems.isEmpty) {
                // Fix: batchIdx is already incremented, so the last item is at batchIdx - 1
                _batch.logits[batchIdx - 1] = 1;
              }
            }
            anyWork = true;
          }
        }
      }

      if (!anyWork) {
        // Case A: Work was signaled while we were processing the last batch.
        // We should loop immediately without sleeping.
        if (_workSignal.isCompleted) {
          _workSignal = Completer<void>();
          continue;
        }

        // Case B: No work pending. Sleep until notified.
        // print("Service: Waiting for work...");
        await _workSignal.future;

        // We woke up (work arrived). Reset signal for the next wait.
        _workSignal = Completer<void>();
        continue;
      }

      _batch.n_tokens = batchIdx;

      // 2. Decode with Safety
      _activeDecodeSync = Completer<void>();

      if (lib.llama_decode(_context, _batch) != 0) {
        print("llama_decode failed");
        _activeDecodeSync?.complete();
        _activeDecodeSync = null;
        await Future.delayed(const Duration(milliseconds: 100));
        continue;
      }

      _activeDecodeSync?.complete();
      _activeDecodeSync = null;

      // 3. Sample & Post-Process
      for (int i = 0; i < batchIdx; i++) {
        if (_batch.logits[i] != 0) {
          final sId = _batch.seq_id[i][0];
          final session = _sessionsBySeqId[sId];
          if (session == null) continue;

          final newToken =
              lib.llama_sampler_sample(session.sampler, _context, i);

          // print("Service: Sampled token $newToken for session $sId");

          // Fix: Accept token
          lib.llama_sampler_accept(session.sampler, newToken);

          bool isEos = lib.llama_token_is_eog(_sharedModel.vocab, newToken);
          String piece = "";
          if (isEos) {
            piece = " [EOS]";
            if (session.pendingBytes.isNotEmpty) {
              // Decode remaining bytes if any
              piece = utf8.decode(session.pendingBytes, allowMalformed: true) +
                  piece;
              session.pendingBytes.clear();
            }
          } else {
            piece = _decodeToken(session, newToken);
          }

          if (piece.isNotEmpty) session.outputStream.add(piece);
          session.nGenerated++;

          bool done = isEos;
          if (session.contextParams.nPredict != -1 &&
              session.nGenerated >= session.contextParams.nPredict) {
            done = true;
          }

          if (done) {
            session.status = LlamaStatus.ready;
          } else {
            session.pendingItems.add(_PendingItem.token(newToken));
            // Trigger next loop immediately since we have a new item
            // But we can just loop.
          }
        }
      }

      // Yield to event loop to allow other microtasks
      await Future.delayed(Duration.zero);
    }
  }

  // --- Internal Helpers ---

  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    _stopSignal.complete();
    _notifyWork(); // Wake up loop to finish

    // Wait for active decode to finish
    if (_activeDecodeSync != null) {
      await _activeDecodeSync!.future
          .timeout(const Duration(seconds: 1), onTimeout: () {});
    }

    // Dispose sessions
    for (final s in _sessions.values) {
      s.dispose(lib);
    }
    _sessions.clear();

    // Dispose shared
    if (_context != nullptr) lib.llama_free(_context);
    try {
      lib.llama_batch_free(_batch);
    } catch (_) {}

    if (_visionEnabled && _mctx != nullptr) {
      lib.mtmd_free(_mctx);
    }

    _releaseModel(_sharedModel);
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('LlamaService disposed');
  }

  void _ensureBackend() {
    if (_backendInitialized) return;
    lib.llama_backend_init();
    if (!Platform.isAndroid) {
      lib.ggml_backend_load_all();
    }
    _backendInitialized = true;
  }

  _ServiceSession _requireSession(String id) {
    final session = _sessions[id];
    if (session == null) {
      throw ArgumentError(
          "Session $id does not exist. Call createSession first.");
    }
    return session;
  }

  Pointer<llama_sampler> _initializeSampler(SamplerParams samplerParams) {
    llama_sampler_chain_params sparams =
        lib.llama_sampler_chain_default_params();
    sparams.no_perf = false;
    final smpl = lib.llama_sampler_chain_init(sparams);

    if (samplerParams.greedy) {
      lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_greedy());
      return smpl;
    }

    final grammarStrPtr = samplerParams.grammarStr.toNativeUtf8().cast<Char>();
    final grammarRootPtr =
        samplerParams.grammarRoot.toNativeUtf8().cast<Char>();
    if (samplerParams.grammarStr.isNotEmpty) {
      final grammar = lib.llama_sampler_init_grammar(
          _sharedModel.vocab, grammarStrPtr, grammarRootPtr);
      if (grammar != nullptr) lib.llama_sampler_chain_add(smpl, grammar);
    }
    malloc.free(grammarStrPtr);
    malloc.free(grammarRootPtr);

    lib.llama_sampler_chain_add(
        smpl,
        lib.llama_sampler_init_penalties(
          samplerParams.penaltyLastTokens,
          samplerParams.penaltyRepeat,
          samplerParams.penaltyFreq,
          samplerParams.penaltyPresent,
        ));

    lib.llama_sampler_chain_add(
        smpl, lib.llama_sampler_init_top_k(samplerParams.topK));
    lib.llama_sampler_chain_add(
        smpl, lib.llama_sampler_init_top_p(samplerParams.topP, 1));
    lib.llama_sampler_chain_add(
        smpl, lib.llama_sampler_init_min_p(samplerParams.minP, 1));
    lib.llama_sampler_chain_add(
        smpl, lib.llama_sampler_init_temp(samplerParams.temp));
    lib.llama_sampler_chain_add(
        smpl, lib.llama_sampler_init_dist(samplerParams.seed));

    return smpl;
  }

  _SharedModelHandle _acquireModel(String path, ModelParams params) {
    final key = _ModelCacheKey(path, params);
    final cached = _modelCache[key];
    if (cached != null) {
      cached.refs++;
      return cached;
    }

    final modelParams = params.get();
    final modelPathPtr = path.toNativeUtf8().cast<Char>();
    Pointer<llama_model> loadedModel = nullptr;
    Pointer<llama_vocab> vocab = nullptr;
    try {
      loadedModel = lib.llama_load_model_from_file(modelPathPtr, modelParams);
      if (loadedModel == nullptr) {
        throw LlamaException("Could not load model at $path");
      }
      vocab = lib.llama_model_get_vocab(loadedModel);
    } finally {
      malloc.free(modelPathPtr);
    }

    final handle = _SharedModelHandle(model: loadedModel, vocab: vocab);
    _modelCache[key] = handle;
    return handle;
  }

  void _releaseModel(_SharedModelHandle handle) {
    handle.refs--;
    if (handle.refs > 0) return;
    _modelCache.removeWhere((_, h) => identical(h, handle));
    if (handle.model.address != 0) {
      lib.llama_free_model(handle.model);
    }
  }

  String _decodeToken(_ServiceSession session, int tokenId) {
    if (session.decodeBuf == nullptr) session.decodeBuf = malloc<Char>(256);
    final buf = session.decodeBuf;

    try {
      int n = lib.llama_token_to_piece(
          _sharedModel.vocab, tokenId, buf, 256, 0, true);
      if (n < 0) return "";

      final newBytes = buf.cast<Uint8>().asTypedList(n);
      session.pendingBytes.addAll(newBytes);

      try {
        String piece = utf8.decode(session.pendingBytes);
        session.pendingBytes.clear();
        return piece;
      } catch (e) {
        return "";
      }
    } catch (_) {
      return "";
    }
  }
}

class _PendingItem {
  final int? token;
  final Float32List? values; // for embeddings
  final int nTokens; // size in tokens

  _PendingItem.token(this.token)
      : values = null,
        nTokens = 1;
  _PendingItem.embedding(this.values, this.nTokens) : token = null;

  bool get isEmbedding => values != null;
}

class _ServiceSession {
  final String id;
  final int seqId;
  final Pointer<llama_sampler> sampler;
  final ContextParams contextParams;

  LlamaStatus status = LlamaStatus.uninitialized;
  Pointer<Char> decodeBuf = nullptr;
  List<int> pendingBytes = []; // For utf8 decoding

  final List<_PendingItem> pendingItems = [];
  final StreamController<String> outputStream = StreamController.broadcast();

  int nPos = 0; // Current context position
  int nPromptTokens = 0;
  int nGenerated = 0;

  _ServiceSession({
    required this.id,
    required this.seqId,
    required this.sampler,
    required this.contextParams,
  });

  void dispose(llama_cpp lib) {
    if (sampler != nullptr) {
      lib.llama_sampler_free(sampler);
    }
    if (decodeBuf != nullptr) {
      malloc.free(decodeBuf);
    }
    outputStream.close();
  }
}

class _SharedModelHandle {
  final Pointer<llama_model> model;
  final Pointer<llama_vocab> vocab;
  int refs = 1;

  _SharedModelHandle({required this.model, required this.vocab});
}

class _ModelCacheKey {
  final String path;
  final String paramsSignature;

  _ModelCacheKey(this.path, ModelParams params)
      : paramsSignature = params.toString();

  @override
  bool operator ==(Object other) {
    return other is _ModelCacheKey &&
        other.path == path &&
        other.paramsSignature == paramsSignature;
  }

  @override
  int get hashCode => Object.hash(path, paramsSignature);
}
