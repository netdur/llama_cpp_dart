import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' show max, min;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'context_params.dart';
import 'llama.dart' show Llama, LlamaException, LlamaStatus;
import 'llama_cpp.dart';
import 'llama_input.dart';
import 'model_params.dart';
import 'sampler_params.dart';

/// Multi-user friendly wrapper around llama.cpp.
class LlamaService {
  static final Map<_ModelCacheKey, _SharedModelHandle> _modelCache = {};

  static llama_cpp get lib => Llama.lib;

  static bool _backendInitialized = false;
  static const int _maxTokensPerSessionPerBatch = 16;

  final String modelPath;
  final ModelParams modelParams;
  final ContextParams defaultContextParams;
  final ContextParams _ctxConfig;
  final SamplerParams defaultSamplerParams;
  final bool _verbose;

  final Map<String, _ServiceSession> _sessions = {};

  final Map<int, _ServiceSession> _sessionsBySeqId = {};

  late final _SharedModelHandle _sharedModel;

  // Shared context and batch
  Pointer<llama_context> _context = nullptr;
  late llama_batch _batch;
  late llama_batch _batchEmbd;

  // Storage for the token pointer we hide from llama_decode
  Pointer<llama_token> _hiddenTokenPtr = nullptr;

  int _nBatch = 0;
  int _contextNCtx = 0;
  int _contextNPredict = -1;
  int _lastScheduledSessionIndex = 0; // For Round-Robin

  // Multimodal (Vision)
  Pointer<mtmd_context> _mctx = nullptr;
  bool _visionEnabled = false;

  // Management
  bool _disposed = false;
  final _stopSignal = Completer<void>();
  Completer<void> _workSignal = Completer<void>();
  Completer<void>? _activeDecodeSync;

  final List<int> _freeSeqIds = [];
  int _maxParallel = 1;

  LlamaService(
    this.modelPath, {
    String? mmprojPath,
    ModelParams? modelParams,
    ContextParams? contextParams,
    SamplerParams? samplerParams,
    bool verbose = false,
  })  : modelParams = modelParams ?? ModelParams(),
        _ctxConfig = contextParams ?? ContextParams(),
        defaultContextParams = _ctxConfig,
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

    final ctxParams = _ctxConfig.get();
    _contextNCtx = ctxParams.n_ctx;
    _contextNPredict = _ctxConfig.nPredict;
    _maxParallel = max(1, ctxParams.n_seq_max);

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

      // Initialize Text Batch
      // We pass 0 for n_embd so the library sets batch.embd = nullptr
      _batch = lib.llama_batch_init(_nBatch, 0, _maxParallel);

      if (_batch.token.address == 0 ||
          _batch.pos.address == 0 ||
          _batch.seq_id.address == 0) {
        throw LlamaException("llama_batch_init failed to allocate memory");
      }

      // Initialize Embedding Batch if Vision is enabled
      if (_visionEnabled) {
        final nEmbd = lib.llama_n_embd(_sharedModel.model);
        // We allocate with _nBatch tokens to ensure 'pos', 'seq_id' arrays are allocated
        _batchEmbd = lib.llama_batch_init(_nBatch, nEmbd, _maxParallel);

        // FIX: Hide the token pointer so llama_decode works, but don't free it
        _hiddenTokenPtr = _batchEmbd.token;
        _batchEmbd.token = nullptr;
      }
    } catch (e) {
      if (_context != nullptr) lib.llama_free(_context);
      try {
        // Attempt to clean up batches if they were initialized
        if (_batch.token.address != 0) lib.llama_batch_free(_batch);
      } catch (_) {}
      try {
        if (_visionEnabled &&
            _batchEmbd.token.address != 0 && // Check structure validity
            _batchEmbd.token != nullptr) {
          // Check pointer validity (though we hid it)
          // If we hid the pointer in _hiddenTokenPtr, we must restore it to free
          if (_hiddenTokenPtr != nullptr) _batchEmbd.token = _hiddenTokenPtr;
          lib.llama_batch_free(_batchEmbd);
        }
      } catch (_) {}

      if (_visionEnabled && _mctx != nullptr) {
        lib.mtmd_free(_mctx);
      }

      _releaseModel(_sharedModel);
      rethrow;
    }

    _runLoop();
  }

  bool get isDisposed => _disposed;

  LlamaStatus status(String sessionId) =>
      _sessions[sessionId]?.status ?? LlamaStatus.uninitialized;

  void createSession(
    String sessionId, {
    SamplerParams? samplerParams,
  }) {
    _checkDisposed();
    if (_sessions.containsKey(sessionId)) return;

    if (_freeSeqIds.isEmpty) {
      throw LlamaException(
          "Max parallel sessions ($_maxParallel) reached. Cannot create session $sessionId.");
    }

    final seqId = _freeSeqIds.removeLast();
    final sampler = _initializeSampler(samplerParams ?? defaultSamplerParams);

    final mem = lib.llama_get_memory(_context);
    lib.llama_memory_seq_rm(mem, seqId, -1, -1);

    final session = _ServiceSession(
      id: sessionId,
      seqId: seqId,
      sampler: sampler,
    );

    session.decodeCapacity = 256;
    session.decodeBuf = malloc<Char>(session.decodeCapacity);

    _sessions[sessionId] = session;
    _sessionsBySeqId[seqId] = session;
    session.status = LlamaStatus.ready;
    _startRequest(session);
  }

  Future<void> freeSession(String sessionId) async {
    final session = _sessions.remove(sessionId);
    if (session == null) return;

    _sessionsBySeqId.remove(session.seqId);

    if (_context != nullptr) {
      final mem = lib.llama_get_memory(_context);
      lib.llama_memory_seq_rm(mem, session.seqId, -1, -1);
    }

    session.dispose(lib);
    _freeSeqIds.add(session.seqId);
  }

  Future<void> setPrompt(
    String sessionId,
    String prompt, {
    bool clearHistory = true,
    void Function(int current, int total)? onProgress,
  }) async {
    _checkDisposed();
    final session = _requireSession(sessionId);
    if (prompt.isEmpty) throw ArgumentError('Prompt cannot be empty');
    if (session.status == LlamaStatus.generating) {
      throw LlamaException(
          "Session $sessionId already generating. Finish or cancel before starting a new request.");
    }

    session.pendingItems.clear();
    session.nGenerated = 0;
    session.nPromptTokens = 0;

    session.status = LlamaStatus.generating;
    _startRequest(session);

    if (clearHistory) {
      final mem = lib.llama_get_memory(_context);
      lib.llama_memory_seq_rm(mem, session.seqId, -1, -1);
      session.nPos = 0;
      if (session.sampler != nullptr) lib.llama_sampler_reset(session.sampler);
    }

    final promptUtf8 = prompt.toNativeUtf8();
    try {
      final nCtx = _contextNCtx;
      final length = promptUtf8.length;
      final addBos = clearHistory;

      int nTokens = -lib.llama_tokenize(_sharedModel.vocab, promptUtf8.cast(),
          length, nullptr, 0, addBos, true);

      if (nTokens < 0) nTokens = -nTokens;

      final tokenPtr = malloc<llama_token>(nTokens);
      try {
        nTokens = lib.llama_tokenize(_sharedModel.vocab, promptUtf8.cast(),
            length, tokenPtr, nTokens, addBos, true);

        if (session.nPos + nTokens > nCtx) {
          throw LlamaException(
              "Prompt too long or context full ($nTokens tokens, pos ${session.nPos} > $nCtx)");
        }

        for (int i = 0; i < nTokens; i++) {
          session.pendingItems.addLast(_PendingItem.token(tokenPtr[i]));
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

  Future<String> generateCompleteText(String sessionId,
      {int? maxTokens}) async {
    final stream = generateText(sessionId);
    final buf = StringBuffer();
    await for (final chunk in stream) {
      buf.write(chunk);
    }
    return buf.toString();
  }

  Stream<String> generateText(String sessionId) {
    _checkDisposed();
    final session = _requireSession(sessionId);
    return session.outputStream.stream;
  }

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
    if (session.status == LlamaStatus.generating) {
      throw LlamaException(
          "Session $sessionId already generating. Finish or cancel before starting a new request.");
    }

    session.pendingItems.clear();
    session.nGenerated = 0;

    session.status = LlamaStatus.generating;
    _startRequest(session);

    if (clearHistory) {
      final mem = lib.llama_get_memory(_context);
      lib.llama_memory_seq_rm(mem, session.seqId, -1, -1);
      session.nPos = 0;
      if (session.sampler != nullptr) lib.llama_sampler_reset(session.sampler);
    }

    final images = inputs.whereType<LlamaImage>().toList();
    const marker = '<image>';

    if (marker.allMatches(prompt).length != images.length) {
      throw ArgumentError(
          "Mismatch between <image> markers impacting prompt and provided inputs");
    }

    final bitmapRefs = <BitmapPointers>[];
    Pointer<Pointer<mtmd_bitmap>>? bmpArr;

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
          final embdPtr = lib.mtmd_get_output_embd(_mctx);
          final nTok = lib.mtmd_input_chunk_get_n_tokens(chunk);
          final nEmbd = lib.llama_n_embd(_sharedModel.model);

          final totalFloats = nTok * nEmbd;
          final floatList = embdPtr.asTypedList(totalFloats);
          final copy = Float32List.fromList(floatList);

          session.pendingItems.addLast(_PendingItem.embedding(copy, nTok));
        } else {
          final nPtr = malloc<Size>();
          final tokPt = lib.mtmd_input_chunk_get_tokens_text(chunk, nPtr);
          final nTok = nPtr.value;
          malloc.free(nPtr);

          for (int k = 0; k < nTok; k++) {
            session.pendingItems.addLast(_PendingItem.token(tokPt[k]));
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

    yield* session.outputStream.stream;
  }

  void _notifyWork() {
    if (!_workSignal.isCompleted) {
      _workSignal.complete();
    }
  }

  void saveSession(String sessionId, String path) {}

  bool loadSession(String sessionId, String path) {
    return false;
  }

  // --- Internal Scheduling Loop ---

  Future<void> _runLoop() async {
    while (!_disposed) {
      if (_loopTimer.elapsedMilliseconds >= 16) {
        await Future.delayed(Duration.zero);
        _loopTimer.reset();
      }

      final activeSessions = _sessions.values.toList();
      if (activeSessions.isEmpty) {
        if (_workSignal.isCompleted) {
          _workSignal = Completer<void>();
        }
        await _workSignal.future;
        _workSignal = Completer<void>();
        continue;
      }
      final Map<_ServiceSession, int> sessionIndex = {};
      for (var i = 0; i < activeSessions.length; i++) {
        sessionIndex[activeSessions[i]] = i;
      }

      // Round-Robin Rotation
      int rotateStart =
          (_lastScheduledSessionIndex + 1) % activeSessions.length;
      final rotatedSessions = [
        ...activeSessions.sublist(rotateStart),
        ...activeSessions.sublist(0, rotateStart)
      ];

      final passes = _visionEnabled ? [false, true] : [false];

      bool didWork = false;

      for (final isEmbeddingPass in passes) {
        int batchIdx = 0;
        llama_batch currentBatch = isEmbeddingPass ? _batchEmbd : _batch;
        currentBatch.n_tokens = 0;
        final Set<int> batchSeqIds = {};

        for (final session in rotatedSessions) {
          if (session.status != LlamaStatus.generating) continue;

          if (session.nPos >= _contextNCtx - 2) {
            session.status = LlamaStatus.ready;
            session.outputStream.add("\n[Context Limit]");
            final mem = lib.llama_get_memory(_context);
            lib.llama_memory_seq_rm(mem, session.seqId, -1, -1);
            session.nPos = 0;
            session.pendingItems.clear();
            session.outputStream.close();
            continue;
          }

          int available = _nBatch - batchIdx;
          if (available <= 0) break;

          if (session.pendingItems.isEmpty) {
            _lastScheduledSessionIndex =
                sessionIndex[session] ?? _lastScheduledSessionIndex;
            continue;
          }

          final item = session.pendingItems.first;
          if (item.isEmbedding != isEmbeddingPass) {
            _lastScheduledSessionIndex =
                sessionIndex[session] ?? _lastScheduledSessionIndex;
            continue;
          }

          final int perSessionQuota =
              min(available, _maxTokensPerSessionPerBatch);

          if (isEmbeddingPass) {
            final remaining = item.remainingEmbeddingTokens;
            if (remaining <= 0) {
              session.pendingItems.removeFirst();
              _lastScheduledSessionIndex =
                  sessionIndex[session] ?? _lastScheduledSessionIndex;
              continue;
            }

            final sendTokens = min(perSessionQuota, remaining);
            final nEmbd = lib.llama_n_embd(_sharedModel.model);
            final start = item.embdOffsetTokens * nEmbd;
            final floatCount = sendTokens * nEmbd;

            final destPtr = currentBatch.embd.elementAt(batchIdx * nEmbd);
            final destList = destPtr.asTypedList(floatCount);
            destList.setRange(0, floatCount, item.values!, start);

            for (int k = 0; k < sendTokens; k++) {
              currentBatch.pos[batchIdx + k] = session.nPos + k;
              currentBatch.n_seq_id[batchIdx + k] = 1;
              currentBatch.seq_id[batchIdx + k][0] = session.seqId;
              currentBatch.logits[batchIdx + k] = 0;
            }

            session.nPos += sendTokens;
            batchSeqIds.add(session.seqId);

            if (remaining == sendTokens) {
              currentBatch.logits[batchIdx + sendTokens - 1] = 1;
            }

            batchIdx += sendTokens;

            if (remaining == sendTokens) {
              session.pendingItems.removeFirst();
            } else {
              item.embdOffsetTokens += sendTokens;
            }

            didWork = true;
          } else {
            int filledCount = 0;

            while (filledCount < perSessionQuota &&
                session.pendingItems.isNotEmpty &&
                !session.pendingItems.first.isEmbedding &&
                batchIdx < _nBatch) {
              final tItem = session.pendingItems.removeFirst();

              currentBatch.token[batchIdx] = tItem.token!;
              currentBatch.pos[batchIdx] = session.nPos;
              currentBatch.n_seq_id[batchIdx] = 1;
              currentBatch.seq_id[batchIdx][0] = session.seqId;
              currentBatch.logits[batchIdx] = 0;

              session.nPos++;
              batchIdx++;
              filledCount++;
              batchSeqIds.add(session.seqId);

              if (session.pendingItems.isEmpty) {
                currentBatch.logits[batchIdx - 1] = 1;
              }
            }
            if (filledCount > 0) didWork = true;
          }

          _lastScheduledSessionIndex =
              sessionIndex[session] ?? _lastScheduledSessionIndex;
          if (batchIdx >= _nBatch) break;
        }

        if (batchIdx == 0) {
          continue;
        }

        currentBatch.n_tokens = batchIdx;

        _activeDecodeSync = Completer<void>();

        if (lib.llama_decode(_context, currentBatch) != 0) {
          _activeDecodeSync?.complete();
          _activeDecodeSync = null;

          for (final seqId in batchSeqIds) {
            final session = _sessionsBySeqId[seqId];
            if (session != null) {
              session.status = LlamaStatus.error;
              session.pendingItems.clear();
              if (!session.outputStream.isClosed) {
                session.outputStream.addError(LlamaException(
                    "llama_decode failed for session ${session.id}"));
                session.outputStream.close();
              }
            }
          }

          await Future.delayed(const Duration(milliseconds: 100));
          continue;
        }

        _activeDecodeSync?.complete();
        _activeDecodeSync = null;

        for (int i = 0; i < batchIdx; i++) {
          if (currentBatch.logits[i] != 0) {
            final sId = currentBatch.seq_id[i][0];
            final session = _sessionsBySeqId[sId];
            if (session == null) continue;

            final newToken =
                lib.llama_sampler_sample(session.sampler, _context, i);

            lib.llama_sampler_accept(session.sampler, newToken);

            bool isEos = lib.llama_token_is_eog(_sharedModel.vocab, newToken);
            String piece = "";
            if (isEos) {
              piece = " [EOS]";
              session.outputStream.add(piece);
            } else {
              piece = _decodeToken(session, newToken);
              if (piece.isNotEmpty) session.outputStream.add(piece);
            }

            session.nGenerated++;

            bool done = isEos;
            if (_contextNPredict != -1 &&
                session.nGenerated >= _contextNPredict) {
              done = true;
            }

            if (done) {
              session.status = LlamaStatus.ready;

              if (!session.outputStream.isClosed) {
                session.outputStream.close();
              }
            } else {
              session.pendingItems.addLast(_PendingItem.token(newToken));
            }
          }
        }
      }

      if (!didWork) {
        if (_workSignal.isCompleted) {
          _workSignal = Completer<void>();
          continue;
        }
        await _workSignal.future;
        _workSignal = Completer<void>();
      }
    }
  }

  Stopwatch _loopTimer = Stopwatch()..start();

  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    _stopSignal.complete();
    _notifyWork();

    if (_activeDecodeSync != null) {
      await _activeDecodeSync!.future
          .timeout(const Duration(seconds: 1), onTimeout: () {});
    }

    for (final s in _sessions.values) {
      s.dispose(lib);
    }
    _sessions.clear();

    if (_context != nullptr) lib.llama_free(_context);
    try {
      lib.llama_batch_free(_batch);
      if (_visionEnabled) {
        // FIX: Restore the hidden pointer before freeing
        if (_hiddenTokenPtr != nullptr) {
          _batchEmbd.token = _hiddenTokenPtr;
        }
        lib.llama_batch_free(_batchEmbd);
      }
    } catch (_) {}

    if (_visionEnabled && _mctx != nullptr) {
      lib.mtmd_free(_mctx);
    }

    _releaseModel(_sharedModel);

    // Ensure global backends are torn down to release Metal residency sets
    // before process exit, otherwise ggml asserts when the runtime unloads.
    if (_modelCache.isEmpty && _backendInitialized) {
      lib.llama_backend_free();
      _backendInitialized = false;
    }
  }

  // ... rest of private helpers (_checkDisposed, _ensureBackend, etc) same as before
  void _startRequest(_ServiceSession session) {
    // Reuse existing stream if open to prevent disconnecting listeners
    if (session._hasStream && !session.outputStream.isClosed) {
      // Keep using it
    } else {
      session.outputStream = StreamController();
      session._hasStream = true;
    }
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
    if (session.decodeCapacity == 0) session.decodeCapacity = 256;
    if (session.decodeBuf == nullptr) {
      session.decodeBuf = malloc<Char>(session.decodeCapacity);
    }
    var buf = session.decodeBuf;

    try {
      int n = lib.llama_token_to_piece(
          _sharedModel.vocab, tokenId, buf, session.decodeCapacity, 0, true);

      if (n < 0) {
        final needed = -n;
        final newCap = max(needed, session.decodeCapacity * 2);
        session.decodeCapacity = newCap;
        if (buf != nullptr) {
          malloc.free(buf);
        }
        session.decodeBuf = malloc<Char>(session.decodeCapacity);
        buf = session.decodeBuf;

        n = lib.llama_token_to_piece(
            _sharedModel.vocab, tokenId, buf, session.decodeCapacity, 0, true);
        if (n < 0) return "";
      }

      final newBytes = buf.cast<Uint8>().asTypedList(n);
      return session.accumulator.process(newBytes);
    } catch (_) {
      return "";
    }
  }
}

/// Helper to accumulate bytes and decode only when valid UTF-8 is available.
class _Utf8Accumulator {
  final List<int> _buffer = [];
  static const int _maxBufferLen = 8192;

  String process(List<int> newBytes) {
    _buffer.addAll(newBytes);
    if (_buffer.isEmpty) return "";

    try {
      final result = utf8.decode(_buffer);
      _buffer.clear();
      return result;
    } on FormatException {
      if (_buffer.length > _maxBufferLen) {
        final result = utf8.decode(_buffer, allowMalformed: true);
        _buffer.clear();
        return result;
      }
      return "";
    }
  }
}

class _PendingItem {
  final int? token;
  final Float32List? values; // for embeddings
  final int nTokens; // size in tokens
  int embdOffsetTokens;

  _PendingItem.token(this.token)
      : values = null,
        nTokens = 1,
        embdOffsetTokens = 0;
  _PendingItem.embedding(this.values, this.nTokens,
      {this.embdOffsetTokens = 0})
      : token = null;

  bool get isEmbedding => values != null;

  int get remainingEmbeddingTokens => nTokens - embdOffsetTokens;
}

class _ServiceSession {
  final String id;
  final int seqId;
  final Pointer<llama_sampler> sampler;

  LlamaStatus status = LlamaStatus.uninitialized;
  Pointer<Char> decodeBuf = nullptr;
  int decodeCapacity = 0;
  _Utf8Accumulator accumulator = _Utf8Accumulator();

  final ListQueue<_PendingItem> pendingItems = ListQueue<_PendingItem>();
  late StreamController<String> outputStream;
  bool _hasStream = false;

  int nPos = 0;
  int nPromptTokens = 0;
  int nGenerated = 0;

  _ServiceSession({
    required this.id,
    required this.seqId,
    required this.sampler,
  });

  void dispose(llama_cpp lib) {
    if (sampler != nullptr) {
      lib.llama_sampler_free(sampler);
    }
    if (decodeBuf != nullptr) {
      malloc.free(decodeBuf);
    }

    if (_hasStream && !outputStream.isClosed) {
      outputStream.close();
    }
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
