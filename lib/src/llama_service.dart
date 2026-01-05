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

class _SessionEvent {
  final int requestId;
  final String text;
  _SessionEvent(this.requestId, this.text);
}

/// Multi-user friendly wrapper around llama.cpp.
class LlamaService {
  static final Map<_ModelCacheKey, _SharedModelHandle> _modelCache = {};

  static llama_cpp get lib => Llama.lib;

  static bool _backendInitialized = false;
  static const int _maxTokensPerSessionPerBatch = 16;

  final String modelPath;
  final ModelParams modelParams;
  late final ContextParams defaultContextParams;
  final ContextParams _ctxConfig;
  final SamplerParams defaultSamplerParams;
  final bool _verbose;

  final Map<String, _ServiceSession> _sessions = {};

  final Map<int, _ServiceSession> _sessionsBySeqId = {};

  late final _SharedModelHandle _sharedModel;

  // Shared context and batch
  // Shared context and batch
  Pointer<llama_context> _context = nullptr;
  llama_batch? _batch;
  llama_batch? _batchEmbd;

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
  Future<void>? _loopFuture; // Track the loop

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
        defaultSamplerParams = samplerParams ?? SamplerParams(),
        _verbose = verbose {
    defaultContextParams = _ctxConfig;
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
      final b = lib.llama_batch_init(_nBatch, 0, _maxParallel);
      _batch = b;

      if (b.token.address == 0 || b.pos.address == 0 || b.seq_id.address == 0) {
        throw LlamaException("llama_batch_init failed to allocate memory");
      }

      // Initialize Embedding Batch if Vision is enabled
      if (_visionEnabled) {
        final nEmbd = lib.llama_n_embd(_sharedModel.model);
        // We allocate with _nBatch tokens to ensure 'pos', 'seq_id' arrays are allocated
        final eb = lib.llama_batch_init(_nBatch, nEmbd, _maxParallel);
        _batchEmbd = eb;

        // FIX: Hide the token pointer so llama_decode works, but don't free it
        _hiddenTokenPtr = eb.token;
        eb.token =
            nullptr; // This is safe because we saved it in _hiddenTokenPtr
      }
    } catch (e) {
      if (_context != nullptr) lib.llama_free(_context);

      // Cleanup _batch
      if (_batch != null) {
        // llama_batch_free handles the cleanup, assuming init succeeded enough to give pointers
        // But if init failed partially inside C (unlikely to return struct with nulls but possible),
        // we check token address if prudent, but llama_batch_free is usually robust if struct is valid.
        // The checking logic in original code was: if (.token.address != 0)
        final b = _batch!;
        if (b.token.address != 0) lib.llama_batch_free(b);
      }

      // Cleanup _batchEmbd
      if (_visionEnabled && _batchEmbd != null) {
        final eb = _batchEmbd!;
        // Restore hidden pointer if we hid it
        if (_hiddenTokenPtr != nullptr) {
          eb.token = _hiddenTokenPtr;
        }

        // Only free if we have a token pointer (meaning allocation happened)
        // Original issue: if eb.token was nullptr (because we set it), and _hiddenTokenPtr was null (init failed before hiding?),
        // calling free might be weird if library expects tokens.
        // But here we restored it.
        // We also check address != 0 to be sure it's not a zero-initialized struct.
        if (eb.token.address != 0 && eb.token != nullptr) {
          lib.llama_batch_free(eb);
        }
      }

      if (_visionEnabled && _mctx != nullptr) {
        lib.mtmd_free(_mctx);
      }

      _releaseModel(_sharedModel);
      rethrow;
    }

    _loopFuture = _runLoop();
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

    // We do NOT start request here automatically, wait for setPrompt
  }

  Future<void> cancel(String sessionId) async {
    final session = _sessions[sessionId];
    if (session == null) return;

    session.status = LlamaStatus.ready;
    session.pendingItems.clear();
    session._outputBuffer.clear();
    session._completeGeneration(session.requestId);
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

  Future<int> setPrompt(
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

    // Reset status from previous errors or ready
    session.status = LlamaStatus.generating;

    // Assign new request ID to isolate this request's stream events
    _startRequest(session);
    final currentRequestId = session.requestId;

    try {
      // _startRequest(session); // Moved up

      if (clearHistory) {
        final mem = lib.llama_get_memory(_context);
        lib.llama_memory_seq_rm(mem, session.seqId, -1, -1);
        session.nPos = 0;
        if (session.sampler != nullptr)
          lib.llama_sampler_reset(session.sampler);
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
    } catch (e, s) {
      session.status = LlamaStatus.error;
      session.pendingItems.clear();
      session._outputBuffer.clear();
      if (!session.controller.isClosed) {
        session.controller.addError(e, s);
      }
      session._completeGeneration(currentRequestId);
      rethrow;
    }
    return currentRequestId;
  }

  Future<String> generateCompleteText(String sessionId) async {
    final session = _requireSession(sessionId);
    final buf = StringBuffer();
    final reqId = session.requestId; // assumes request already started
    Object? error;
    StackTrace? stack;

    final sub = session.stream.listen(
      (event) {
        if (event.requestId == reqId) buf.write(event.text);
      },
      onError: (e, s) {
        error = e;
        stack = s;
      },
    );

    try {
      // reqId already captured above
      // Wait for specific request if possible, or just the latest future
      if (session.requestCompleters.containsKey(reqId)) {
        await session.requestCompleters[reqId]!.future;
      } else {
        // Fallback if no specific completer (unlikely if called after start)
        // or verify if we should just wait for next signal.
        // Actually, if we just came from _requireSession, we might not have started?
        // generateCompleteText assumes the request is running or about to run?
        // NO, generateCompleteText usually wraps setPrompt itself OR is called on a running session?
        // The legacy method didn't call setPrompt.
        // If called manually, we rely on streams.
        // Let's assume the user calls setPrompt separately?
        // Actually, this method waits for "session.generationDone.future" in old code.
        // We should probably wait for the *latest* request's completion.
        final latest = session.requestCompleters[session.requestId]?.future;
        if (latest != null) await latest;
      }

      if (error != null) {
        Error.throwWithStackTrace(error!, stack ?? StackTrace.empty);
      }
    } finally {
      await sub.cancel();
    }

    return buf.toString();
  }

  Stream<String> generateText(String sessionId, {String? prompt}) {
    _checkDisposed();
    final session = _requireSession(sessionId);

    // Legacy/Manual mode: return persistent stream
    if (prompt == null) {
      return session.stream.map((e) => e.text);
    }

    // Scoped mode: set prompt and return finite stream
    return _generateTextWithPrompt(sessionId, prompt);
  }

  Stream<String> _generateTextWithPrompt(
      String sessionId, String prompt) async* {
    final session = _requireSession(sessionId);
    final reqId = await setPrompt(sessionId, prompt);
    yield* _streamRequest(session, requestId: reqId).map((e) => e.text);
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

    // Reset status from previous errors or ready
    session.status = LlamaStatus.generating;

    _startRequest(session);
    final currentRequestId = session.requestId;

    try {
      if (clearHistory) {
        final mem = lib.llama_get_memory(_context);
        lib.llama_memory_seq_rm(mem, session.seqId, -1, -1);
        session.nPos = 0;
        if (session.sampler != nullptr) {
          lib.llama_sampler_reset(session.sampler);
        }
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

      yield* _streamRequest(session, requestId: currentRequestId)
          .map((e) => e.text);
    } catch (e, s) {
      session.status = LlamaStatus.error;
      session.pendingItems.clear();
      session._outputBuffer.clear();
      if (!session.controller.isClosed) {
        session.controller.addError(e, s);
      }
      session.lastError = e;
      session.lastStackTrace = s;
      session.lastErrorTime = DateTime.now();
      session._completeGeneration(currentRequestId);
      rethrow;
    }
    // The patch indicates adding a return here, but the method is async* (generator)
    // and already yields. Adding a direct return currentRequestId; would be a type mismatch.
    // Assuming the intent was to return the requestId if the method was not a generator,
    // or if it was meant to be a Future<int> instead of Stream<String>.
    // Given the current signature, no direct return is possible here.
    // The patch seems to be malformed or intended for a different method signature.
    // Keeping the original method signature and behavior.
  }

  Stream<_SessionEvent> _streamRequest(_ServiceSession session,
      {int? requestId}) {
    // Determine which request to listen to
    final targetReqId = requestId ?? session.requestId;

    // Check if that request has a completer
    final doneFuture =
        session.requestCompleters[targetReqId]?.future ?? Future.value();

    StreamSubscription<_SessionEvent>? sub;

    // Create a controller to forward events until done, with onCancel cleanup
    final controller = StreamController<_SessionEvent>(
      onCancel: () {
        sub?.cancel();
      },
    );

    sub = session.stream.listen(
      (data) {
        // Filter out events from other requests (race condition protection)
        if (data.requestId != targetReqId) return;
        if (!controller.isClosed) controller.add(data);
      },
      onError: (e, s) {
        if (!controller.isClosed) controller.addError(e, s);
      },
      onDone: () {
        // Persistent stream shouldn't close, but if it does:
        if (!controller.isClosed) controller.close();
      },
    );

    doneFuture.then((_) {
      sub?.cancel();
      if (!controller.isClosed) controller.close();
    });

    return controller.stream;
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
    try {
      while (!_disposed) {
        if (_stopSignal.isCompleted) break;

        if (_loopTimer.elapsedMilliseconds >= 16) {
          await Future.delayed(Duration.zero);
          _loopTimer.reset();
        }

        final activeSessions = _sessions.values.toList();

        // Wait for work if nothing to do
        if (activeSessions.isEmpty) {
          if (!_workSignal.isCompleted && !_stopSignal.isCompleted) {
            await Future.any([_workSignal.future, _stopSignal.future]);
          }
          // Reset signal only if we consumed it (it was completed)
          if (_workSignal.isCompleted) {
            _workSignal = Completer<void>();
          }
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
          final currentBatch = isEmbeddingPass ? _batchEmbd! : _batch!;
          currentBatch.n_tokens = 0;
          final Set<int> batchSeqIds = {};

          for (final session in rotatedSessions) {
            if (session.status != LlamaStatus.generating) continue;

            // Capture 'requestId' at scheduling to tag outputs and prevent zombie tokens
            final currentSessionRequestId = session.requestId;

            if (session.nPos >= _contextNCtx - 2) {
              session.status = LlamaStatus.ready;
              session.controller.add(
                  _SessionEvent(currentSessionRequestId, "\n[Context Limit]"));
              final mem = lib.llama_get_memory(_context);
              lib.llama_memory_seq_rm(mem, session.seqId, -1, -1);
              session.nPos = 0;
              session.pendingItems.clear();
              session._completeGeneration(currentSessionRequestId);
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
                // We don't have batchRequestId here easily, but we can assume 'session.requestId' is still relevant?
                // Or we should store it map. But simplified: use session.requestId as best effort for decode errors.
                // Actually, if a new request started mid-decode, completing the new request with old error is OK (fail fast).
                final rid = session.requestId;
                if (!session.controller.isClosed) {
                  session.controller.addError(LlamaException(
                      "llama_decode failed for session ${session.id}"));

                  session.lastError = LlamaException("llama_decode failed");
                  session.lastErrorTime = DateTime.now();
                  session._completeGeneration(rid);
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

              final rid = session.requestId;
              if (session.status != LlamaStatus.generating) {
                continue;
              }

              final newToken =
                  lib.llama_sampler_sample(session.sampler, _context, i);

              lib.llama_sampler_accept(session.sampler, newToken);

              bool isEos = lib.llama_token_is_eog(_sharedModel.vocab, newToken);
              String piece = "";
              if (isEos) {
                piece = " [EOS]";
                session._outputBuffer.write(piece);
              } else {
                piece = _decodeToken(session, newToken);
                if (piece.isNotEmpty) session._outputBuffer.write(piece);
              }

              // Flush check
              // Late cancellation check: don't buffer if cancelled
              if (session.status == LlamaStatus.generating) {
                session.flush(requestId: rid);
              }

              session.nGenerated++;

              bool done = isEos;
              if (_contextNPredict != -1 &&
                  session.nGenerated >= _contextNPredict) {
                done = true;
              }

              if (done) {
                session.status = LlamaStatus.ready;

                if (!session.controller.isClosed) {
                  session._completeGeneration(rid);
                }
              } else {
                session.pendingItems.addLast(_PendingItem.token(newToken));
              }
            }
          }
        }

        if (!didWork) {
          // No work found in this pass, wait for new work signal
          if (!_workSignal.isCompleted && !_stopSignal.isCompleted) {
            // We use Future.any to ensure we wake up on stop too
            await Future.any([_workSignal.future, _stopSignal.future]);
          }
          // Reset signal
          if (_workSignal.isCompleted) {
            _workSignal = Completer<void>();
          }
        }
      }
    } catch (e, stack) {
      // Unobserved error in run loop! Log it.
      // In a real app we might want to surface this or restart, but at least don't crash silently.
      // ignore: avoid_print
      print("Error in LlamaService run loop: $e\n$stack");
    }
  }

  Stopwatch _loopTimer = Stopwatch()..start();

  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    _stopSignal.complete();
    // No need to notify work, the loop wakes on stopSignal

    // Wait for loop to finish
    if (_loopFuture != null) {
      await _loopFuture!;
    }

    if (_activeDecodeSync != null) {
      await _activeDecodeSync!.future
          .timeout(const Duration(seconds: 1), onTimeout: () {});
    }

    for (final s in _sessions.values) {
      s.dispose(lib);
    }
    _sessions.clear();

    if (_context != nullptr) lib.llama_free(_context);

    if (_batch != null) {
      try {
        lib.llama_batch_free(_batch!);
      } catch (_) {}
    }

    if (_visionEnabled && _batchEmbd != null) {
      try {
        final eb = _batchEmbd!;
        // FIX: Restore the hidden pointer before freeing
        if (_hiddenTokenPtr != nullptr) {
          eb.token = _hiddenTokenPtr;
        }
        lib.llama_batch_free(eb);
      } catch (_) {}
    }

    if (_visionEnabled && _mctx != nullptr) {
      lib.mtmd_free(_mctx);
    }

    _releaseModel(_sharedModel);

    // Ensure global backends are torn down to release Metal residency sets
    // before process exit, otherwise ggml asserts when the runtime unloads.
    // Ensure global backends are torn down to release Metal residency sets
    // before process exit, otherwise ggml asserts when the runtime unloads.
    if (_modelCache.isEmpty && _backendInitialized) {
      lib.llama_backend_free();
      _backendInitialized = false;
    }
  }

  // ... rest of private helpers (_checkDisposed, _ensureBackend, etc) same as before
  void _startRequest(_ServiceSession session) {
    if (session.status != LlamaStatus.generating) return;

    // Reset accumulator for new request
    session.accumulator = _Utf8Accumulator();
    session._outputBuffer.clear();

    // Create a new completion future for this request
    session.requestId++;
    session.requestCompleters[session.requestId] = Completer<void>();
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

    final handle =
        _SharedModelHandle(model: loadedModel, vocab: vocab, key: key);
    _modelCache[key] = handle;
    return handle;
  }

  void _releaseModel(_SharedModelHandle handle) {
    handle.refs--;
    if (handle.refs > 0) return;

    // Optimized removal using the key stored in the handle
    if (handle.key != null) {
      _modelCache.remove(handle.key);
    } else {
      // Fallback or legacy (shouldn't happen with new handles)
      _modelCache.removeWhere((_, h) => identical(h, handle));
    }

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
  _PendingItem.embedding(this.values, this.nTokens, {this.embdOffsetTokens = 0})
      : token = null;

  bool get isEmbedding => values != null;

  int get remainingEmbeddingTokens => nTokens - embdOffsetTokens;
}

class _ServiceSession {
  final String id;
  final int seqId;
  final Pointer<llama_sampler> sampler;
  int requestId = 0; // Monotonically increasing request ID

  LlamaStatus status = LlamaStatus.uninitialized;

  // Last error info
  Object? lastError;
  StackTrace? lastStackTrace;
  DateTime? lastErrorTime;

  Pointer<Char> decodeBuf = nullptr;
  int decodeCapacity = 0;
  _Utf8Accumulator accumulator = _Utf8Accumulator();

  final ListQueue<_PendingItem> pendingItems = ListQueue<_PendingItem>();
  late final StreamController<_SessionEvent> controller;
  bool paused = false;

  // Map of requestID -> Completer
  final Map<int, Completer<void>> requestCompleters = {};

  // Legacy single completer shim if needed, or just remove it.
  // We keep the old field name but deprecate or remove usage if possible.
  // Actually, let's remove generationDone and use requestCompleters logic.

  int nPos = 0;
  int nPromptTokens = 0;
  int nGenerated = 0;

  _ServiceSession({
    required this.id,
    required this.seqId,
    required this.sampler,
  }) {
    // Persistent controller (Broadcast)
    // Note: Broadcast streams do not support onPause/onResume backpressure hooks by default.
    controller = StreamController<_SessionEvent>.broadcast();
  }

  Stream<_SessionEvent> get stream => controller.stream;

  // Backpressure buffering
  final StringBuffer _outputBuffer = StringBuffer();
  DateTime _lastFlushTime = DateTime.now();

  void flush({bool force = false, required int requestId}) {
    if (_outputBuffer.isEmpty) return;

    // Broadcast buffering policy:
    // If no listeners, we buffer up to a cap (e.g. 5MB).
    // If listeners exist, we emit.

    if (!controller.hasListener) {
      // Check cap
      if (_outputBuffer.length > 5 * 1024 * 1024) {
        // 5MB char count roughly
        // Cap exceeded.
        // We could clear, or error.
        _outputBuffer.clear();
        _outputBuffer.write(
            "[Error: Output buffer exceeded 5MB with no listeners. Stream dropped.]");
        // We might want to cancel the session too?
        // For now just dropping old data is "safe" enough to prevent OOM.
        return;
      }
      // Just return, keep buffering (replay logic)
      return;
    }

    final now = DateTime.now();
    // Flush if forced, buffer too large (>1KB), or time elapsed (>50ms)
    if (force ||
        _outputBuffer.length > 1024 ||
        now.difference(_lastFlushTime).inMilliseconds > 50) {
      if (!controller.isClosed) {
        controller.add(_SessionEvent(requestId, _outputBuffer.toString()));
      }
      _outputBuffer.clear();
      _lastFlushTime = now;
    }
  }

  void _completeGeneration(int requestId) {
    flush(force: true, requestId: requestId); // Ensure pending data is sent

    // Complete the completer for the CURRENT request ID
    if (requestCompleters.containsKey(requestId)) {
      final c = requestCompleters[requestId]!;
      if (!c.isCompleted) c.complete();
      // We can clean up old completers here or periodically
      requestCompleters.removeWhere((k, v) => k < requestId);
    }
  }

  void dispose(llama_cpp lib) {
    if (sampler != nullptr) {
      lib.llama_sampler_free(sampler);
    }
    if (decodeBuf != nullptr) {
      malloc.free(decodeBuf);
    }
    controller.close();
  }
}

class _SharedModelHandle {
  final Pointer<llama_model> model;
  final Pointer<llama_vocab> vocab;
  final _ModelCacheKey? key;
  int refs = 1;

  _SharedModelHandle({required this.model, required this.vocab, this.key});
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
