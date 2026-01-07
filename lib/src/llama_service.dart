import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' show max, min;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'context_params.dart';
import 'llama.dart'
    show Llama, LlamaException, LlamaLogCallback, LlamaStatus;
import 'llama_cpp.dart';
import 'llama_input.dart';
import 'model_params.dart';
import 'sampler_params.dart';

class _SessionEvent {
  final int requestId;
  final String text;
  _SessionEvent(this.requestId, this.text);
}

class TokenUsage {
  int promptTokens = 0;
  int completionTokens = 0;
  final DateTime startTime = DateTime.now();
  DateTime? endTime;

  int get total => promptTokens + completionTokens;

  double get tokensPerSecond {
    final end = endTime ?? DateTime.now();
    final duration = end.difference(startTime).inMilliseconds / 1000.0;
    if (duration <= 0) return 0.0;
    return completionTokens / duration;
  }
}

/// Multi-user friendly wrapper around llama.cpp.
///
/// Key properties:
/// - Multiple concurrent sessions share a single llama_context using seq_id.
/// - Request-scoped streaming: every output chunk is tagged with requestId.
/// - Avoid "zombie tokens": tokens emitted only while session is generating AND match requestId.
/// - Optional multimodal (vision) via mtmd_* APIs.
class LlamaService {
  static final Map<_ModelCacheKey, _SharedModelHandle> _modelCache = {};

  static llama_cpp get lib => Llama.lib;

  static bool _backendInitialized = false;

  // Per-batch fairness: cap tokens per session per scheduling pass
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

  // Shared context and batches
  Pointer<llama_context> _context = nullptr;
  llama_batch? _batchTokens;
  llama_batch? _batchEmbd;

  // Hide token pointer for embedding batch (needed for free)
  Pointer<llama_token> _hiddenTokenPtr = nullptr;

  int _nBatch = 0;
  int _contextNCtx = 0;
  int _contextNPredict = -1;
  int _lastScheduledSessionIndex = 0;

  // Multimodal (Vision)
  Pointer<mtmd_context> _mctx = nullptr;
  bool _visionEnabled = false;

  // Loop management
  bool _disposed = false;
  final _stopSignal = Completer<void>();
  Completer<void> _workSignal = Completer<void>();
  Completer<void>? _activeDecodeSync;
  Future<void>? _loopFuture;
  final Stopwatch _loopTimer = Stopwatch()..start();

  // Seq allocation
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
    if (!_verbose) {
      final nullCallbackPointer =
          Pointer.fromFunction<LlamaLogCallback>(Llama.llamaLogCallbackNull);
      lib.llama_log_set(nullCallbackPointer, nullptr);
    }
    defaultContextParams = _ctxConfig;

    _ensureBackend();
    _sharedModel = _acquireModel(modelPath, this.modelParams);

    final ptr = lib.llama_print_system_info();
    final sysInfo = ptr.cast<Utf8>().toDartString();
    // ignore: avoid_print
    print(sysInfo);

    // Vision (optional)
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

      // Token batch
      final tb = lib.llama_batch_init(_nBatch, 0, _maxParallel);
      _batchTokens = tb;
      if (tb.token.address == 0 ||
          tb.pos.address == 0 ||
          tb.seq_id.address == 0) {
        throw LlamaException(
            "llama_batch_init (token batch) failed to allocate memory");
      }

      // Embedding batch (vision)
      if (_visionEnabled) {
        final nEmbd = lib.llama_n_embd(_sharedModel.model);
        final eb = lib.llama_batch_init(_nBatch, nEmbd, _maxParallel);
        _batchEmbd = eb;

        // IMPORTANT: hide token pointer (so llama_decode treats this as embd batch),
        // but keep it to restore for llama_batch_free.
        _hiddenTokenPtr = eb.token;
        eb.token = nullptr;
      }
    } catch (e) {
      _cleanupOnConstructorFailure();
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
        "Max parallel sessions ($_maxParallel) reached. Cannot create session $sessionId.",
      );
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
  }

  Future<void> cancel(String sessionId) async {
    final session = _sessions[sessionId];
    if (session == null) return;

    // End the current request (if any) and return to ready.
    final rid = session.requestId;
    session.status = LlamaStatus.ready;
    session.pendingItems.clear();
    session._outputBuffer.clear();
    session._completeGeneration(rid);
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
        "Session $sessionId already generating. Finish or cancel before starting a new request.",
      );
    }

    session.pendingItems.clear();
    session.nGenerated = 0;
    session.nPromptTokens = 0;
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

      final promptUtf8 = prompt.toNativeUtf8();
      try {
        final nCtx = _contextNCtx;
        final length = promptUtf8.length;
        final addBos = clearHistory;

        int nTokens = -lib.llama_tokenize(
          _sharedModel.vocab,
          promptUtf8.cast(),
          length,
          nullptr,
          0,
          addBos,
          true,
        );
        if (nTokens < 0) nTokens = -nTokens;

        final tokenPtr = malloc<llama_token>(nTokens);
        try {
          nTokens = lib.llama_tokenize(
            _sharedModel.vocab,
            promptUtf8.cast(),
            length,
            tokenPtr,
            nTokens,
            addBos,
            true,
          );

          if (session.nPos + nTokens > nCtx) {
            throw LlamaException(
              "Prompt too long or context full ($nTokens tokens, pos ${session.nPos} > $nCtx)",
            );
          }

          for (int i = 0; i < nTokens; i++) {
            session.pendingItems.addLast(_PendingItem.token(tokenPtr[i]));
          }
          session.nPromptTokens = nTokens;
          session.usage.promptTokens = nTokens;
          if (clearHistory) {
            session.nKeep = nTokens;
          }
        } finally {
          malloc.free(tokenPtr);
        }
      } finally {
        malloc.free(promptUtf8);
      }

      _notifyWork();
      return currentRequestId;
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
  }

  /// Collects the full text for the currently-running request for this session.
  ///
  /// Expected use:
  ///   await setPrompt(...)
  ///   final text = await generateCompleteText(...)
  Future<String> generateCompleteText(String sessionId) async {
    _checkDisposed();
    final session = _requireSession(sessionId);

    if (session.status != LlamaStatus.generating) {
      throw LlamaException("Session not generating. Call setPrompt first.");
    }

    final currentReqId = session.requestId;
    final buffer = StringBuffer();
    final completer = Completer<String>();

    StreamSubscription<_SessionEvent>? sub;
    sub = session.stream.listen(
      (event) {
        if (event.requestId == currentReqId) buffer.write(event.text);
      },
      onError: (e, s) async {
        await sub?.cancel();
        if (!completer.isCompleted) completer.completeError(e, s);
      },
    );

    final requestCompleter = session.requestCompleters[currentReqId];
    if (requestCompleter == null) {
      await sub.cancel();
      completer.completeError(LlamaException("Request completer not found"));
      return completer.future;
    }

    requestCompleter.future.then((_) async {
      await sub?.cancel();
      if (!completer.isCompleted) completer.complete(buffer.toString());
    }).catchError((e, s) async {
      await sub?.cancel();
      if (!completer.isCompleted) completer.completeError(e, s);
    });

    return completer.future;
  }

  /// Streaming API.
  ///
  /// - If [prompt] is null: returns the persistent session stream (all outputs).
  /// - If [prompt] is provided: starts a request and returns a finite stream for that request only.
  Stream<String> generateText(String sessionId, {String? prompt}) {
    _checkDisposed();
    final session = _requireSession(sessionId);

    if (prompt == null) {
      return session.stream.map((e) => e.text);
    }
    return _generateTextWithPrompt(sessionId, prompt);
  }

  Stream<String> _generateTextWithPrompt(
      String sessionId, String prompt) async* {
    final session = _requireSession(sessionId);
    final reqId = await setPrompt(sessionId, prompt);
    yield* _streamRequest(session, requestId: reqId).map((e) => e.text);
  }

  /// Vision/multimodal streaming.
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
        "Session $sessionId already generating. Finish or cancel before starting a new request.",
      );
    }

    session.pendingItems.clear();
    session.nGenerated = 0;
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
            "Mismatch between <image> markers and provided inputs");
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
  }

  /// Returns a finite stream for a specific requestId.
  Stream<_SessionEvent> _streamRequest(
    _ServiceSession session, {
    required int requestId,
  }) {
    final targetReqId = requestId;
    final doneFuture =
        session.requestCompleters[targetReqId]?.future ?? Future.value();

    StreamSubscription<_SessionEvent>? sub;

    final controller = StreamController<_SessionEvent>(
      onCancel: () async {
        await sub?.cancel();
        // Don't reference controller here; no need to close it manually.
      },
    );

    sub = session.stream.listen(
      (data) {
        if (data.requestId != targetReqId) return;
        if (!controller.isClosed) controller.add(data);
      },
      onError: (e, s) {
        if (!controller.isClosed) controller.addError(e, s);
      },
      onDone: () async {
        if (!controller.isClosed) await controller.close();
      },
    );

    Future<void> cleanup() async {
      await sub?.cancel();
      if (!controller.isClosed) await controller.close();
    }

    doneFuture.then((_) => cleanup());/*.timeout(
          const Duration(seconds: 60),
          onTimeout: () => cleanup(),
        );*/

    return controller.stream;
  }

  void _notifyWork() {
    if (!_workSignal.isCompleted) _workSignal.complete();
  }

  // -------------------- Scheduling loop --------------------

  Future<void> _runLoop() async {
    try {
      while (!_disposed) {
        if (_stopSignal.isCompleted) break;

        // Cooperative yield
        if (_loopTimer.elapsedMilliseconds >= 16) {
          await Future.delayed(Duration.zero);
          _loopTimer.reset();
        }

        final activeSessions = _sessions.values.toList();

        if (activeSessions.isEmpty) {
          if (!_workSignal.isCompleted && !_stopSignal.isCompleted) {
            await Future.any([_workSignal.future, _stopSignal.future]);
          }
          if (_workSignal.isCompleted) _workSignal = Completer<void>();
          continue;
        }

        final Map<_ServiceSession, int> sessionIndex = {};
        for (var i = 0; i < activeSessions.length; i++) {
          sessionIndex[activeSessions[i]] = i;
        }

        // Round-robin rotation
        final int rotateStart =
            (_lastScheduledSessionIndex + 1) % activeSessions.length;
        final rotatedSessions = [
          ...activeSessions.sublist(rotateStart),
          ...activeSessions.sublist(0, rotateStart),
        ];

        final passes = _visionEnabled ? [false, true] : [false];
        bool didWork = false;

        // Track requestId used when scheduling each seq_id into the batch.
        final Map<int, int> batchRidBySeqId = {};

        for (final isEmbeddingPass in passes) {
          final currentBatch = isEmbeddingPass ? _batchEmbd : _batchTokens;
          if (currentBatch == null) continue;

          int batchIdx = 0;
          currentBatch.n_tokens = 0;
          final Set<int> batchSeqIds = {};

          // Fill batch
          for (final session in rotatedSessions) {
            if (session.status != LlamaStatus.generating) continue;

            final ridAtSchedule = session.requestId;

            // Context shift guard
            if (session.nPos >= _contextNCtx - 10) {
              final mem = lib.llama_get_memory(_context);
              final nLeft = session.nPos - session.nKeep;
              final nShift = nLeft ~/ 2;
              if (nShift > 0) {
                lib.llama_memory_seq_rm(
                  mem,
                  session.seqId,
                  session.nKeep,
                  session.nKeep + nShift,
                );
                lib.llama_memory_seq_add(
                  mem,
                  session.seqId,
                  session.nKeep + nShift,
                  -1,
                  -nShift,
                );
                session.nPos -= nShift;
                // ignore: avoid_print
                print("Context Shifted");
              }
            }

            final available = _nBatch - batchIdx;
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

            final int quota = min(available, _maxTokensPerSessionPerBatch);

            if (isEmbeddingPass) {
              final remaining = item.remainingEmbeddingTokens;
              if (remaining <= 0) {
                session.pendingItems.removeFirst();
                _lastScheduledSessionIndex =
                    sessionIndex[session] ?? _lastScheduledSessionIndex;
                continue;
              }

              final sendTokens = min(quota, remaining);
              final nEmbd = lib.llama_n_embd(_sharedModel.model);
              final start = item.embdOffsetTokens * nEmbd;
              final floatCount = sendTokens * nEmbd;

              // Copy embeddings into batch.embd
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
              batchRidBySeqId[session.seqId] = ridAtSchedule;

              if (remaining == sendTokens) {
                // request logits on final token of this item
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
              int filled = 0;

              while (filled < quota &&
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
                filled++;
                batchSeqIds.add(session.seqId);
                batchRidBySeqId[session.seqId] = ridAtSchedule;

                // If that was the last pending item for now, request logits
                if (session.pendingItems.isEmpty) {
                  currentBatch.logits[batchIdx - 1] = 1;
                }
              }

              if (filled > 0) didWork = true;
            }

            _lastScheduledSessionIndex =
                sessionIndex[session] ?? _lastScheduledSessionIndex;
            if (batchIdx >= _nBatch) break;
          }

          if (batchIdx == 0) continue;

          currentBatch.n_tokens = batchIdx;

          _activeDecodeSync = Completer<void>();

          final decodeRc = lib.llama_decode(_context, currentBatch);
          if (decodeRc != 0) {
            _activeDecodeSync?.complete();
            _activeDecodeSync = null;

            for (final seqId in batchSeqIds) {
              final s = _sessionsBySeqId[seqId];
              if (s == null) continue;

              s.status = LlamaStatus.error;
              s.pendingItems.clear();

              final rid = batchRidBySeqId[seqId] ?? s.requestId;

              if (!s.controller.isClosed) {
                s.controller.addError(
                    LlamaException("llama_decode failed for session ${s.id}"));
              }
              s.lastError = LlamaException("llama_decode failed");
              s.lastErrorTime = DateTime.now();
              s._completeGeneration(rid);
            }

            await Future.delayed(const Duration(milliseconds: 50));
            continue;
          }

          _activeDecodeSync?.complete();
          _activeDecodeSync = null;

          // Sample logits per token where currentBatch.logits[i] == 1
          for (int i = 0; i < batchIdx; i++) {
            if (currentBatch.logits[i] == 0) continue;

            final seqId = currentBatch.seq_id[i][0];
            final session = _sessionsBySeqId[seqId];
            if (session == null) continue;

            final rid = batchRidBySeqId[seqId] ?? session.requestId;

            // Zombie protection
            if (session.status != LlamaStatus.generating) continue;
            if (session.requestId != rid) continue;

            final newToken =
                lib.llama_sampler_sample(session.sampler, _context, i);
            lib.llama_sampler_accept(session.sampler, newToken);
            session.usage.completionTokens++;

            final isEos = lib.llama_token_is_eog(_sharedModel.vocab, newToken);

            if (isEos) {
              session._outputBuffer.write(" [EOS]");
            } else {
              final piece = _decodeToken(session, newToken);
              if (piece.isNotEmpty) session._outputBuffer.write(piece);
            }

            // Emit buffered output (or keep buffering if no listeners)
            session.flush(requestId: rid);

            session.nGenerated++;

            bool done = isEos;
            if (_contextNPredict != -1 &&
                session.nGenerated >= _contextNPredict) {
              done = true;
            }

            if (done) {
              session.status = LlamaStatus.ready;
              _printMetrics(session.id, session);
              session._completeGeneration(rid);
            } else {
              session.pendingItems.addLast(_PendingItem.token(newToken));
            }
          }
        }

        if (!didWork) {
          if (!_workSignal.isCompleted && !_stopSignal.isCompleted) {
            await Future.any([_workSignal.future, _stopSignal.future]);
          }
          if (_workSignal.isCompleted) _workSignal = Completer<void>();
        }
      }
    } catch (e, s) {
      // ignore: avoid_print
      print("Error in LlamaService run loop: $e\n$s");
    }
  }

  // -------------------- Disposal --------------------

  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;

    _stopSignal.complete();
    if (!_workSignal.isCompleted) _workSignal.complete();

    if (_loopFuture != null) {
      try {
        await _loopFuture!.timeout(const Duration(seconds: 2));
      } on TimeoutException {
        // ignore: avoid_print
        print("Warning: run loop did not exit cleanly");
      }
    }

    if (_activeDecodeSync != null && !_activeDecodeSync!.isCompleted) {
      _activeDecodeSync!.complete();
    }

    for (final s in List.of(_sessions.values)) {
      s.dispose(lib);
    }
    _sessions.clear();
    _sessionsBySeqId.clear();

    if (_context != nullptr) {
      lib.llama_free(_context);
      _context = nullptr;
    }

    if (_batchTokens != null) {
      try {
        lib.llama_batch_free(_batchTokens!);
      } catch (_) {}
      _batchTokens = null;
    }

    if (_visionEnabled && _batchEmbd != null) {
      try {
        final eb = _batchEmbd!;
        if (_hiddenTokenPtr != nullptr) {
          eb.token = _hiddenTokenPtr;
        }
        lib.llama_batch_free(eb);
      } catch (_) {}
      _batchEmbd = null;
    }

    if (_visionEnabled && _mctx != nullptr) {
      lib.mtmd_free(_mctx);
      _mctx = nullptr;
    }

    _releaseModel(_sharedModel);

    if (_modelCache.isEmpty && _backendInitialized) {
      lib.llama_backend_free();
      _backendInitialized = false;
    }
  }

  // -------------------- Internals --------------------

  void _startRequest(_ServiceSession session) {
    if (session.status != LlamaStatus.generating) return;

    session.accumulator = _Utf8Accumulator();
    session._outputBuffer.clear();

    session.requestId++;
    session.requestCompleters[session.requestId] = Completer<void>();
    session.usage = TokenUsage();

    // If someone subscribes after output started, we want to replay buffered content.
    session._bufferRequestId = session.requestId;
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

  void _printMetrics(String sessionId, _ServiceSession session) {
    session.usage.endTime = DateTime.now();
    final tps = session.usage.tokensPerSecond.toStringAsFixed(2);
    final usage = session.usage;

    final currentRss = ProcessInfo.currentRss / (1024 * 1024);
    final ctxBytes = lib.llama_get_state_size(_context);
    final ctxMb = ctxBytes / (1024 * 1024);

    // ASCII-only output to keep logs compatible with restricted consoles.
    print("""
[Metrics: $sessionId]
------------------------------------------
Speed:       $tps t/s
Tokens:      ${usage.promptTokens} prompt + ${usage.completionTokens} completion = ${usage.total} total
Context Mem: ${ctxMb.toStringAsFixed(1)} MB (KV Cache)
App RAM:     ${currentRss.toStringAsFixed(1)} MB (RSS)
------------------------------------------
""");
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
    final sparams = lib.llama_sampler_chain_default_params();
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
        _sharedModel.vocab,
        grammarStrPtr,
        grammarRootPtr,
      );
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
      ),
    );

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
        _SharedModelHandle(model: loadedModel, vocab: vocab, key: key);
    _modelCache[key] = handle;
    return handle;
  }

  void _releaseModel(_SharedModelHandle handle) {
    handle.refs--;
    if (handle.refs > 0) return;

    if (handle.key != null) {
      _modelCache.remove(handle.key);
    } else {
      _modelCache.removeWhere((_, h) => identical(h, handle));
    }

    if (handle.model.address != 0) {
      lib.llama_free_model(handle.model);
    }
  }

  String _decodeToken(_ServiceSession session, int tokenId) {
    const int minCapacity = 256;

    if (session.decodeCapacity < minCapacity || session.decodeBuf == nullptr) {
      if (session.decodeBuf != nullptr) {
        malloc.free(session.decodeBuf);
      }
      session.decodeCapacity = minCapacity;
      session.decodeBuf = malloc<Char>(session.decodeCapacity);
    }

    var buf = session.decodeBuf;

    int n = lib.llama_token_to_piece(
      _sharedModel.vocab,
      tokenId,
      buf,
      session.decodeCapacity,
      0,
      true,
    );

    if (n < 0) {
      final needed = -n;
      final newCapacity = max(needed, session.decodeCapacity * 2);

      final newBuf = malloc<Char>(newCapacity);
      malloc.free(session.decodeBuf);

      session.decodeBuf = newBuf;
      session.decodeCapacity = newCapacity;
      buf = newBuf;

      n = lib.llama_token_to_piece(
        _sharedModel.vocab,
        tokenId,
        buf,
        session.decodeCapacity,
        0,
        true,
      );

      if (n < 0) {
        // Give up; reset decode buffer (will be reallocated next time).
        malloc.free(session.decodeBuf);
        session.decodeBuf = nullptr;
        session.decodeCapacity = 0;
        return "";
      }
    }

    try {
      final newBytes = buf.cast<Uint8>().asTypedList(n);
      return session.accumulator.process(newBytes);
    } catch (_) {
      return "";
    }
  }

  void _cleanupOnConstructorFailure() {
    if (_context != nullptr) {
      lib.llama_free(_context);
      _context = nullptr;
    }

    if (_batchTokens != null) {
      final b = _batchTokens!;
      try {
        if (b.token.address != 0) lib.llama_batch_free(b);
      } catch (_) {}
      _batchTokens = null;
    }

    if (_visionEnabled && _batchEmbd != null) {
      final eb = _batchEmbd!;
      try {
        if (_hiddenTokenPtr != nullptr) eb.token = _hiddenTokenPtr;
        if (eb.token != nullptr && eb.token.address != 0) {
          lib.llama_batch_free(eb);
        }
      } catch (_) {}
      _batchEmbd = null;
    }

    if (_visionEnabled && _mctx != nullptr) {
      try {
        lib.mtmd_free(_mctx);
      } catch (_) {}
      _mctx = nullptr;
    }

    _releaseModel(_sharedModel);
  }

  void saveSession(String sessionId, String path) {}
  bool loadSession(String sessionId, String path) => false;
}

/// Accumulates UTF-8 bytes until they decode cleanly.
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
  final Float32List? values; // embeddings
  final int nTokens;
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

  int requestId = 0;
  int _bufferRequestId = 0;

  LlamaStatus status = LlamaStatus.uninitialized;

  Object? lastError;
  StackTrace? lastStackTrace;
  DateTime? lastErrorTime;

  Pointer<Char> decodeBuf = nullptr;
  int decodeCapacity = 0;
  _Utf8Accumulator accumulator = _Utf8Accumulator();

  final ListQueue<_PendingItem> pendingItems = ListQueue<_PendingItem>();

  late final StreamController<_SessionEvent> controller;

  // requestId -> completion
  final Map<int, Completer<void>> requestCompleters = {};

  int nPos = 0;
  int nPromptTokens = 0;
  int nGenerated = 0;
  int nKeep = 0; // Number of tokens to preserve (System Prompt)
  TokenUsage usage = TokenUsage();

  // Output buffering
  final StringBuffer _outputBuffer = StringBuffer();
  DateTime _lastFlushTime = DateTime.now();

  bool _isDisposed = false;

  _ServiceSession({
    required this.id,
    required this.seqId,
    required this.sampler,
  }) {
    // Broadcast stream with replay of buffered output when a listener attaches.
    controller = StreamController<_SessionEvent>.broadcast(
      onListen: () {
        // If we have buffered output for the last request, flush it now.
        if (_outputBuffer.isNotEmpty) {
          flush(force: true, requestId: _bufferRequestId);
        }
      },
    );
  }

  Stream<_SessionEvent> get stream => controller.stream;

  void flush({bool force = false, required int requestId}) {
    if (_outputBuffer.isEmpty) return;

    // If no listeners, keep buffering (bounded).
    if (!controller.hasListener) {
      _bufferRequestId = requestId;
      if (_outputBuffer.length > 5 * 1024 * 1024) {
        _outputBuffer.clear();
        _outputBuffer.write(
          "[Error: Output buffer exceeded 5MB with no listeners. Stream dropped.]",
        );
      }
      return;
    }

    final now = DateTime.now();
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
    // Ensure pending buffer is emitted (or stays buffered for replay).
    flush(force: true, requestId: requestId);

    final c = requestCompleters[requestId];
    if (c != null && !c.isCompleted) c.complete();

    // Drop old completers to avoid leaks.
    requestCompleters.removeWhere((k, _) => k < requestId);
  }

  void dispose(llama_cpp lib) {
    if (_isDisposed) return;
    _isDisposed = true;

    if (sampler != nullptr) {
      lib.llama_sampler_free(sampler);
    }
    if (decodeBuf != nullptr) {
      malloc.free(decodeBuf);
      decodeBuf = nullptr;
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
  bool operator ==(Object other) =>
      other is _ModelCacheKey &&
      other.path == path &&
      other.paramsSignature == paramsSignature;

  @override
  int get hashCode => Object.hash(path, paramsSignature);
}
