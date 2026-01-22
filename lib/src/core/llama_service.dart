import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' show max, min;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'context_params.dart';
import 'llama.dart' show Llama, LlamaException, LlamaLogCallback, LlamaStatus;
import 'llama_cpp.dart';
import 'llama_input.dart';
import 'model_params.dart';
import 'sampler_params.dart';
import 'service/model_cache.dart';
import 'service/pending_item.dart';
import 'service/sampler_factory.dart';
import 'service/session.dart';
import 'service/state_codec.dart';
import 'service/utf8_accumulator.dart';
import 'service/vision_helper.dart';

/// Multi-user friendly wrapper around llama.cpp.
class LlamaService {
  static llama_cpp get lib => Llama.lib;

  static bool _backendInitialized = false;

  // Per-batch fairness: cap tokens per session per scheduling pass
  static const int _maxTokensPerSessionPerBatch = 64;

  final String modelPath;
  final ModelParams modelParams;
  late final ContextParams defaultContextParams;
  final ContextParams _ctxConfig;
  final SamplerParams defaultSamplerParams;
  final bool _verbose;
  final String sessionHome;
  final int maxSystemRamMb;

  final Map<String, ServiceSession> _sessions = {};
  final Map<int, ServiceSession> _sessionsBySeqId = {};

  late final SharedModelHandle _sharedModel;

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
  final Stopwatch _memoryJanitorTimer = Stopwatch()..start();

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
    String? sessionHome,
    int? maxSystemRamMb,
  })  : modelParams = modelParams ?? ModelParams(),
        _ctxConfig = contextParams ?? ContextParams(),
        defaultSamplerParams = samplerParams ?? SamplerParams(),
        _verbose = verbose,
        sessionHome = sessionHome ?? './sessions',
        maxSystemRamMb = maxSystemRamMb ?? 8192 {
    Directory(this.sessionHome).createSync(recursive: true);
    if (!_verbose) {
      final nullCallbackPointer =
          Pointer.fromFunction<LlamaLogCallback>(Llama.llamaLogCallbackNull);
      lib.llama_log_set(nullCallbackPointer, nullptr);
    }
    defaultContextParams = _ctxConfig;

    _ensureBackend();
    _sharedModel = ModelCache.acquire(
      lib: lib,
      path: modelPath,
      params: this.modelParams,
    );

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

    final sampler = SamplerFactory.build(
      lib: lib,
      vocab: _sharedModel.vocab,
      params: samplerParams ?? defaultSamplerParams,
    );

    final session = ServiceSession(
      id: sessionId,
      seqId: null,
      sampler: sampler,
    );

    final savedPath = '$sessionHome/$sessionId.state';
    final savedFile = File(savedPath);
    if (savedFile.existsSync()) {
      session.tier = SessionTier.cold;
      session.coldFilePath = savedPath;
      // ignore: avoid_print
      print("Found persisted session $sessionId at $savedPath");
    } else {
      session.tier = SessionTier.warm;
    }

    session.decodeCapacity = 256;
    session.decodeBuf = malloc<Char>(session.decodeCapacity);

    _sessions[sessionId] = session;
    session.status = LlamaStatus.ready;
  }

  Future<void> cancel(String sessionId) async {
    final session = _sessions[sessionId];
    if (session == null) return;

    // End the current request (if any) and return to ready.
    final rid = session.requestId;
    session.status = LlamaStatus.ready;
    for (var item in session.pendingItems) {
      item.dispose();
    }
    session.pendingItems.clear();
    session.outputBuffer.clear();
    session.completeGeneration(rid);
  }

  Future<void> freeSession(
    String sessionId, {
    bool deleteDiskFile = false,
  }) async {
    final session = _sessions.remove(sessionId);
    if (session == null) return;

    for (var item in session.pendingItems) {
      item.dispose();
    }
    session.pendingItems.clear();

    if (session.seqId != null) {
      _sessionsBySeqId.remove(session.seqId);
    }

    if (_context != nullptr) {
      final seqId = session.seqId;
      if (seqId != null) {
        final mem = lib.llama_get_memory(_context);
        lib.llama_memory_seq_rm(mem, seqId, -1, -1);
      }
    }

    session.dispose(lib);
    final seqId = session.seqId;
    if (seqId != null) {
      _freeSeqIds.add(seqId);
    }

    if (deleteDiskFile && session.coldFilePath != null) {
      final file = File(session.coldFilePath!);
      if (await file.exists()) {
        await file.delete();
      }
    }
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

    session.lastActiveTime = DateTime.now();
    if (session.tier != SessionTier.hot) {
      // ignore: avoid_print
      print("Restoring session $sessionId from ${session.tier}...");
      await _restoreSession(session);
      if (session.status != LlamaStatus.generating) {
        throw LlamaException("Request cancelled during session restore.");
      }
    }

    _startRequest(session);
    final currentRequestId = session.requestId;

    try {
      if (clearHistory) {
        final seqId = session.seqId;
        if (seqId == null) {
          throw LlamaException("Session $sessionId is not hot");
        }
        final mem = lib.llama_get_memory(_context);
        lib.llama_memory_seq_rm(mem, seqId, -1, -1);
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
            session.pendingItems.addLast(PendingItem.token(tokenPtr[i]));
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
      session.outputBuffer.clear();
      if (!session.controller.isClosed) {
        session.controller.addError(e, s);
      }
      session.completeGeneration(currentRequestId);
      rethrow;
    }
  }

  /// Collects the full text for the currently-running request for this session.
  Future<String> generateCompleteText(String sessionId) async {
    _checkDisposed();
    final session = _requireSession(sessionId);

    if (session.status != LlamaStatus.generating) {
      throw LlamaException("Session not generating. Call setPrompt first.");
    }

    final currentReqId = session.requestId;
    final buffer = StringBuffer();
    final completer = Completer<String>();

    StreamSubscription<SessionEvent>? sub;
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
  }) {
    // FIX: Race Condition Check
    // We must perform validation and state transitions SYNCHRONOUSLY.
    // If we use async*, the function body doesn't run until the first listener
    // attaches and the microtask queue flushes. By that time, the Isolate
    // might have already checked 'status' and found it still 'Ready'.

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

    // Set status immediately to block concurrent saves/requests
    session.pendingItems.clear();
    session.nGenerated = 0;
    session.status = LlamaStatus.generating;

    // Delegate to the async generator
    return _generateWithMediaBody(session, prompt, inputs, clearHistory);
  }

  Stream<String> _generateWithMediaBody(
    ServiceSession session,
    String prompt,
    List<LlamaInput> inputs,
    bool clearHistory,
  ) async* {
    session.lastActiveTime = DateTime.now();
    if (session.tier != SessionTier.hot) {
      // ignore: avoid_print
      print("Restoring session ${session.id} from ${session.tier}...");
      await _restoreSession(session);
      if (session.status != LlamaStatus.generating) {
        throw LlamaException("Request cancelled during session restore.");
      }
    }

    _startRequest(session);
    final currentRequestId = session.requestId;

    try {
      if (clearHistory) {
        final seqId = session.seqId;
        if (seqId == null) {
          throw LlamaException("Session ${session.id} is not hot");
        }
        final mem = lib.llama_get_memory(_context);
        lib.llama_memory_seq_rm(mem, seqId, -1, -1);
        session.nPos = 0;
        if (session.sampler != nullptr) {
          lib.llama_sampler_reset(session.sampler);
        }
      }

      final result = VisionHelper.buildPendingItems(
        lib: lib,
        mctx: _mctx,
        model: _sharedModel.model,
        prompt: prompt,
        inputs: inputs,
      );

      session.pendingItems.addAll(result.items);

      session.nPromptTokens = result.totalPromptTokens;
      session.usage.promptTokens = result.totalPromptTokens;
      if (clearHistory) {
        session.nKeep = result.totalPromptTokens;
      }

      _notifyWork();

      yield* _streamRequest(session, requestId: currentRequestId)
          .map((e) => e.text);
    } catch (e, s) {
      session.status = LlamaStatus.error;
      session.pendingItems.clear();
      session.outputBuffer.clear();
      if (!session.controller.isClosed) {
        session.controller.addError(e, s);
      }
      session.lastError = e;
      session.lastStackTrace = s;
      session.lastErrorTime = DateTime.now();
      session.completeGeneration(currentRequestId);
      rethrow;
    }
  }

  /// Returns a finite stream for a specific requestId.
  Stream<SessionEvent> _streamRequest(
    ServiceSession session, {
    required int requestId,
  }) {
    final targetReqId = requestId;
    // We must grab the completer synchronously
    final completer = session.requestCompleters[targetReqId];
    final doneFuture = completer?.future ?? Future.value();

    StreamSubscription<SessionEvent>? sub;
    late final StreamController<SessionEvent> controller;

    controller = StreamController<SessionEvent>(
      onListen: () {
        sub = session.stream
            .where((event) => event.requestId == targetReqId)
            .listen(
          (event) {
            if (!controller.isClosed) controller.add(event);
          },
          onError: (e, s) {
            if (!controller.isClosed) controller.addError(e, s);
          },
        );

        // When the request completes, close this controller.
        doneFuture.then((_) async {
          await sub?.cancel();
          if (!controller.isClosed) await controller.close();
        });
      },
      onCancel: () async {
        await sub?.cancel();
      },
    );

    return controller.stream;
  }

  void _notifyWork() {
    if (!_workSignal.isCompleted) _workSignal.complete();
  }

  // -------------------- Scheduling loop --------------------

  Future<void> _runLoop() async {
    final sliceTimer = Stopwatch()..start();
    try {
      while (!_disposed) {
        if (_stopSignal.isCompleted) break;

        // Cooperative yield (time-sliced)
        if (sliceTimer.elapsedMilliseconds > 10) {
          await Future.delayed(Duration.zero);
          sliceTimer.reset();
        }

        if (_memoryJanitorTimer.elapsed.inSeconds >= 5) {
          _memoryJanitorTimer.reset();

          final currentRssMb = ProcessInfo.currentRss / (1024 * 1024);
          if (currentRssMb > maxSystemRamMb) {
            if (_verbose) {
              // ignore: avoid_print
              print(
                "RAM Pressure: ${currentRssMb.round()}MB > ${maxSystemRamMb}MB. Archiving warm sessions...",
              );
            }

            final candidates = _sessions.values
                .where((s) => s.tier == SessionTier.warm)
                .toList()
              ..sort((a, b) => a.lastActiveTime.compareTo(b.lastActiveTime));

            for (final session in candidates) {
              await _archiveSession(session);

              final newRss = ProcessInfo.currentRss / (1024 * 1024);
              if (newRss <= maxSystemRamMb) break;
            }
          }
        }

        final activeSessions = _sessions.values.toList();

        if (activeSessions.isEmpty) {
          if (!_workSignal.isCompleted && !_stopSignal.isCompleted) {
            await Future.any([_workSignal.future, _stopSignal.future]);
          }
          if (_workSignal.isCompleted) _workSignal = Completer<void>();
          continue;
        }

        final Map<ServiceSession, int> sessionIndex = {};
        for (var i = 0; i < activeSessions.length; i++) {
          sessionIndex[activeSessions[i]] = i;
        }

        // Round-robin rotation
        final int rotateStart =
            (_lastScheduledSessionIndex + 1) % activeSessions.length;
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
          for (int i = 0; i < activeSessions.length; i++) {
            final index = (rotateStart + i) % activeSessions.length;
            final session = activeSessions[index];
            if (session.status != LlamaStatus.generating) continue;

            final ridAtSchedule = session.requestId;

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

            final seqId = session.seqId;
            if (seqId == null) {
              session.status = LlamaStatus.error;
              if (!session.controller.isClosed) {
                session.controller.addError(
                  LlamaException("Session ${session.id} is not hot"),
                );
              }
              session.completeGeneration(ridAtSchedule);
              continue;
            }

            // Context shift guard
            if (session.nPos >= _contextNCtx - 10) {
              final mem = lib.llama_get_memory(_context);
              final nLeft = session.nPos - session.nKeep;
              final nShift = nLeft ~/ 2;
              if (nShift > 0) {
                lib.llama_memory_seq_rm(
                  mem,
                  seqId,
                  session.nKeep,
                  session.nKeep + nShift,
                );
                lib.llama_memory_seq_add(
                  mem,
                  seqId,
                  session.nKeep + nShift,
                  -1,
                  -nShift,
                );
                session.nPos -= nShift;
                // ignore: avoid_print
                print("Context Shifted");
              }
            }

            final int sessionCap =
                session.pendingItems.length > 1 ? _nBatch : _maxTokensPerSessionPerBatch;
            final int quota = min(available, sessionCap);

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

              // FIX: Copy from Native Store -> Batch
              // item.nativeValues is our malloc'd pointer
              final srcPtr = item.nativeValues! + start;
              final destPtr = currentBatch.embd + (batchIdx * nEmbd);

              // Low-level memory copy (fastest)
              final srcList = srcPtr.asTypedList(floatCount);
              final destList = destPtr.asTypedList(floatCount);
              destList.setAll(0, srcList); // Copies raw bytes directly

              for (int k = 0; k < sendTokens; k++) {
                currentBatch.pos[batchIdx + k] = session.nPos + k;
                currentBatch.n_seq_id[batchIdx + k] = 1;
                currentBatch.seq_id[batchIdx + k][0] = seqId;
                currentBatch.logits[batchIdx + k] = 0;
              }

              session.nPos += sendTokens;
              batchSeqIds.add(seqId);
              batchRidBySeqId[seqId] = ridAtSchedule;

              if (remaining == sendTokens) {
                final finishedItem = session.pendingItems.removeFirst();

                // IMPORTANT: Free the native memory now!
                finishedItem.dispose();

                // FIX: Order of generation
                // Only request logits if this is the last pending item (text included)
                if (session.pendingItems.isEmpty) {
                  currentBatch.logits[batchIdx + sendTokens - 1] = 1;
                }
              } else {
                item.embdOffsetTokens += sendTokens;
              }

              batchIdx += sendTokens;
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
                currentBatch.seq_id[batchIdx][0] = seqId;
                currentBatch.logits[batchIdx] = 0;

                session.nPos++;
                batchIdx++;
                filled++;
                batchSeqIds.add(seqId);
                batchRidBySeqId[seqId] = ridAtSchedule;

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
              s.completeGeneration(rid);
            }

            await Future.delayed(const Duration(milliseconds: 50));
            continue;
          }

          _activeDecodeSync?.complete();
          _activeDecodeSync = null;

          if (!isEmbeddingPass) {
            for (int i = 0; i < batchIdx; i++) {
              final seqId = currentBatch.seq_id[i][0];
              final session = _sessionsBySeqId[seqId];
              if (session != null) {
                lib.llama_sampler_accept(
                  session.sampler,
                  currentBatch.token[i],
                );
              }
            }
          }

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
            session.usage.completionTokens++;

            final isEos = lib.llama_token_is_eog(_sharedModel.vocab, newToken);

            if (isEos) {
              session.outputBuffer.write(" [EOS]");
            } else {
              final piece = _decodeToken(session, newToken);
              if (piece.isNotEmpty) session.outputBuffer.write(piece);
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
              session.completeGeneration(rid);
            } else {
              session.pendingItems.addLast(PendingItem.token(newToken));
            }
          }
        }

        if (!didWork) {
          if (!_workSignal.isCompleted && !_stopSignal.isCompleted) {
            await Future.any([_workSignal.future, _stopSignal.future]);
          }
          sliceTimer.reset();
          if (_workSignal.isCompleted) _workSignal = Completer<void>();
        }
      }
    } catch (e, s) {
      // ignore: avoid_print
      print("Error in LlamaService run loop: $e\n$s");
    }
  }

  // -------------------- Disposal --------------------

  /// Gracefully shuts down the service.
  /// 1. Stops the generation loop.
  /// 2. Moves all active VRAM sessions to RAM (evict).
  /// 3. Writes all RAM sessions to Disk (.state files).
  /// 4. Frees all C resources.
  Future<void> persistAllAndDispose() async {
    if (_disposed) return;

    // ignore: avoid_print
    print("Initiating graceful shutdown...");

    if (!_stopSignal.isCompleted) _stopSignal.complete();

    await Future.delayed(const Duration(milliseconds: 50));

    for (final session in _sessions.values) {
      if (session.tier == SessionTier.hot) {
        try {
          _evictSession(session);
        } catch (e) {
          // ignore: avoid_print
          print("Error evicting session ${session.id}: $e");
        }
      }
    }

    final saveFutures = <Future<void>>[];
    for (final session in _sessions.values) {
      if (session.tier == SessionTier.warm) {
        saveFutures.add(_archiveSession(session).catchError((e) {
          // ignore: avoid_print
          print("Error saving session ${session.id} to disk: $e");
        }));
      }
    }

    if (saveFutures.isNotEmpty) {
      // ignore: avoid_print
      print("Persisting ${saveFutures.length} sessions to disk...");
      await Future.wait(saveFutures);
    }

    // ignore: avoid_print
    print("All states saved. Disposing resources.");

    await dispose();
  }

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

    ModelCache.release(lib: lib, handle: _sharedModel);

    if (ModelCache.isEmpty && _backendInitialized) {
      lib.llama_backend_free();
      _backendInitialized = false;
    }
  }

  // -------------------- Internals --------------------

  void _startRequest(ServiceSession session) {
    if (session.status != LlamaStatus.generating) return;

    session.accumulator = Utf8Accumulator();
    session.outputBuffer.clear();

    session.requestId++;
    session.requestCompleters[session.requestId] = Completer<void>();
    session.usage = TokenUsage();

    // If someone subscribes after output started, we want to replay buffered content.
    session.bufferRequestId = session.requestId;
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

  void _evictSession(ServiceSession session) {
    final seqId = session.seqId;
    if (session.tier != SessionTier.hot || seqId == null) return;

    final size = lib.llama_state_seq_get_size(_context, seqId);
    if (size > 0) {
      final dataPtr = malloc<Uint8>(size);
      try {
        lib.llama_state_seq_get_data(_context, dataPtr, size, seqId);
        final snapshot = dataPtr.asTypedList(size);
        session.stateBuffer = Uint8List.fromList(snapshot);
      } finally {
        malloc.free(dataPtr);
      }
    } else {
      session.stateBuffer = Uint8List(0);
    }

    final mem = lib.llama_get_memory(_context);
    lib.llama_memory_seq_rm(mem, seqId, -1, -1);

    _sessionsBySeqId.remove(seqId);
    _freeSeqIds.add(seqId);

    session.seqId = null;
    session.tier = SessionTier.warm;
  }

  Future<void> _archiveSession(ServiceSession session) async {
    if (session.stateBuffer == null || session.stateBuffer!.isEmpty) return;

    final path = '$sessionHome/${session.id}.state';
    final file = File(path);
    if (_verbose) {
      // ignore: avoid_print
      print("Archiving ${session.id} to disk ($path)...");
    }
    final wrapped = StateCodec.encode(
      session.stateBuffer!,
      nPos: session.nPos,
      nKeep: session.nKeep,
    );
    await file.writeAsBytes(wrapped, flush: true);

    session.coldFilePath = path;
    session.stateBuffer = null;
    session.tier = SessionTier.cold;
  }

  Future<void> _restoreSession(ServiceSession session) async {
    if (session.tier == SessionTier.hot && session.seqId != null) return;

    if (_freeSeqIds.isEmpty) {
      final candidates = _sessions.values
          .where((s) =>
              s.tier == SessionTier.hot && s.status != LlamaStatus.generating)
          .toList()
        ..sort((a, b) => a.lastActiveTime.compareTo(b.lastActiveTime));

      if (candidates.isEmpty) {
        throw LlamaException("No available session slots to restore");
      }
      _evictSession(candidates.first);
    }

    final newSeqId = _freeSeqIds.removeLast();

    Uint8List? data;
    if (session.tier == SessionTier.cold) {
      final path = session.coldFilePath;
      if (path == null) {
        throw LlamaException("Session ${session.id} has no archive path");
      }
      final bytes = await File(path).readAsBytes();
      final decoded = StateCodec.decode(bytes);
      data = decoded.payload;
      if (decoded.hasHeader) {
        session.nPos = decoded.nPos ?? session.nPos;
        session.nKeep = decoded.nKeep ?? session.nKeep;
      }
    } else {
      data = session.stateBuffer;
    }

    if (data == null) {
      if (session.tier == SessionTier.warm) {
        final mem = lib.llama_get_memory(_context);
        lib.llama_memory_seq_rm(mem, newSeqId, -1, -1);
        session.seqId = newSeqId;
        session.tier = SessionTier.hot;
        _sessionsBySeqId[newSeqId] = session;
        return;
      }
      throw LlamaException("Session ${session.id} has no state to restore");
    }

    final size = data.length;
    final dataPtr = malloc<Uint8>(size);
    try {
      dataPtr.asTypedList(size).setAll(0, data);
      lib.llama_state_seq_set_data(_context, dataPtr, size, newSeqId);
    } finally {
      malloc.free(dataPtr);
    }

    session.seqId = newSeqId;
    session.tier = SessionTier.hot;
    session.stateBuffer = null;
    _sessionsBySeqId[newSeqId] = session;
  }

  void _printMetrics(String sessionId, ServiceSession session) {
    session.usage.endTime = DateTime.now();
    final tps = session.usage.tokensPerSecond.toStringAsFixed(2);
    final usage = session.usage;

    final currentRss = ProcessInfo.currentRss / (1024 * 1024);
    final ctxBytes = lib.llama_get_state_size(_context);
    final ctxMb = ctxBytes / (1024 * 1024);

    // ASCII-only output to keep logs compatible with restricted consoles.
    // ignore: avoid_print
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

  ServiceSession _requireSession(String id) {
    final session = _sessions[id];
    if (session == null) {
      throw ArgumentError(
          "Session $id does not exist. Call createSession first.");
    }
    return session;
  }

  String _decodeToken(ServiceSession session, int tokenId) {
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

    ModelCache.release(lib: lib, handle: _sharedModel);
  }

  /// Manually saves the current state of a session to a specific file path.
  Future<void> saveSession(String sessionId, String path) async {
    _checkDisposed();
    final session = _requireSession(sessionId);

    if (session.status == LlamaStatus.generating) {
      throw LlamaException("Cannot save session while generating");
    }

    // Case 1: Session is HOT (Currently in VRAM/KV Cache)
    if (session.tier == SessionTier.hot && session.seqId != null) {
      final seqId = session.seqId!;
      final size = lib.llama_state_seq_get_size(_context, seqId);
      final dataPtr = malloc<Uint8>(size);
      try {
        // Extract directly from C++ context
        lib.llama_state_seq_get_data(_context, dataPtr, size, seqId);
        final payload = dataPtr.asTypedList(size);
        final wrapped = StateCodec.encode(
          payload,
          nPos: session.nPos,
          nKeep: session.nKeep,
        );
        await File(path).writeAsBytes(wrapped, flush: true);
      } finally {
        malloc.free(dataPtr);
      }
    }
    // Case 2: Session is WARM (Currently in RAM buffer)
    else if (session.tier == SessionTier.warm && session.stateBuffer != null) {
      final wrapped = StateCodec.encode(
        session.stateBuffer!,
        nPos: session.nPos,
        nKeep: session.nKeep,
      );
      await File(path).writeAsBytes(wrapped, flush: true);
    }
    // Case 3: Session is COLD (Already on disk in the default location)
    else if (session.tier == SessionTier.cold && session.coldFilePath != null) {
      final bytes = await File(session.coldFilePath!).readAsBytes();
      final decoded = StateCodec.decode(bytes);
      final wrapped = StateCodec.encode(
        decoded.payload,
        nPos: decoded.nPos ?? session.nPos,
        nKeep: decoded.nKeep ?? session.nKeep,
      );
      await File(path).writeAsBytes(wrapped, flush: true);
    }
  }

  /// Manually loads a session from a specific file path.
  /// Returns true if successful.
  Future<bool> loadSession(String sessionId, String path) async {
    _checkDisposed();
    final file = File(path);
    if (!await file.exists()) return false;

    // If a session with this ID already exists, remove it first.
    if (_sessions.containsKey(sessionId)) {
      await freeSession(sessionId);
    }

    try {
      final bytes = await file.readAsBytes();
      final decoded = StateCodec.decode(bytes);

      // Initialize a new WARM session (it will be moved to HOT when a prompt is sent)
      final sampler = SamplerFactory.build(
        lib: lib,
        vocab: _sharedModel.vocab,
        params: defaultSamplerParams,
      );

      final session = ServiceSession(
        id: sessionId,
        seqId: null,
        sampler: sampler,
      );

      session.tier = SessionTier.warm;
      session.stateBuffer = decoded.payload; // Store raw seq bytes
      if (decoded.hasHeader) {
        session.nPos = decoded.nPos ?? session.nPos;
        session.nKeep = decoded.nKeep ?? session.nKeep;
      }

      // Allocate decode buffer
      session.decodeCapacity = 256;
      session.decodeBuf = malloc<Char>(session.decodeCapacity);

      session.status = LlamaStatus.ready;

      _sessions[sessionId] = session;
      return true;
    } catch (e) {
      // ignore: avoid_print
      print("Error loading session from $path: $e");
      return false;
    }
  }
}
