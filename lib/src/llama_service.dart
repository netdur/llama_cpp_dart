import 'dart:async';
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
/// - Reuses loaded weights across service instances when possible.
/// - Keeps per-session KV caches and samplers.
/// - Supports multimodal prompts and session save/load like [Llama].
/// - Exposes hooks for round-robin/continuous batching schedulers.
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
  final Map<String, LlamaStatus> _status = {};

  late final _SharedModelHandle _sharedModel;
  Pointer<mtmd_context> _mctx = nullptr;
  bool _visionEnabled = false;
  bool _disposed = false;

  /// Creates a service that can host multiple sessions on the same weights.
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
      // ignore: avoid_print
      print("modelPath: $modelPath");
      // ignore: avoid_print
      print("libraryPath: ${Llama.libraryPath}");
    }

    if (mmprojPath != null && mmprojPath.isNotEmpty) {
      final mprojPathPtr = mmprojPath.toNativeUtf8().cast<Char>();
      try {
        var mparam = lib.mtmd_context_params_default();
        mparam.use_gpu = this.modelParams.nGpuLayers != 0;
        _mctx = lib.mtmd_init_from_file(mprojPathPtr, _sharedModel.model, mparam);
        if (_mctx == nullptr) {
          throw LlamaException(
              "Failed to create multimodal projector context from $mmprojPath");
        }
        _visionEnabled = true;
      } finally {
        malloc.free(mprojPathPtr);
      }
    }
  }

  /// Returns true when the service or all sessions have been disposed.
  bool get isDisposed => _disposed;

  /// Current status for a specific session.
  LlamaStatus status(String sessionId) => _status[sessionId] ?? LlamaStatus.uninitialized;

  /// Creates a new logical user session with its own KV cache and sampler.
  void createSession(
    String sessionId, {
    ContextParams? contextParams,
    SamplerParams? samplerParams,
  }) {
    _checkDisposed();
    if (_sessions.containsKey(sessionId)) return;

    final ctxParams = contextParams ?? defaultContextParams;
    final sampler = _initializeSampler(samplerParams ?? defaultSamplerParams);

    final nativeParams = ctxParams.get();
    Pointer<llama_context> ctx = nullptr;
    llama_batch batch;
    final seqIds = <Pointer<llama_seq_id>>[];

    try {
      ctx = lib.llama_new_context_with_model(_sharedModel.model, nativeParams);
      if (ctx == nullptr) {
        throw LlamaException("Could not create context for $sessionId");
      }

      batch = lib.llama_batch_init(nativeParams.n_batch, 0, 1);
      for (int i = 0; i < nativeParams.n_batch; i++) {
        final seq = calloc<llama_seq_id>();
        seq.value = 0;
        seqIds.add(seq);
      }
    } catch (e) {
      if (ctx != nullptr) lib.llama_free(ctx);
      lib.llama_sampler_free(sampler);
      for (final ptr in seqIds) {
        calloc.free(ptr);
      }
      rethrow;
    }

    final session = _ServiceSession(
      id: sessionId,
      context: ctx,
      batch: batch,
      seqIds: seqIds,
      sampler: sampler,
      contextParams: ctxParams,
    );

    session.decodeBuf = malloc<Char>(256);

    _sessions[sessionId] = session;
    _status[sessionId] = LlamaStatus.ready;
  }

  /// Removes a session and frees its context memory.
  void freeSession(String sessionId) {
    final session = _sessions.remove(sessionId);
    _status.remove(sessionId);
    if (session == null) return;
    session.dispose(lib);
  }

  /// Sets the prompt for a specific session.
  Future<void> setPrompt(
    String sessionId,
    String prompt, {
    void Function(int current, int total)? onProgress,
  }) {
    _checkDisposed();
    final session = _requireSession(sessionId);
    if (prompt.isEmpty) throw ArgumentError('Prompt cannot be empty');

    final nCtx = session.contextParams.nCtx;
    final autoTrim = session.contextParams.autoTrimContext;
    if (!autoTrim && session.nPos >= nCtx - 10) {
      throw LlamaException("Context full (pos: ${session.nPos}, limit: $nCtx)");
    }

    Pointer<Utf8>? promptUtf8Ptr;

    try {
      _status[sessionId] = LlamaStatus.generating;
      session.pendingBytes.clear();

      if (session.nPos == 0) {
        if (session.context.address != 0) {
          final mem = lib.llama_get_memory(session.context);
          lib.llama_memory_clear(mem, true);
        }
        if (session.sampler != nullptr) lib.llama_sampler_reset(session.sampler);
        session.batch.n_tokens = 0;
      }

      promptUtf8Ptr = prompt.toNativeUtf8();
      final int promptBytes = promptUtf8Ptr.length;
      final Pointer<Char> promptCharPtr = promptUtf8Ptr.cast<Char>();

      final bool addBos = session.nPos == 0;

      session.nPrompt = -lib.llama_tokenize(_sharedModel.vocab, promptCharPtr,
          promptBytes, nullptr, 0, addBos, true);

      if (session.nPrompt <= 0) throw LlamaException("Token estimate failed");
      if (session.nPrompt > nCtx - 10) throw LlamaException("Prompt too large");
      if (!autoTrim && session.nPos + session.nPrompt >= nCtx) {
        throw LlamaException("Context limit exceeded");
      }

      if (session.tokens != nullptr) malloc.free(session.tokens);
      session.tokens = malloc<llama_token>(session.nPrompt);

      final int actualTokens = lib.llama_tokenize(_sharedModel.vocab, promptCharPtr,
          promptBytes, session.tokens, session.nPrompt, addBos, true);

      if (actualTokens < 0) throw LlamaException("Tokenization failed");
      session.nPrompt = actualTokens;

      final (_, trimmed) = _maybeTrimContext(session, session.nPrompt);
      if (trimmed) {
        session.batch.n_tokens = 0;
      }

      if (session.nPrompt > nCtx - 10) throw LlamaException("Prompt too large");
      if (session.nPos + session.nPrompt >= nCtx) {
        throw LlamaException("Context limit exceeded");
      }

      int batchCapacity = session.contextParams.nBatch;
      int processed = 0;
      final startPos = session.nPos;

      while (processed < session.nPrompt) {
        int nChunk = session.nPrompt - processed;
        if (nChunk > batchCapacity) nChunk = batchCapacity;

        session.batch.n_tokens = nChunk;
        for (int i = 0; i < nChunk; i++) {
          int tokenIdx = processed + i;
          session.batch.token[i] = session.tokens[tokenIdx];
          session.batch.pos[i] = startPos + processed + i;
          session.batch.n_seq_id[i] = 1;
          session.batch.seq_id[i] = session.seqIds[i];
          session.batch.seq_id[i].value = 0;
          final isLastGlobalToken = tokenIdx == session.nPrompt - 1;
          session.batch.logits[i] = isLastGlobalToken ? 1 : 0;
        }

        if (lib.llama_decode(session.context, session.batch) != 0) {
          throw LlamaException("Failed to decode prompt chunk");
        }

        session.nPos += nChunk;
        processed += nChunk;
        onProgress?.call(processed, session.nPrompt);

        if (processed < session.nPrompt && processed % (batchCapacity * 4) == 0) {
          await Future<void>.delayed(Duration.zero);
        }
      }

      session.nGeneratedTotal = 0;
    } catch (e) {
      _status[sessionId] = LlamaStatus.error;
      if (session.tokens != nullptr) {
        malloc.free(session.tokens);
        session.tokens = nullptr;
      }
      rethrow;
    } finally {
      if (promptUtf8Ptr != null) malloc.free(promptUtf8Ptr);
    }
  }

  (String, bool, bool) getNextWithStatus(String sessionId) {
    _checkDisposed();
    final session = _requireSession(sessionId);

    try {
      final nCtx = session.contextParams.nCtx;
      final tokensToAdd =
          session.batch.n_tokens == 0 ? 1 : session.batch.n_tokens;
      final (shifted, trimmed) = _maybeTrimContext(session, tokensToAdd);

      if (trimmed) {
        if (shifted > 0) {
          for (int i = 0; i < session.batch.n_tokens; i++) {
            session.batch.pos[i] = max(0, session.batch.pos[i] - shifted);
          }
        } else {
          for (int i = 0; i < session.batch.n_tokens; i++) {
            session.batch.pos[i] = i;
          }
        }
      }

      if (session.nPos >= nCtx - 2) {
        return ("\n\n[Context limit reached]", true, true);
      }

      final nGenerated = session.nGeneratedTotal;
      final targetPredict = session.contextParams.nPredict;
      if (targetPredict > 0 && nGenerated >= targetPredict) {
        return ("", true, false);
      }

      if (lib.llama_decode(session.context, session.batch) != 0) {
        if (session.nPos >= nCtx - 10) {
          return ("\n\n[Context limit]", true, true);
        }
        throw LlamaException("Failed to eval");
      }

      session.nPos += session.batch.n_tokens;

      int newTokenId = lib.llama_sampler_sample(session.sampler, session.context, -1);
      bool isEos = lib.llama_token_is_eog(_sharedModel.vocab, newTokenId);

      String piece;
      if (isEos) {
        if (session.pendingBytes.isNotEmpty) {
          piece = utf8.decode(session.pendingBytes, allowMalformed: true);
          session.pendingBytes.clear();
        } else {
          piece = "";
        }
      } else {
        piece = _decodeToken(session, newTokenId);
      }

      session.batch.token[0] = newTokenId;
      session.batch.pos[0] = session.nPos;
      session.batch.n_seq_id[0] = 1;
      session.batch.seq_id[0] = session.seqIds[0];
      session.batch.seq_id[0].value = 0;
      session.batch.logits[0] = 1;
      session.batch.n_tokens = 1;

      if (!isEos) {
        session.nGeneratedTotal += 1;
      }

      return (piece, isEos, false);
    } catch (e) {
      _status[sessionId] = LlamaStatus.error;
      throw LlamaException('Error generating text for $sessionId', e);
    }
  }

  Stream<String> generateText(String sessionId) async* {
    _checkDisposed();
    try {
      while (true) {
        final (text, isDone, contextLimitReached) = getNextWithStatus(sessionId);
        if (text.isNotEmpty) yield text;
        if (contextLimitReached) {
          _status[sessionId] = LlamaStatus.ready;
          break;
        }
        if (isDone) break;
      }
    } catch (e) {
      _status[sessionId] = LlamaStatus.error;
      throw LlamaException('Error in text generation stream for $sessionId', e);
    } finally {
      if (_status[sessionId] != LlamaStatus.error && !_disposed) {
        _status[sessionId] = LlamaStatus.ready;
      }
    }
  }

  Future<String> generateCompleteText(String sessionId, {int? maxTokens}) async {
    _checkDisposed();
    final buffer = StringBuffer();
    int tokenCount = 0;
    final session = _requireSession(sessionId);
    final limit = maxTokens ?? session.contextParams.nPredict;

    try {
      while (true) {
        final (text, isDone, contextLimitReached) = getNextWithStatus(sessionId);
        buffer.write(text);
        tokenCount++;
        if (contextLimitReached) {
          _status[sessionId] = LlamaStatus.ready;
          break;
        }
        if (isDone || (limit > 0 && tokenCount >= limit)) break;
      }
      return buffer.toString();
    } catch (e) {
      _status[sessionId] = LlamaStatus.error;
      throw LlamaException('Error generating complete text for $sessionId', e);
    } finally {
      if (_status[sessionId] != LlamaStatus.error && !_disposed) {
        _status[sessionId] = LlamaStatus.ready;
      }
    }
  }

  Stream<String> generateWithMedia(
    String sessionId,
    String prompt, {
    required List<LlamaInput> inputs,
  }) async* {
    _checkDisposed();
    if (!_visionEnabled || _mctx == nullptr) {
      throw LlamaException('Vision disabled – construct with mmprojPath.');
    }
    if (inputs.isEmpty) throw ArgumentError('No images given');

    final session = _requireSession(sessionId);
    final images = inputs.whereType<LlamaImage>().toList();
    const marker = '<image>';
    final mCnt = marker.allMatches(prompt).length;
    if (mCnt != images.length) {
      throw ArgumentError(
          'Prompt has $mCnt <image> marker(s) but ${images.length} image(s) supplied.');
    }

    Pointer<mtmd_input_text> txtPtr = nullptr;
    Pointer<Char> fullPtr = nullptr;
    Pointer<mtmd_input_chunks> chunks = nullptr;
    final bitmapRefs = <BitmapPointers>[];
    Pointer<Pointer<mtmd_bitmap>> bmpArr = nullptr;

    try {
      _status[sessionId] = LlamaStatus.generating;
      lib.llama_sampler_reset(session.sampler);
      clearSession(sessionId);

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

      var nPast = 0;
      final nCtx = session.contextParams.nCtx;
      final nChunks = lib.mtmd_input_chunks_size(chunks);

      final b = session.batch;
      final Pointer<llama_token> originalTokenPtr = b.token;
      final int batchCapacity = session.contextParams.nBatch;

      for (var i = 0; i < nChunks; ++i) {
        b.n_tokens = 0;
        final chunk = lib.mtmd_input_chunks_get(chunks, i);
        final type = lib.mtmd_input_chunk_get_type(chunk);

        if (type == mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_IMAGE) {
          if (lib.mtmd_encode_chunk(_mctx, chunk) != 0) {
            throw LlamaException('encode image failed');
          }
          final embd = lib.mtmd_get_output_embd(_mctx);
          final nTok = lib.mtmd_input_chunk_get_n_tokens(chunk);

          if (nTok > batchCapacity) throw LlamaException('Image chunk > batch');
          if (nPast + nTok > nCtx) throw LlamaException('n_ctx overflow');

          b.token = nullptr;
          b.embd = embd;
          for (var k = 0; k < nTok; ++k) {
            b.pos[k] = nPast + k;
            b.n_seq_id[k] = 1;
            b.seq_id[k] = session.seqIds[k];
            b.seq_id[k].value = 0;
            b.logits[k] = 0;
          }
          b.logits[nTok - 1] = 1;
          b.n_tokens = nTok;

          if (lib.llama_decode(session.context, b) != 0) {
            throw LlamaException('decode image failed');
          }
          nPast += nTok;
        } else {
          final nPtr = malloc<Size>();
          final tokPt = lib.mtmd_input_chunk_get_tokens_text(chunk, nPtr);
          final nTok = nPtr.value;
          malloc.free(nPtr);

          if (nTok > batchCapacity) throw LlamaException('Text chunk > batch');
          if (nPast + nTok > nCtx) throw LlamaException('n_ctx overflow');

          b.token = originalTokenPtr;
          b.embd = nullptr;
          for (var k = 0; k < nTok; ++k) {
            b.token[k] = tokPt[k];
            b.pos[k] = nPast + k;
            b.n_seq_id[k] = 1;
            b.seq_id[k] = session.seqIds[k];
            b.seq_id[k].value = 0;
            b.logits[k] = 0;
          }
          b.logits[nTok - 1] = 1;
          b.n_tokens = nTok;

          if (lib.llama_decode(session.context, b) != 0) {
            throw LlamaException('decode text failed');
          }
          nPast += nTok;
        }
      }
      session.nPos = nPast;

      var produced = 0;
      final predictLimit = session.contextParams.nPredict;
      while (nPast < nCtx && (predictLimit == -1 || produced < predictLimit)) {
        b.token = originalTokenPtr;
        b.embd = nullptr;

        final tok = lib.llama_sampler_sample(session.sampler, session.context, -1);
        bool isEos = lib.llama_token_is_eog(_sharedModel.vocab, tok);

        String piece;
        if (isEos) {
          if (session.pendingBytes.isNotEmpty) {
            piece = utf8.decode(session.pendingBytes, allowMalformed: true);
            session.pendingBytes.clear();
          } else {
            piece = "";
          }
        } else {
          piece = _decodeToken(session, tok);
        }
        if (piece.isNotEmpty) yield piece;
        if (isEos) break;

        b.n_tokens = 0;
        b.token[0] = tok;
        b.pos[0] = nPast;
        b.n_seq_id[0] = 1;
        b.seq_id[0] = session.seqIds[0];
        b.seq_id[0].value = 0;
        b.logits[0] = 1;
        b.n_tokens = 1;

        if (lib.llama_decode(session.context, b) != 0) {
          throw LlamaException('decode token failed');
        }
        ++nPast;
        ++produced;
        session.nPos = nPast;
      }
      session.nPos = nPast;
    } catch (e) {
      _status[sessionId] = LlamaStatus.error;
      rethrow;
    } finally {
      if (session.batch.seq_id != nullptr) {
        final batchCapacity = session.contextParams.nBatch;
        for (int i = 0; i < batchCapacity; ++i) {
          session.batch.seq_id[i] = nullptr;
        }
      }
      if (chunks != nullptr) lib.mtmd_input_chunks_free(chunks);
      if (bmpArr != nullptr) malloc.free(bmpArr);
      for (final r in bitmapRefs) {
        if (r.bitmap != nullptr) lib.mtmd_bitmap_free(r.bitmap);
      }
      if (txtPtr != nullptr) calloc.free(txtPtr);
      if (fullPtr != nullptr) malloc.free(fullPtr);

      if (_status[sessionId] != LlamaStatus.error && !_disposed) {
        _status[sessionId] = LlamaStatus.ready;
      }
    }
  }

  bool wasContextLimitReached(String sessionId) {
    final session = _requireSession(sessionId);
    final nCtx = session.contextParams.nCtx;
    return session.nPos >= nCtx - 2;
  }

  int getRemainingContextSpace(String sessionId) {
    final session = _requireSession(sessionId);
    final nCtx = session.contextParams.nCtx;
    return nCtx - session.nPos;
  }

  /// Saves session KV cache for a specific session to disk (with header).
  void saveSession(String sessionId, String path) {
    _checkDisposed();
    final session = _requireSession(sessionId);

    final tempPath = "${path}.tmp.${DateTime.now().millisecondsSinceEpoch}";
    final tempPathPtr = tempPath.toNativeUtf8().cast<Char>();

    try {
      final result =
          lib.llama_state_save_file(session.context, tempPathPtr, nullptr, 0);
      if (!result) {
        throw LlamaException(
            'Failed to save session (llama_state_save_file returned false)');
      }

      final tempFile = File(tempPath);
      final finalFile = File(path);

      final header = ByteData(16);
      header.setUint32(0, 0x44415254, Endian.little);
      header.setUint32(4, 1, Endian.little);
      header.setUint32(8, session.nPos, Endian.little);
      header.setUint32(12, session.nPrompt, Endian.little);

      final headerBytes = header.buffer.asUint8List();

      final writer = finalFile.openSync(mode: FileMode.write);
      try {
        writer.writeFromSync(headerBytes);

        final reader = tempFile.openSync(mode: FileMode.read);
        try {
          const chunkSize = 1024 * 1024; // 1MB
          while (true) {
            final chunk = reader.readSync(chunkSize);
            if (chunk.isEmpty) break;
            writer.writeFromSync(chunk);
          }
        } finally {
          reader.closeSync();
        }
        writer.flushSync();
      } finally {
        writer.closeSync();
      }

      if (_verbose) {
        // ignore: avoid_print
        print("Session $sessionId saved to $path");
      }

      if (tempFile.existsSync()) tempFile.deleteSync();
    } finally {
      malloc.free(tempPathPtr);
    }
  }

  /// Loads session KV cache for a specific session from disk.
  bool loadSession(String sessionId, String path) {
    _checkDisposed();
    final session = _requireSession(sessionId);
    final file = File(path);
    if (!file.existsSync()) return false;

    final bytes = file.readAsBytesSync();
    if (bytes.length < 16) {
      // ignore: avoid_print
      print("Warning: Session file too small.");
      return false;
    }

    final header = ByteData.sublistView(bytes, 0, 16);
    final magic = header.getUint32(0, Endian.little);

    if (magic == 0x44415254) {
      final savedPos = header.getUint32(8, Endian.little);
      final savedPrompt = header.getUint32(12, Endian.little);
      if (_verbose) {
        // ignore: avoid_print
        print("Custom header found. Saved nPos=$savedPos");
      }

      final stateData = bytes.sublist(16);

      final tempPath =
          "${path}.tmp.load.${DateTime.now().millisecondsSinceEpoch}";
      final tempFile = File(tempPath);
      tempFile.writeAsBytesSync(stateData, flush: true);

      final tempPathPtr = tempPath.toNativeUtf8().cast<Char>();
      final countOut = calloc<Size>(1);

      try {
        final result = lib.llama_state_load_file(
            session.context, tempPathPtr, nullptr, 0, countOut);
        if (_verbose) {
          // ignore: avoid_print
          print("Native load result: $result");
        }
        if (result) {
          final restoredPos = countOut.value;
          session.nPos = restoredPos != 0 ? restoredPos : savedPos;
          session.nPrompt = savedPrompt != 0 ? savedPrompt : session.nPos;
        }
        return result;
      } finally {
        malloc.free(tempPathPtr);
        calloc.free(countOut);
        if (tempFile.existsSync()) tempFile.deleteSync();
      }
    } else {
      final pathPtr = path.toNativeUtf8().cast<Char>();
      final countOut = calloc<Size>(1);
      try {
        final result =
            lib.llama_state_load_file(session.context, pathPtr, nullptr, 0, countOut);
        if (result) {
          session.nPos = countOut.value;
          final metaFile = File("$path.meta");
          if (metaFile.existsSync()) {
            try {
              final json = jsonDecode(metaFile.readAsStringSync());
              session.nPos = session.nPos != 0 ? session.nPos : (json["nPos"] ?? 0);
              session.nPrompt = json["nPrompt"] ?? session.nPos;
            } catch (_) {}
          } else {
            session.nPrompt = session.nPos;
          }
        }
        return result;
      } finally {
        malloc.free(pathPtr);
        calloc.free(countOut);
      }
    }
  }

  /// Saves the current session state to RAM as bytes.
  Uint8List saveState(String sessionId) {
    _checkDisposed();
    final session = _requireSession(sessionId);

    final int stateSize = lib.llama_get_state_size(session.context);
    const int headerSize = 16;
    final int totalSize = stateSize + headerSize;

    final ptr = malloc<Uint8>(totalSize);

    try {
      final headerData = ptr.asTypedList(headerSize).buffer.asByteData();
      headerData.setUint32(0, 0x4C4C5346, Endian.little);
      headerData.setUint32(4, 1, Endian.little);
      headerData.setUint32(8, session.nPos, Endian.little);
      headerData.setUint32(12, session.nPrompt, Endian.little);

      final dataPtr = Pointer<Uint8>.fromAddress(ptr.address + headerSize);
      lib.llama_copy_state_data(session.context, dataPtr);

      return Uint8List.fromList(ptr.asTypedList(totalSize));
    } finally {
      malloc.free(ptr);
    }
  }

  /// Restores session state from RAM bytes.
  void loadState(String sessionId, Uint8List stateData) {
    _checkDisposed();
    final session = _requireSession(sessionId);

    const int headerSize = 16;
    if (stateData.length < headerSize) {
      throw LlamaException('State data too short');
    }

    final header = ByteData.sublistView(stateData, 0, headerSize);
    final magic = header.getUint32(0, Endian.little);
    final version = header.getUint32(4, Endian.little);

    if (magic != 0x4C4C5346 || version != 1) {
      throw LlamaException('Invalid state data header');
    }

    session.nPos = header.getUint32(8, Endian.little);
    session.nPrompt = header.getUint32(12, Endian.little);

    final int expectedStateSize = lib.llama_get_state_size(session.context);
    if (stateData.length - headerSize != expectedStateSize) {
      // ignore: avoid_print
      print(
          "Warning: State size mismatch. Expected $expectedStateSize, got ${stateData.length - headerSize}");
    }

    final ptr = malloc<Uint8>(expectedStateSize);

    try {
      final dataView = stateData.sublist(headerSize);
      ptr.asTypedList(expectedStateSize).setAll(0, dataView);

      lib.llama_set_state_data(session.context, ptr);
    } finally {
      malloc.free(ptr);
    }
  }

  /// Resets a session KV cache without touching the shared weights.
  void clearSession(String sessionId) {
    _checkDisposed();
    final session = _requireSession(sessionId);
    try {
      if (session.tokens != nullptr) {
        malloc.free(session.tokens);
        session.tokens = nullptr;
      }
      session.nPrompt = 0;
      session.nPos = 0;
      session.pendingBytes.clear();

      if (session.context.address != 0) {
        final mem = lib.llama_get_memory(session.context);
        lib.llama_memory_clear(mem, true);
      }
      session.batch.n_tokens = 0;
      _status[sessionId] = LlamaStatus.ready;
    } catch (e) {
      _status[sessionId] = LlamaStatus.error;
      throw LlamaException('Failed to clear $sessionId', e);
    }
  }

  List<int> tokenize(String sessionId, String text, bool addBos) {
    _checkDisposed();
    final session = _requireSession(sessionId);
    if (text.isEmpty) throw ArgumentError('Empty text');
    final utf8Ptr = text.toNativeUtf8();
    final length = utf8Ptr.length;
    final textPtr = utf8Ptr.cast<Char>();
    try {
      int nTokens = -lib.llama_tokenize(
          _sharedModel.vocab, textPtr, length, nullptr, 0, addBos, true);
      final tokens = malloc<llama_token>(nTokens);
      try {
        int actual = lib.llama_tokenize(_sharedModel.vocab, textPtr, length,
            tokens, nTokens, addBos, true);
        return List<int>.generate(actual, (i) => tokens[i]);
      } finally {
        malloc.free(tokens);
      }
    } finally {
      malloc.free(utf8Ptr);
    }
  }

  /// Helper to drive multiple sessions in a round-robin fashion.
  /// Each iteration advances every provided session by one token.
  Stream<(String sessionId, String text)> roundRobin(
    Iterable<String> sessionIds,
  ) async* {
    final active = Set<String>.from(sessionIds);
    while (active.isNotEmpty) {
      final toRemove = <String>[];
      for (final id in List<String>.from(active)) {
        final (piece, isDone, limit) = getNextWithStatus(id);
        if (piece.isNotEmpty) {
          yield (id, piece);
        }
        if (isDone || limit) {
          toRemove.add(id);
          _status[id] = LlamaStatus.ready;
        }
      }
      for (final id in toRemove) {
        active.remove(id);
      }
      if (active.isEmpty) break;
    }
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    for (final session in _sessions.values) {
      session.dispose(lib);
    }
    _sessions.clear();
    _status.clear();

    if (_visionEnabled && _mctx != nullptr) {
      lib.mtmd_free(_mctx);
      _mctx = nullptr;
    }

    _releaseModel(_sharedModel);
  }

  // --- internal helpers ----------------------------------------------------

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
      throw ArgumentError("Session $id does not exist. Call createSession first.");
    }
    return session;
  }

  (int shiftApplied, bool trimmed) _maybeTrimContext(
      _ServiceSession session, int tokensNeeded) {
    final params = session.contextParams;
    if (!params.autoTrimContext) return (0, false);

    final nCtx = params.nCtx;
    final needed = tokensNeeded <= 0 ? 1 : tokensNeeded;

    if (session.nPos + needed < nCtx - 1) return (0, false);

    final keepCap = nCtx - needed;
    if (keepCap <= 0) {
      return (0, false);
    }

    final keepTokens = max(0, min(params.trimKeepTokens, keepCap));
    final trimStart = max(0, session.nPos - keepTokens);
    if (trimStart <= 0) return (0, false);

    final mem = lib.llama_get_memory(session.context);
    final removed = lib.llama_memory_seq_rm(mem, 0, 0, trimStart);
    if (!removed) return (0, false);

    if (!lib.llama_memory_can_shift(mem)) {
      lib.llama_memory_clear(mem, true);
      session.nPos = 0;
      session.nPrompt = 0;
      session.pendingBytes.clear();
      if (_verbose) {
        // ignore: avoid_print
        print("llama_service: auto-trim fell back to clear (backend cannot shift)");
      }
      return (0, true);
    }

    lib.llama_memory_seq_add(mem, 0, trimStart, -1, -trimStart);
    session.nPos = max(0, session.nPos - trimStart);
    session.nPrompt = max(0, session.nPrompt - trimStart);

    if (_verbose) {
      // ignore: avoid_print
      print(
          "llama_service: auto-trimmed $trimStart token(s), keeping ${session.nPos}");
    }

    return (trimStart, true);
  }

  String _decodeToken(_ServiceSession session, int tokenId) {
    final buf = session.decodeBuf != nullptr ? session.decodeBuf : malloc<Char>(256);
    try {
      int n =
          lib.llama_token_to_piece(_sharedModel.vocab, tokenId, buf, 256, 0, true);
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
    } finally {
      if (session.decodeBuf == nullptr) {
        // Only free if we allocated ad-hoc.
        malloc.free(buf);
      }
    }
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
      final grammar =
          lib.llama_sampler_init_grammar(_sharedModel.vocab, grammarStrPtr, grammarRootPtr);
      if (grammar != nullptr) {
        lib.llama_sampler_chain_add(smpl, grammar);
      }
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

    if (samplerParams.dryMultiplier > 0.0) {
      try {
        final breakers = samplerParams.dryBreakers;
        final breakerCount = breakers.length;

        final breakersPtr = malloc<Pointer<Char>>(breakerCount);
        final allocatedStrings = <Pointer<Char>>[];

        for (int i = 0; i < breakerCount; i++) {
          final strPtr = breakers[i].toNativeUtf8().cast<Char>();
          breakersPtr[i] = strPtr;
          allocatedStrings.add(strPtr);
        }

        final int nCtxTrain = lib.llama_model_n_ctx_train(_sharedModel.model);

        lib.llama_sampler_chain_add(
            smpl,
            lib.llama_sampler_init_dry(
              _sharedModel.vocab,
              nCtxTrain,
              samplerParams.dryMultiplier,
              samplerParams.dryBase,
              samplerParams.dryAllowedLen,
              samplerParams.dryPenaltyLastN,
              breakersPtr,
              breakerCount,
            ));

        for (var ptr in allocatedStrings) {
          malloc.free(ptr);
        }
        malloc.free(breakersPtr);
      } catch (e) {}
    }

    if (samplerParams.mirostat == 2) {
      lib.llama_sampler_chain_add(
          smpl,
          lib.llama_sampler_init_mirostat_v2(samplerParams.seed,
              samplerParams.mirostatTau, samplerParams.mirostatEta));
    } else if (samplerParams.mirostat == 1) {
      lib.llama_sampler_chain_add(
          smpl,
          lib.llama_sampler_init_mirostat(
              lib.llama_n_vocab(_sharedModel.vocab),
              samplerParams.seed,
              samplerParams.mirostatTau,
              samplerParams.mirostatEta,
              samplerParams.mirostatM));
    } else {
      lib.llama_sampler_chain_add(
          smpl, lib.llama_sampler_init_top_k(samplerParams.topK));

      lib.llama_sampler_chain_add(
          smpl, lib.llama_sampler_init_top_p(samplerParams.topP, 1));

      lib.llama_sampler_chain_add(
          smpl, lib.llama_sampler_init_min_p(samplerParams.minP, 1));

      lib.llama_sampler_chain_add(
          smpl, lib.llama_sampler_init_typical(samplerParams.typical, 1));

      if (samplerParams.dynatempRange > 0.0) {
        try {
          lib.llama_sampler_chain_add(
              smpl,
              lib.llama_sampler_init_temp_ext(samplerParams.temp,
                  samplerParams.dynatempRange, samplerParams.dynatempExponent));
        } catch (_) {
          lib.llama_sampler_chain_add(
              smpl, lib.llama_sampler_init_temp(samplerParams.temp));
        }
      } else {
        lib.llama_sampler_chain_add(
            smpl, lib.llama_sampler_init_temp(samplerParams.temp));
      }

      if (samplerParams.xtcProbability > 0.0) {
        try {
          lib.llama_sampler_chain_add(
              smpl,
              lib.llama_sampler_init_xtc(samplerParams.xtcProbability,
                  samplerParams.xtcThreshold, 1, samplerParams.seed));
        } catch (_) {}
      }
    }

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
}

class _ServiceSession {
  final String id;
  final Pointer<llama_context> context;
  final llama_batch batch;
  final List<Pointer<llama_seq_id>> seqIds;
  final Pointer<llama_sampler> sampler;
  final ContextParams contextParams;

  Pointer<llama_token> tokens = nullptr;
  Pointer<Char> decodeBuf = nullptr;
  List<int> pendingBytes = [];
  int nPos = 0;
  int nPrompt = 0;
  int nGeneratedTotal = 0;

  _ServiceSession({
    required this.id,
    required this.context,
    required this.batch,
    required this.seqIds,
    required this.sampler,
    required this.contextParams,
  });

  void dispose(llama_cpp lib) {
    if (tokens != nullptr) {
      malloc.free(tokens);
      tokens = nullptr;
    }
    if (decodeBuf != nullptr) {
      malloc.free(decodeBuf);
      decodeBuf = nullptr;
    }
    if (sampler != nullptr) {
      lib.llama_sampler_free(sampler);
    }
    if (context.address != 0) {
      lib.llama_free(context);
    }
    try {
      if (batch.seq_id != nullptr) {
        final batchCapacity = contextParams.nBatch;
        for (int i = 0; i < batchCapacity; ++i) {
          batch.seq_id[i] = nullptr;
        }
      }
      lib.llama_batch_free(batch);
    } catch (_) {}
    for (final ptr in seqIds) {
      calloc.free(ptr);
    }
    seqIds.clear();
  }
}

class _SharedModelHandle {
  final Pointer<llama_model> model;
  final Pointer<llama_vocab> vocab;
  int refs;

  _SharedModelHandle({required this.model, required this.vocab, this.refs = 1});
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
