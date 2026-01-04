import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:async';
import 'dart:math' show max, min, sqrt;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'sampler_params.dart';
import 'model_params.dart';
import 'llama_cpp.dart';
import 'context_params.dart';
import 'llama_input.dart';

typedef LlamaLogCallback = Void Function(
    UnsignedInt level, Pointer<Char> text, Pointer<Void> userData);
typedef LlamaLogCallbackDart = void Function(
    int level, Pointer<Char> text, Pointer<Void> userData);

/// Custom exception for Llama-specific errors
class LlamaException implements Exception {
  final String message;
  final dynamic originalError;

  LlamaException(this.message, [this.originalError]);

  @override
  String toString() =>
      'LlamaException: $message${originalError != null ? ' ($originalError)' : ''}';
}

/// Status tracking for the Llama instance
enum LlamaStatus {
  uninitialized,
  loading,
  ready,
  generating,
  error,
  disposed,
}

/// A Dart wrapper for llama.cpp functionality.
class Llama {
  static llama_cpp? _lib;

  late Pointer<llama_model> model;
  late Pointer<llama_vocab> vocab;
  late llama_batch batch;

  Pointer<llama_sampler> _smpl = nullptr;
  Pointer<llama_token> _tokens = nullptr;
  Pointer<llama_token> _tokenPtr = nullptr;
  Pointer<mtmd_context> _mctx = nullptr;

  final List<Pointer<llama_seq_id>> _batchSeqIds = [];

  final Map<String, _LlamaSlot> _slots = {};
  String _currentSlotId = "default";

  int _nPredict = 32;

  bool _verbose = false;
  bool _isInitialized = false;
  bool _isDisposed = false;
  bool _isVisionEnabled = false;
  LlamaStatus _status = LlamaStatus.uninitialized;

  static String? libraryPath = Platform.isAndroid ? "libmtmd.so" : null;

  LlamaStatus get status => _status;
  bool get isDisposed => _isDisposed;

  ContextParams? _contextParams;

  static llama_cpp get lib {
    if (_lib == null) {
      if (libraryPath != null) {
        _lib = llama_cpp(DynamicLibrary.open(libraryPath!));
      } else {
        _lib = llama_cpp(DynamicLibrary.process());
      }
    }
    return _lib!;
  }

  llama_cpp getLib() => _lib!;

  Pointer<llama_context> get context => _slots[_currentSlotId]!.context;

  int get _nPos => _slots[_currentSlotId]!.nPos;
  set _nPos(int v) => _slots[_currentSlotId]!.nPos = v;

  int get _nPrompt => _slots[_currentSlotId]!.nPrompt;
  set _nPrompt(int v) => _slots[_currentSlotId]!.nPrompt = v;

  List<int> get _pendingBytes => _slots[_currentSlotId]!.pendingBytes;

  /// Creates a new Llama instance.
  Llama(
    String modelPath, {
    String? mmprojPath,
    ModelParams? modelParams,
    ContextParams? contextParams,
    SamplerParams? samplerParams,
    bool verbose = false,
  }) {
    try {
      _verbose = verbose;
      _status = LlamaStatus.loading;

      _initializeLlama(
          modelPath, mmprojPath, modelParams, contextParams, samplerParams);

      contextParams ??= ContextParams();
      _contextParams = contextParams;
      _nPredict = contextParams.nPredict;

      _validateConfiguration();

      var nativeContextParams = contextParams.get();
      try {
        batch = lib.llama_batch_init(nativeContextParams.n_batch, 0, 1);

        for (int i = 0; i < nativeContextParams.n_batch; i++) {
          final seqIdPtr = calloc<llama_seq_id>();
          seqIdPtr.value = 0;
          _batchSeqIds.add(seqIdPtr);
        }
      } catch (e) {
        if (_smpl != nullptr) lib.llama_sampler_free(_smpl);
        if (context.address != 0) lib.llama_free(context);
        if (model.address != 0) lib.llama_free_model(model);
        if (_mctx != nullptr) lib.mtmd_free(_mctx);

        for (final ptr in _batchSeqIds) {
          calloc.free(ptr);
        }
        _batchSeqIds.clear();

        throw LlamaException('Failed to initialize batch', e);
      }

      _isInitialized = true;
      _status = LlamaStatus.ready;
    } catch (e) {
      _status = LlamaStatus.error;
      _isDisposed = true;
      throw LlamaException('Failed to initialize Llama', e);
    }
  }

  /// Creates a new slot (context) in VRAM for a specific user/session.
  /// [slotId] - Unique identifier for the user.
  void createSlot(String slotId) {
    if (_isDisposed) throw StateError('Disposed');
    if (_slots.containsKey(slotId)) return;

    final paramsNative = _contextParams!.get();
    final newCtx = lib.llama_new_context_with_model(model, paramsNative);
    if (newCtx == nullptr) {
      throw LlamaException("Failed to allocate VRAM for slot: $slotId");
    }

    _slots[slotId] = _LlamaSlot(newCtx);
  }

  /// Switches the active "brain" to the specified user.
  void setSlot(String slotId) {
    if (!_slots.containsKey(slotId)) {
      throw ArgumentError(
          "Slot $slotId does not exist. Call createSlot first.");
    }

    batch.n_tokens = 0;

    _currentSlotId = slotId;

    if (_smpl != nullptr) {
      lib.llama_sampler_reset(_smpl);
    }
  }

  /// Frees a specific slot from VRAM.
  void freeSlot(String slotId) {
    if (!_slots.containsKey(slotId)) return;

    if (slotId == _currentSlotId) {
      // If we are freeing the active slot, try to switch to another existing slot
      // or simply detach the current slot reference.
      String? nextSlot;
      if (slotId != "default" && _slots.containsKey("default")) {
        nextSlot = "default";
      } else if (_slots.isNotEmpty) {
        for (final key in _slots.keys) {
          if (key != slotId) {
            nextSlot = key;
            break;
          }
        }
      }

      if (nextSlot != null) {
        try {
          setSlot(nextSlot);
        } catch (_) {
          _currentSlotId = "";
        }
      } else {
        // No other slot available. We are freeing the last/only active slot.
        // Detach current slot ID so the equality check passes and we don't block freeing.
        _currentSlotId = "";
      }
    }

    final slot = _slots.remove(slotId)!;
    if (slot.context.address != 0) {
      lib.llama_free(slot.context);
    }

    // If we detached the current slot but still have others allocated, pick one
    // so subsequent calls to [context] have a valid target.
    if (_currentSlotId.isEmpty && _slots.isNotEmpty) {
      _currentSlotId = _slots.keys.first;
    }
  }

  void _validateConfiguration() {
    if (_nPredict == 0 || _nPredict < -1) {
      throw ArgumentError('nPredict must be positive or -1 for unlimited');
    }
  }

  /// Ensures there is room in the KV cache for [tokensNeeded] new tokens.
  /// If auto trimming is enabled, old tokens are removed and the remaining
  /// positions are shifted down to reclaim space.
  (int shiftApplied, bool trimmed) _maybeTrimContext(int tokensNeeded) {
    final params = _contextParams;
    if (params == null || !params.autoTrimContext) return (0, false);

    final nCtx = params.nCtx;
    final needed = tokensNeeded <= 0 ? 1 : tokensNeeded;

    if (_nPos + needed < nCtx - 1) return (0, false);

    final keepCap = nCtx - needed;
    if (keepCap <= 0) {
      return (0, false);
    }

    final keepTokens = max(0, min(params.trimKeepTokens, keepCap));
    final trimStart = max(0, _nPos - keepTokens);
    if (trimStart <= 0) return (0, false);

    final mem = lib.llama_get_memory(context);
    final removed = lib.llama_memory_seq_rm(mem, 0, 0, trimStart);
    if (!removed) return (0, false);

    if (!lib.llama_memory_can_shift(mem)) {
      lib.llama_memory_clear(mem, true);
      _nPos = 0;
      _nPrompt = 0;
      _pendingBytes.clear();
      if (_verbose) {
        // ignore: avoid_print
        print("llama: auto-trim fell back to clear (backend cannot shift)");
      }
      return (0, true);
    }

    lib.llama_memory_seq_add(mem, 0, trimStart, -1, -trimStart);
    _nPos = max(0, _nPos - trimStart);
    _nPrompt = max(0, _nPrompt - trimStart);

    if (_verbose) {
      // ignore: avoid_print
      print("llama: auto-trimmed $trimStart token(s), keeping $_nPos");
    }

    return (trimStart, true);
  }

  static void llamaLogCallbackNull(
      int level, Pointer<Char> text, Pointer<Void> userData) {}

  static void llamaLogCallbackPrint(
    int level,
    Pointer<Char> text,
    Pointer<Void> userData,
  ) {
    // char*  to utf8  to dart string
    final msg = text.cast<Utf8>().toDartString();
    print('llama native [$level] $msg');
  }

  void _initializeLlama(
      String modelPath,
      String? mmprojPath,
      ModelParams? modelParamsDart,
      ContextParams? contextParamsDart,
      SamplerParams? samplerParams) {
    if (_verbose == false) {
      final nullCallbackPointer =
          Pointer.fromFunction<LlamaLogCallback>(Llama.llamaLogCallbackNull);
      lib.llama_log_set(nullCallbackPointer, nullptr);
    }

    lib.llama_backend_init();
    if (!Platform.isAndroid) {
      lib.ggml_backend_load_all();
    }

    final ptr = lib.llama_print_system_info();
    final sysInfo = ptr.cast<Utf8>().toDartString();
    print(sysInfo);
    print("modelPath: $modelPath");
    print("libraryPath: $libraryPath");

    modelParamsDart ??= ModelParams();
    var modelParams = modelParamsDart.get();

    final modelPathPtr = modelPath.toNativeUtf8().cast<Char>();
    Pointer<llama_model> loadedModel = nullptr;
    try {
      loadedModel = lib.llama_load_model_from_file(modelPathPtr, modelParams);
      if (loadedModel == nullptr) {
        throw LlamaException("Could not load model at $modelPath");
      }
      model = loadedModel;
      vocab = lib.llama_model_get_vocab(model);
    } finally {
      malloc.free(modelPathPtr);
    }

    contextParamsDart ??= ContextParams();
    _contextParams = contextParamsDart;
    _nPredict = contextParamsDart.nPredict;
    var contextParams = contextParamsDart.get();

    Pointer<llama_context> loadedContext = nullptr;
    try {
      loadedContext = lib.llama_new_context_with_model(model, contextParams);

      if (loadedContext == nullptr) {
        lib.llama_free_model(model);
        throw LlamaException("Could not create context!");
      }
      _slots["default"] = _LlamaSlot(loadedContext);
      _currentSlotId = "default";
    } catch (e) {
      if (loadedContext != nullptr) lib.llama_free(loadedContext);
      lib.llama_free_model(model);
      rethrow;
    }

    _initializeSampler(samplerParams);

    if (mmprojPath != null && mmprojPath.isNotEmpty) {
      final mprojPathPtr = mmprojPath.toNativeUtf8().cast<Char>();
      try {
        var mparam = lib.mtmd_context_params_default();
        mparam.use_gpu = modelParamsDart.nGpuLayers != 0;
        _mctx = lib.mtmd_init_from_file(mprojPathPtr, model, mparam);
        if (_mctx == nullptr) {
          throw LlamaException(
              "Failed to create multimodal projector context from $mmprojPath");
        }
        _isVisionEnabled = true;
      } finally {
        malloc.free(mprojPathPtr);
      }
    }

    _tokenPtr = malloc<llama_token>();
  }

  void _initializeSampler(SamplerParams? samplerParams) {
    samplerParams ??= SamplerParams();

    llama_sampler_chain_params sparams =
        lib.llama_sampler_chain_default_params();
    sparams.no_perf = false;
    _smpl = lib.llama_sampler_chain_init(sparams);

    if (samplerParams.greedy) {
      lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_greedy());
      return;
    }

    final grammarStrPtr = samplerParams.grammarStr.toNativeUtf8().cast<Char>();
    final grammarRootPtr =
        samplerParams.grammarRoot.toNativeUtf8().cast<Char>();

    if (samplerParams.grammarStr.isNotEmpty) {
      final grammar =
          lib.llama_sampler_init_grammar(vocab, grammarStrPtr, grammarRootPtr);
      if (grammar != nullptr) {
        lib.llama_sampler_chain_add(_smpl, grammar);
      }
    }
    malloc.free(grammarStrPtr);
    malloc.free(grammarRootPtr);

    lib.llama_sampler_chain_add(
        _smpl,
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

        final int nCtxTrain = lib.llama_model_n_ctx_train(model);

        lib.llama_sampler_chain_add(
            _smpl,
            lib.llama_sampler_init_dry(
              vocab,
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
          _smpl,
          lib.llama_sampler_init_mirostat_v2(samplerParams.seed,
              samplerParams.mirostatTau, samplerParams.mirostatEta));
    } else if (samplerParams.mirostat == 1) {
      lib.llama_sampler_chain_add(
          _smpl,
          lib.llama_sampler_init_mirostat(
              lib.llama_n_vocab(vocab),
              samplerParams.seed,
              samplerParams.mirostatTau,
              samplerParams.mirostatEta,
              samplerParams.mirostatM));
    } else {
      lib.llama_sampler_chain_add(
          _smpl, lib.llama_sampler_init_top_k(samplerParams.topK));

      lib.llama_sampler_chain_add(
          _smpl, lib.llama_sampler_init_top_p(samplerParams.topP, 1));

      lib.llama_sampler_chain_add(
          _smpl, lib.llama_sampler_init_min_p(samplerParams.minP, 1));

      lib.llama_sampler_chain_add(
          _smpl, lib.llama_sampler_init_typical(samplerParams.typical, 1));

      if (samplerParams.dynatempRange > 0.0) {
        try {
          lib.llama_sampler_chain_add(
              _smpl,
              lib.llama_sampler_init_temp_ext(samplerParams.temp,
                  samplerParams.dynatempRange, samplerParams.dynatempExponent));
        } catch (_) {
          lib.llama_sampler_chain_add(
              _smpl, lib.llama_sampler_init_temp(samplerParams.temp));
        }
      } else {
        lib.llama_sampler_chain_add(
            _smpl, lib.llama_sampler_init_temp(samplerParams.temp));
      }

      if (samplerParams.xtcProbability > 0.0) {
        try {
          lib.llama_sampler_chain_add(
              _smpl,
              lib.llama_sampler_init_xtc(samplerParams.xtcProbability,
                  samplerParams.xtcThreshold, 1, samplerParams.seed));
        } catch (_) {}
      }
    }

    lib.llama_sampler_chain_add(
        _smpl, lib.llama_sampler_init_dist(samplerParams.seed));
  }

  void setPrompt(String prompt,
      {void Function(int current, int total)? onProgress}) {
    if (prompt.isEmpty) throw ArgumentError('Prompt cannot be empty');
    if (_isDisposed) throw StateError('Disposed');

    final nCtx = _contextParams?.nCtx ?? 2048;
    final autoTrim = _contextParams?.autoTrimContext ?? false;
    if (!autoTrim && _nPos >= nCtx - 10) {
      throw LlamaException("Context full (pos: $_nPos, limit: $nCtx)");
    }

    Pointer<Utf8>? promptUtf8Ptr;

    try {
      _status = LlamaStatus.generating;
      _pendingBytes.clear();

      if (_nPos == 0) {
        if (context.address != 0) {
          final mem = lib.llama_get_memory(context);
          lib.llama_memory_clear(mem, true);
        }
        if (_smpl != nullptr) lib.llama_sampler_reset(_smpl);
        batch.n_tokens = 0;
      }

      promptUtf8Ptr = prompt.toNativeUtf8();
      final int promptBytes = promptUtf8Ptr.length;
      final Pointer<Char> promptCharPtr = promptUtf8Ptr.cast<Char>();

      _nPrompt = -lib.llama_tokenize(
          vocab, promptCharPtr, promptBytes, nullptr, 0, true, true);

      if (_nPrompt <= 0) throw LlamaException("Token estimate failed");
      if (_nPrompt > nCtx - 10) throw LlamaException("Prompt too large");
      if (!autoTrim && _nPos + _nPrompt >= nCtx) {
        throw LlamaException("Context limit exceeded");
      }

      if (_tokens != nullptr) malloc.free(_tokens);
      _tokens = malloc<llama_token>(_nPrompt);

      final int actualTokens = lib.llama_tokenize(
          vocab, promptCharPtr, promptBytes, _tokens, _nPrompt, true, true);

      if (actualTokens < 0) throw LlamaException("Tokenization failed");
      _nPrompt = actualTokens;

      final (_, trimmed) = _maybeTrimContext(_nPrompt);
      if (trimmed) {
        batch.n_tokens = 0;
      }

      if (_nPrompt > nCtx - 10) throw LlamaException("Prompt too large");
      if (_nPos + _nPrompt >= nCtx) {
        throw LlamaException("Context limit exceeded");
      }

      int batchCapacity = _contextParams?.nBatch ?? 512;
      if (_nPrompt > batchCapacity) {
        throw LlamaException(
            "Prompt tokens ($_nPrompt) > batch capacity ($batchCapacity)");
      }

      for (int i = 0; i < _nPrompt; i++) {
        batch.token[i] = _tokens[i];
        batch.pos[i] = _nPos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = _batchSeqIds[i];
        batch.seq_id[i].value = 0;
        batch.logits[i] = i == _nPrompt - 1 ? 1 : 0;
      }
      batch.n_tokens = _nPrompt;
    } catch (e) {
      _status = LlamaStatus.error;
      if (_tokens != nullptr) {
        malloc.free(_tokens);
        _tokens = nullptr;
      }
      rethrow;
    } finally {
      if (promptUtf8Ptr != null) malloc.free(promptUtf8Ptr);
    }
  }

  String _decodeToken(int tokenId) {
    final buf = malloc<Char>(256);
    try {
      int n = lib.llama_token_to_piece(vocab, tokenId, buf, 256, 0, true);
      if (n < 0) return "";

      final newBytes = buf.cast<Uint8>().asTypedList(n);
      _pendingBytes.addAll(newBytes);

      try {
        String piece = utf8.decode(_pendingBytes);
        _pendingBytes.clear();
        return piece;
      } catch (e) {
        return "";
      }
    } finally {
      malloc.free(buf);
    }
  }

  (String, bool) getNext() {
    final result = getNextWithStatus();
    return (result.$1, result.$2);
  }

  (String, bool, bool) getNextWithStatus() {
    if (_isDisposed) throw StateError('Disposed');

    try {
      final nCtx = _contextParams?.nCtx ?? 2048;
      final tokensToAdd = batch.n_tokens == 0 ? 1 : batch.n_tokens;
      final (shifted, trimmed) = _maybeTrimContext(tokensToAdd);

      if (trimmed) {
        if (shifted > 0) {
          for (int i = 0; i < batch.n_tokens; i++) {
            batch.pos[i] = max(0, batch.pos[i] - shifted);
          }
        } else {
          for (int i = 0; i < batch.n_tokens; i++) {
            batch.pos[i] = i;
          }
        }
      }

      if (_nPos >= nCtx - 2) {
        return ("\n\n[Context limit reached]", true, true);
      }

      final nGenerated = _nPos > _nPrompt ? _nPos - _nPrompt : 0;
      if (_nPredict > 0 && nGenerated >= _nPredict) {
        return ("", true, false);
      }

      if (lib.llama_decode(context, batch) != 0) {
        if (_nPos >= nCtx - 10) return ("\n\n[Context limit]", true, true);
        throw LlamaException("Failed to eval");
      }

      _nPos += batch.n_tokens;

      int newTokenId = lib.llama_sampler_sample(_smpl, context, -1);
      bool isEos = lib.llama_token_is_eog(vocab, newTokenId);

      String piece;
      if (isEos) {
        if (_pendingBytes.isNotEmpty) {
          piece = utf8.decode(_pendingBytes, allowMalformed: true);
          _pendingBytes.clear();
        } else {
          piece = "";
        }
      } else {
        piece = _decodeToken(newTokenId);
      }

      batch.token[0] = newTokenId;
      batch.pos[0] = _nPos;
      batch.n_seq_id[0] = 1;
      batch.seq_id[0] = _batchSeqIds[0];
      batch.seq_id[0].value = 0;
      batch.logits[0] = 1;
      batch.n_tokens = 1;

      return (piece, isEos, false);
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Error generating text', e);
    }
  }

  Stream<String> generateText() async* {
    if (_isDisposed) throw StateError('Disposed');
    try {
      while (true) {
        final (text, isDone, contextLimitReached) = getNextWithStatus();
        if (text.isNotEmpty) yield text;
        if (contextLimitReached) {
          _status = LlamaStatus.ready;
          break;
        }
        if (isDone) break;
      }
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Error in text generation stream', e);
    }
  }

  Future<String> generateCompleteText({int? maxTokens}) async {
    if (_isDisposed) throw StateError('Disposed');
    final buffer = StringBuffer();
    int tokenCount = 0;
    final limit = maxTokens ?? _nPredict;

    try {
      while (true) {
        final (text, isDone, contextLimitReached) = getNextWithStatus();
        buffer.write(text);
        tokenCount++;
        if (contextLimitReached) {
          _status = LlamaStatus.ready;
          break;
        }
        if (isDone || (limit > 0 && tokenCount >= limit)) break;
      }
      return buffer.toString();
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Error generating complete text', e);
    }
  }

  bool wasContextLimitReached() {
    final nCtx = _contextParams?.nCtx ?? 2048;
    return _nPos >= nCtx - 2;
  }

  int getRemainingContextSpace() {
    final nCtx = _contextParams?.nCtx ?? 2048;
    return nCtx - _nPos;
  }

  Stream<String> generateWithMedia(
    String prompt, {
    required List<LlamaInput> inputs,
  }) async* {
    if (_isDisposed) throw StateError('Disposed');
    if (!_isVisionEnabled || _mctx == nullptr) {
      throw LlamaException('Vision disabled – construct with mmprojPath.');
    }
    if (inputs.isEmpty) throw ArgumentError('No images given');

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
      _status = LlamaStatus.generating;
      lib.llama_sampler_reset(_smpl);
      clear();

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
      final nCtx = _contextParams?.nCtx ?? 2048;
      final nChunks = lib.mtmd_input_chunks_size(chunks);

      final b = batch;
      final Pointer<llama_token> originalTokenPtr = b.token;
      final int batchCapacity = _contextParams?.nBatch ?? 512;

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
            b.seq_id[k] = _batchSeqIds[k];
            b.seq_id[k].value = 0;
            b.logits[k] = 0;
          }
          b.logits[nTok - 1] = 1;
          b.n_tokens = nTok;

          if (lib.llama_decode(context, b) != 0) {
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
            b.seq_id[k] = _batchSeqIds[k];
            b.seq_id[k].value = 0;
            b.logits[k] = 0;
          }
          b.logits[nTok - 1] = 1;
          b.n_tokens = nTok;

          if (lib.llama_decode(context, b) != 0) {
            throw LlamaException('decode text failed');
          }
          nPast += nTok;
        }
      }
      _nPos = nPast;

      var produced = 0;
      while (nPast < nCtx && (_nPredict == -1 || produced < _nPredict)) {
        b.token = originalTokenPtr;
        b.embd = nullptr;

        final tok = lib.llama_sampler_sample(_smpl, context, -1);
        bool isEos = lib.llama_token_is_eog(vocab, tok);

        String piece;
        if (isEos) {
          if (_pendingBytes.isNotEmpty) {
            piece = utf8.decode(_pendingBytes, allowMalformed: true);
            _pendingBytes.clear();
          } else {
            piece = "";
          }
        } else {
          piece = _decodeToken(tok);
        }
        if (piece.isNotEmpty) yield piece;
        if (isEos) break;

        b.n_tokens = 0;
        b.token[0] = tok;
        b.pos[0] = nPast;
        b.n_seq_id[0] = 1;
        b.seq_id[0] = _batchSeqIds[0];
        b.seq_id[0].value = 0;
        b.logits[0] = 1;
        b.n_tokens = 1;

        if (lib.llama_decode(context, b) != 0) {
          throw LlamaException('decode token failed');
        }
        ++nPast;
        ++produced;
        _nPos = nPast;
      }
    } finally {
      if (batch.seq_id != nullptr) {
        final batchCapacity = _contextParams?.nBatch ?? 512;
        for (int i = 0; i < batchCapacity; ++i) {
          batch.seq_id[i] = nullptr;
        }
      }
      if (chunks != nullptr) lib.mtmd_input_chunks_free(chunks);
      if (bmpArr != nullptr) malloc.free(bmpArr);
      for (final r in bitmapRefs) {
        if (r.bitmap != nullptr) lib.mtmd_bitmap_free(r.bitmap);
      }
      if (txtPtr != nullptr) calloc.free(txtPtr);
      if (fullPtr != nullptr) malloc.free(fullPtr);
      _status = LlamaStatus.ready;
    }
  }

  void dispose() {
    if (_isDisposed) return;
    try {
      if (_tokens != nullptr) {
        malloc.free(_tokens);
        _tokens = nullptr;
      }
      if (_tokenPtr != nullptr) {
        malloc.free(_tokenPtr);
        _tokenPtr = nullptr;
      }
      if (_smpl != nullptr) {
        lib.llama_sampler_free(_smpl);
        _smpl = nullptr;
      }

      if (_isInitialized) {
        if (_mctx != nullptr) {
          lib.mtmd_free(_mctx);
          _mctx = nullptr;
        }

        for (final slot in _slots.values) {
          if (slot.context.address != 0) lib.llama_free(slot.context);
        }
        _slots.clear();
        if (model.address != 0) lib.llama_free_model(model);

        try {
          if (batch.seq_id != nullptr) {
            final batchCapacity = _contextParams?.nBatch ?? 512;
            for (int i = 0; i < batchCapacity; ++i) {
              batch.seq_id[i] = nullptr;
            }
          }
          lib.llama_batch_free(batch);
        } catch (_) {}

        for (final ptr in _batchSeqIds) {
          calloc.free(ptr);
        }
        _batchSeqIds.clear();
      }

      if (_mctx != nullptr) {
        lib.mtmd_free(_mctx);
        _mctx = nullptr;
      }
      lib.llama_backend_free();
    } finally {
      _isDisposed = true;
      _status = LlamaStatus.disposed;
    }
  }

  void clear() {
    if (_isDisposed) throw StateError('Disposed');
    try {
      if (_tokens != nullptr) {
        malloc.free(_tokens);
        _tokens = nullptr;
      }
      _nPrompt = 0;
      _nPos = 0;
      _pendingBytes.clear();

      if (_isInitialized && context.address != 0) {
        final mem = lib.llama_get_memory(context);
        lib.llama_memory_clear(mem, true);
      }
      batch.n_tokens = 0;
      _status = LlamaStatus.ready;
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Failed to clear', e);
    }
  }

  List<int> tokenize(String text, bool addBos) {
    if (_isDisposed) throw StateError('Disposed');
    if (text.isEmpty) throw ArgumentError('Empty text');
    final utf8Ptr = text.toNativeUtf8();
    final length = utf8Ptr.length;
    final textPtr = utf8Ptr.cast<Char>();
    try {
      int nTokens =
          -lib.llama_tokenize(vocab, textPtr, length, nullptr, 0, addBos, true);
      final tokens = malloc<llama_token>(nTokens);
      try {
        int actual = lib.llama_tokenize(
            vocab, textPtr, length, tokens, nTokens, addBos, true);
        return List<int>.generate(actual, (i) => tokens[i]);
      } finally {
        malloc.free(tokens);
      }
    } finally {
      malloc.free(utf8Ptr);
    }
  }

  List<double> getEmbeddings(String prompt,
      {bool addBos = true, bool normalize = true}) {
    if (_isDisposed) throw StateError('Disposed');
    if (prompt.isEmpty) throw ArgumentError('Empty prompt');

    llama_batch? promptBatch;
    final List<Pointer<llama_seq_id>> tempSeqIds = [];
    int batchCapacity = 0;

    try {
      List<int> tokens = tokenize(prompt, addBos);
      int nTokens = tokens.length;
      int maxBatch = _contextParams?.nBatch ?? 512;
      if (nTokens > maxBatch) {
        tokens = tokens.sublist(0, maxBatch - 1);
        nTokens = tokens.length;
      }

      promptBatch = lib.llama_batch_init(nTokens, 0, 1);
      batchCapacity = nTokens;

      for (int i = 0; i < nTokens; i++) {
        promptBatch.token[i] = tokens[i];
        promptBatch.pos[i] = i;
        promptBatch.n_seq_id[i] = 1;
        final seqId = calloc<llama_seq_id>()..value = 0;
        tempSeqIds.add(seqId);
        promptBatch.seq_id[i] = seqId;
        promptBatch.logits[i] = i == nTokens - 1 ? 1 : 0;
      }
      promptBatch.n_tokens = nTokens;

      final mem = lib.llama_get_memory(context);
      lib.llama_memory_clear(mem, true);

      bool isEncoderOnly = lib.llama_model_has_encoder(model) &&
          !lib.llama_model_has_decoder(model);

      if (isEncoderOnly) {
        if (lib.llama_encode(context, promptBatch) != 0) {
          throw LlamaException("Encode failed");
        }
      } else {
        if (lib.llama_decode(context, promptBatch) != 0) {
          throw LlamaException("Decode failed");
        }
      }

      final int nEmbd = lib.llama_n_embd(model);
      Pointer<Float> embeddingsPtr;
      try {
        embeddingsPtr = lib.llama_get_embeddings_seq(context, 0);
      } catch (e) {
        try {
          embeddingsPtr = lib.llama_get_embeddings_ith(context, nTokens - 1);
        } catch (e) {
          embeddingsPtr = lib.llama_get_embeddings(context);
        }
      }
      if (embeddingsPtr == nullptr) throw LlamaException("No embeddings");

      final List<double> embeddings = List<double>.filled(nEmbd, 0.0);
      for (int i = 0; i < nEmbd; i++) {
        embeddings[i] = embeddingsPtr[i].toDouble();
      }

      if (normalize) {
        double sum = 0.0;
        for (int i = 0; i < nEmbd; i++) {
          sum += embeddings[i] * embeddings[i];
        }
        final double norm = sqrt(sum);
        if (norm > 0) {
          for (int i = 0; i < nEmbd; i++) {
            embeddings[i] /= norm;
          }
        }
      }
      return embeddings;
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Error generating embeddings', e);
    } finally {
      if (promptBatch != null) {
        if (promptBatch.seq_id != nullptr) {
          for (int i = 0; i < batchCapacity; i++) {
            promptBatch.seq_id[i] = nullptr;
          }
        }
        lib.llama_batch_free(promptBatch);
      }
      for (final ptr in tempSeqIds) {
        calloc.free(ptr);
      }
    }
  }

  /// Saves session to disk.
  /// Optimized to stream directly from C-memory to Disk to avoid RAM spikes.
  void saveSession(String path) {
    if (_isDisposed) throw StateError('Disposed');

    final int stateSize = lib.llama_get_state_size(context);
    final ptr = malloc<Uint8>(stateSize);

    try {
      lib.llama_copy_state_data(context, ptr);

      final header = ByteData(16)
        ..setUint32(0, 0x4C4C5346, Endian.little)
        ..setUint32(4, 1, Endian.little)
        ..setUint32(8, _nPos, Endian.little)
        ..setUint32(12, _nPrompt, Endian.little);

      final file = File(path);
      final raf = file.openSync(mode: FileMode.write);

      try {
        raf.writeFromSync(header.buffer.asUint8List());

        final externalView = ptr.asTypedList(stateSize);
        raf.writeFromSync(externalView);

        raf.flushSync();
      } finally {
        raf.closeSync();
      }
    } catch (e) {
      throw LlamaException('Failed to save session', e);
    } finally {
      malloc.free(ptr);
    }
  }

  /// Loads session from disk.
  /// Optimized to read directly from Disk to C-memory.
  bool loadSession(String path) {
    if (_isDisposed) throw StateError('Disposed');
    final file = File(path);
    if (!file.existsSync()) return false;

    final int expectedStateSize = lib.llama_get_state_size(context);
    final raf = file.openSync(mode: FileMode.read);
    const int headerSize = 16;

    try {
      if (raf.lengthSync() < headerSize + expectedStateSize) {
        throw LlamaException(
            'Session file corrupted or model configuration changed');
      }

      final headerBytes = raf.readSync(headerSize);
      final header = ByteData.sublistView(headerBytes);

      if (header.getUint32(0, Endian.little) != 0x4C4C5346 ||
          header.getUint32(4, Endian.little) != 1) {
        throw LlamaException('Invalid session header');
      }

      _nPos = header.getUint32(8, Endian.little);
      _nPrompt = header.getUint32(12, Endian.little);

      final ptr = malloc<Uint8>(expectedStateSize);

      try {
        final externalView = ptr.asTypedList(expectedStateSize);
        raf.readIntoSync(externalView);

        lib.llama_set_state_data(context, ptr);
      } finally {
        malloc.free(ptr);
      }

      return true;
    } catch (e) {
      throw LlamaException('Failed to load session', e);
    } finally {
      raf.closeSync();
    }
  }

  /// Saves the current state to RAM (Heap) as a byte array.
  /// Useful for fast context switching between users/chats without disk I/O.
  Uint8List saveState() {
    if (_isDisposed) throw StateError('Disposed');

    final int stateSize = lib.llama_get_state_size(context);
    const int headerSize = 16;
    final int totalSize = stateSize + headerSize;

    final ptr = malloc<Uint8>(totalSize);

    try {
      final headerData = ptr.asTypedList(headerSize).buffer.asByteData();
      headerData.setUint32(0, 0x4C4C5346, Endian.little);
      headerData.setUint32(4, 1, Endian.little);
      headerData.setUint32(8, _nPos, Endian.little);
      headerData.setUint32(12, _nPrompt, Endian.little);

      final dataPtr = Pointer<Uint8>.fromAddress(ptr.address + headerSize);
      lib.llama_copy_state_data(context, dataPtr);

      return Uint8List.fromList(ptr.asTypedList(totalSize));
    } finally {
      malloc.free(ptr);
    }
  }

  /// Restores state from a RAM byte array.
  void loadState(Uint8List stateData) {
    if (_isDisposed) throw StateError('Disposed');

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

    _nPos = header.getUint32(8, Endian.little);
    _nPrompt = header.getUint32(12, Endian.little);

    final int expectedStateSize = lib.llama_get_state_size(context);
    if (stateData.length - headerSize != expectedStateSize) {
      // ignore: avoid_print
      print(
          "Warning: State size mismatch. Expected $expectedStateSize, got ${stateData.length - headerSize}");
    }

    final ptr = malloc<Uint8>(expectedStateSize);

    try {
      final dataView = stateData.sublist(headerSize);
      ptr.asTypedList(expectedStateSize).setAll(0, dataView);

      lib.llama_set_state_data(context, ptr);
    } finally {
      malloc.free(ptr);
    }
  }
}

/// Holds the state for a specific user/conversation slot
class _LlamaSlot {
  final Pointer<llama_context> context;
  int nPos = 0;
  int nPrompt = 0;
  List<int> pendingBytes = [];

  _LlamaSlot(this.context);
}
