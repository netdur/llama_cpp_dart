import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:async';
import 'dart:math' show sqrt;
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
/// Provides text generation capabilities using the llama model.
class Llama {
  static llama_cpp? _lib;

  // Core Llama.cpp pointers
  late Pointer<llama_model> model;
  late Pointer<llama_context> context;
  late Pointer<llama_vocab> vocab;
  late llama_batch batch;

  // Helper pointers
  Pointer<llama_sampler> _smpl = nullptr;
  Pointer<llama_token> _tokens = nullptr;
  Pointer<llama_token> _tokenPtr = nullptr;
  Pointer<mtmd_context> _mctx = nullptr;

  // FIXED: Pre-allocated pool of sequence IDs to prevent memory leaks.
  // These are allocated once during initialization and freed on dispose.
  final List<Pointer<llama_seq_id>> _batchSeqIds = [];

  // FIXED: Buffer for accumulating partial UTF-8 bytes (handling split emojis/characters)
  final List<int> _pendingBytes = [];

  // State variables
  int _nPrompt = 0;
  int _nPredict = 32;
  int _nPos = 0;

  bool _verbose = false;
  bool _isInitialized = false;
  bool _isDisposed = false;
  bool _isVisionEnabled = false;
  LlamaStatus _status = LlamaStatus.uninitialized;

  static String? libraryPath = Platform.isAndroid ? "libllama.so" : null;

  /// Gets the current status of the Llama instance
  LlamaStatus get status => _status;

  /// Checks if the instance has been disposed
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

  llama_cpp getLib() {
    return _lib!;
  }

  /// Creates a new Llama instance with the specified model path.
  ///
  /// Uses named parameters for optional configuration.
  ///
  /// [modelPath] - Path to the GGUF model file.
  /// [mmprojPath] - (Optional) Path to the multimodal projector file (e.g., mmproj-model-f16.gguf).
  /// [modelParams] - (Optional) Hardware configuration (GPU layers, etc).
  /// [contextParams] - (Optional) Context window size, batch size, etc.
  /// [samplerParams] - (Optional) Sampling strategy (temperature, top_k, etc).
  /// [verbose] - (Optional) Enable verbose logging.
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

      // Initialize core Llama backend and load model/context
      _initializeLlama(
          modelPath, mmprojPath, modelParams, contextParams, samplerParams);

      // Save params for later reference
      contextParams ??= ContextParams();
      _contextParams = contextParams;
      _nPredict = contextParams.nPredict;

      _validateConfiguration();

      var nativeContextParams = contextParams.get();
      try {
        // Initialize the batch with the configured batch size
        batch = lib.llama_batch_init(nativeContextParams.n_batch, 0, 1);

        // FIXED: Pre-allocate sequence ID pointers corresponding to the batch size.
        // We do this ONCE here, instead of doing malloc/calloc inside every generation loop.
        for (int i = 0; i < nativeContextParams.n_batch; i++) {
          final seqIdPtr = calloc<llama_seq_id>();
          seqIdPtr.value = 0;
          _batchSeqIds.add(seqIdPtr);
        }
      } catch (e) {
        // If batch init fails, we must clean up what we already loaded
        if (_smpl != nullptr) lib.llama_sampler_free(_smpl);
        if (context.address != 0) lib.llama_free(context);
        if (model.address != 0) lib.llama_free_model(model);
        if (_mctx != nullptr) lib.mtmd_free(_mctx);

        // Clean up any seq_ids that might have been partially allocated
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

  void _validateConfiguration() {
    if (_nPredict == 0 || _nPredict < -1) {
      throw ArgumentError('nPredict must be positive or -1 for unlimited');
    }
  }

  static void llamaLogCallbackNull(
      int level, Pointer<Char> text, Pointer<Void> userData) {}

  void _initializeLlama(
      String modelPath,
      String? mmprojPath,
      ModelParams? modelParamsDart,
      ContextParams? contextParamsDart,
      SamplerParams? samplerParams) {
    if (!_verbose) {
      final nullCallbackPointer =
          Pointer.fromFunction<LlamaLogCallback>(Llama.llamaLogCallbackNull);
      lib.llama_log_set(nullCallbackPointer, nullptr);
    }

    lib.llama_backend_init();

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
      context = loadedContext;
    } catch (e) {
      if (loadedContext != nullptr) {
        lib.llama_free(loadedContext);
      }
      lib.llama_free_model(model);
      rethrow;
    }

    // Initialize Samplers
    _initializeSampler(samplerParams);

    // Initialize Multimodal (Vision) if path provided
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
    } else {
      final bool useMirostat2 = (samplerParams.mirostat2Tau > 0.0);
      final bool useMirostat1 =
          !useMirostat2 && (samplerParams.mirostatTau > 0.0);

      // Grammar
      final grammarStrPtr =
          samplerParams.grammarStr.toNativeUtf8().cast<Char>();
      final grammarRootPtr =
          samplerParams.grammarRoot.toNativeUtf8().cast<Char>();
      final grammar =
          lib.llama_sampler_init_grammar(vocab, grammarStrPtr, grammarRootPtr);
      if (grammar != nullptr) {
        lib.llama_sampler_chain_add(_smpl, grammar);
      }
      if (grammarStrPtr != nullptr) malloc.free(grammarStrPtr);
      if (grammarRootPtr != nullptr) malloc.free(grammarRootPtr);

      // Penalties
      lib.llama_sampler_chain_add(
          _smpl,
          lib.llama_sampler_init_penalties(
            samplerParams.penaltyLastTokens,
            samplerParams.penaltyRepeat,
            samplerParams.penaltyFreq,
            samplerParams.penaltyPresent,
          ));

      if (useMirostat2) {
        lib.llama_sampler_chain_add(
            _smpl,
            lib.llama_sampler_init_mirostat_v2(samplerParams.seed,
                samplerParams.mirostat2Tau, samplerParams.mirostat2Eta));
      } else if (useMirostat1) {
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
            _smpl,
            lib.llama_sampler_init_top_p(
                samplerParams.topP, samplerParams.topPKeep));
        lib.llama_sampler_chain_add(
            _smpl,
            lib.llama_sampler_init_min_p(
                samplerParams.minP, samplerParams.minPKeep));
        lib.llama_sampler_chain_add(
            _smpl,
            lib.llama_sampler_init_typical(
                samplerParams.typical, samplerParams.typicalKeep));
        lib.llama_sampler_chain_add(
            _smpl, lib.llama_sampler_init_temp(samplerParams.temp));
        lib.llama_sampler_chain_add(
            _smpl,
            lib.llama_sampler_init_xtc(
                samplerParams.xtcTemperature,
                samplerParams.xtcStartValue,
                samplerParams.xtcKeep,
                samplerParams.xtcLength));
      }

      lib.llama_sampler_chain_add(
          _smpl, lib.llama_sampler_init_dist(samplerParams.seed));
    }
  }

  /// Sets the prompt for text generation.
  void setPrompt(String prompt,
      {void Function(int current, int total)? onProgress}) {
    if (prompt.isEmpty) {
      throw ArgumentError('Prompt cannot be empty');
    }
    if (_isDisposed) {
      throw StateError('Cannot set prompt on disposed instance');
    }

    final nCtx = _contextParams?.nCtx ?? 2048;
    if (_nPos >= nCtx - 10) {
      throw LlamaException(
          "Context is full (position: $_nPos, limit: $nCtx). Please create a new instance or call clear().");
    }

    Pointer<Utf8>? promptUtf8Ptr;

    try {
      _status = LlamaStatus.generating;
      // FIXED: Clear pending bytes when starting a new prompt to avoid pollution
      _pendingBytes.clear();

      if (_nPos == 0) {
        if (context.address != 0) {
          final mem = lib.llama_get_memory(context);
          lib.llama_memory_clear(mem, true);
        }
        if (_smpl != nullptr) {
          lib.llama_sampler_reset(_smpl);
        }
        batch.n_tokens = 0;
      }

      promptUtf8Ptr = prompt.toNativeUtf8();
      final int promptBytes = promptUtf8Ptr.length;
      final Pointer<Char> promptCharPtr = promptUtf8Ptr.cast<Char>();

      _nPrompt = -lib.llama_tokenize(
          vocab, promptCharPtr, promptBytes, nullptr, 0, true, true);

      if (_nPrompt <= 0) {
        throw LlamaException(
            "Failed to estimate token count (returned $_nPrompt)");
      }

      if (_nPrompt > nCtx - 10) {
        throw LlamaException(
            "Prompt is too large ($_nPrompt tokens) for context size ($nCtx). Maximum prompt size is ${nCtx - 10} tokens.");
      }

      if (_nPos + _nPrompt >= nCtx) {
        throw LlamaException(
            "Adding this prompt would exceed context limit. Current position: $_nPos, prompt size: $_nPrompt, context limit: $nCtx");
      }

      if (_tokens != nullptr) {
        malloc.free(_tokens);
      }
      _tokens = malloc<llama_token>(_nPrompt);
      final int actualTokens = lib.llama_tokenize(
          vocab, promptCharPtr, promptBytes, _tokens, _nPrompt, true, true);

      if (actualTokens < 0) {
        malloc.free(_tokens);
        _tokens = nullptr;
        throw LlamaException(
            "Failed to tokenize prompt (returned $actualTokens)");
      }
      _nPrompt = actualTokens;

      int batchCapacity = _contextParams?.nBatch ?? 512;
      if (_nPrompt > batchCapacity) {
        malloc.free(_tokens);
        _tokens = nullptr;
        throw LlamaException(
            "Prompt token count ($_nPrompt) exceeds batch capacity ($batchCapacity)");
      }

      // Fill batch using pre-allocated seq_ids
      for (int i = 0; i < _nPrompt; i++) {
        batch.token[i] = _tokens[i];
        batch.pos[i] = _nPos + i;
        batch.n_seq_id[i] = 1;

        // FIXED: Use the pre-allocated pointer from our list
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
      if (promptUtf8Ptr != null) {
        malloc.free(promptUtf8Ptr);
      }
    }
  }

  /// Internal helper to decode tokens while handling split UTF-8 characters
  String _decodeToken(int tokenId) {
    final buf = malloc<Char>(256);
    try {
      // special=true allows other special tokens (like newlines) to render,
      // but we handled EOS specifically in getNextWithStatus so it won't be printed.
      int n = lib.llama_token_to_piece(vocab, tokenId, buf, 256, 0, true);
      if (n < 0) return "";

      final newBytes = buf.cast<Uint8>().asTypedList(n);
      _pendingBytes.addAll(newBytes);

      try {
        // Attempt strict decoding. If it fails (split emoji), bytes remain in buffer.
        String piece = utf8.decode(_pendingBytes);
        _pendingBytes.clear();
        return piece;
      } catch (e) {
        // Wait for next token to complete the character
        return "";
      }
    } finally {
      malloc.free(buf);
    }
  }

  /// Generates the next token in the sequence (backward compatible).
  (String, bool) getNext() {
    final result = getNextWithStatus();
    return (result.$1, result.$2);
  }

  /// Generates the next token in the sequence.
  (String, bool, bool) getNextWithStatus() {
    if (_isDisposed) {
      throw StateError('Cannot generate text on disposed instance');
    }

    try {
      final nCtx = _contextParams?.nCtx ?? 2048;

      if (_nPos >= nCtx - 2) {
        return (
          "\n\n[Context limit reached. Please start a new conversation.]",
          true,
          true
        );
      }

      final nGenerated = _nPos > _nPrompt ? _nPos - _nPrompt : 0;
      if (_nPredict > 0 && nGenerated >= _nPredict) {
        return ("", true, false);
      }

      if (lib.llama_decode(context, batch) != 0) {
        if (_nPos >= nCtx - 10) {
          return ("\n\n[Context limit reached during decode.]", true, true);
        }
        throw LlamaException("Failed to eval");
      }

      _nPos += batch.n_tokens;

      int newTokenId = lib.llama_sampler_sample(_smpl, context, -1);
      bool isEos = lib.llama_token_is_eog(vocab, newTokenId);

      String piece;

      if (isEos) {
        // FIX: If it is EOS, DO NOT decode the token itself (which would be "<end_of_turn>").
        // Instead, just flush any pending partial UTF-8 bytes from previous tokens.
        if (_pendingBytes.isNotEmpty) {
          piece = utf8.decode(_pendingBytes, allowMalformed: true);
          _pendingBytes.clear();
        } else {
          piece = "";
        }
      } else {
        // Normal decoding for non-EOS tokens
        piece = _decodeToken(newTokenId);
      }

      // Prepare batch for next iteration
      batch.token[0] = newTokenId;
      batch.pos[0] = _nPos;
      batch.n_seq_id[0] = 1;

      // Use pre-allocated pointer
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
    if (_isDisposed) {
      throw StateError('Cannot generate text on disposed instance');
    }

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
    if (_isDisposed) {
      throw StateError('Cannot generate text on disposed instance');
    }

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

        if (isDone || (limit > 0 && tokenCount >= limit)) {
          break;
        }
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

  /// Multimodal generation.
  Stream<String> generateWithMedia(
    String prompt, {
    required List<LlamaInput> inputs,
  }) async* {
    if (_isDisposed) {
      throw StateError('Instance disposed');
    }
    if (!_isVisionEnabled || _mctx == nullptr) {
      throw LlamaException(
          'Vision disabled – construct Llama with mmprojPath.');
    }
    if (inputs.isEmpty) {
      throw ArgumentError('No images given; for text only use setPrompt().');
    }

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
      clear(); // Resets batch tokens and pending bytes

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
            throw LlamaException('encode image chunk #$i failed');
          }

          final embd = lib.mtmd_get_output_embd(_mctx);
          final nTok = lib.mtmd_input_chunk_get_n_tokens(chunk);

          if (nTok > batchCapacity) {
            throw LlamaException(
                'Image chunk size ($nTok) exceeds batch capacity ($batchCapacity)');
          }
          if (nPast + nTok > nCtx) throw LlamaException('n_ctx overflow');

          b.token = nullptr;
          b.embd = embd;

          for (var k = 0; k < nTok; ++k) {
            b.pos[k] = nPast + k;
            b.n_seq_id[k] = 1;

            // FIXED: Use pre-allocated pointers
            b.seq_id[k] = _batchSeqIds[k];
            b.seq_id[k].value = 0;

            b.logits[k] = 0;
          }
          b.logits[nTok - 1] = 1;
          b.n_tokens = nTok;

          if (lib.llama_decode(context, b) != 0) {
            throw LlamaException('llama_decode image chunk #$i failed');
          }
          nPast += nTok;
        } else {
          // Text chunk
          final nPtr = malloc<Size>();
          final tokPt = lib.mtmd_input_chunk_get_tokens_text(chunk, nPtr);
          final nTok = nPtr.value;
          malloc.free(nPtr);

          if (nTok > batchCapacity) {
            throw LlamaException(
                'Text chunk size ($nTok) exceeds batch capacity ($batchCapacity)');
          }
          if (nPast + nTok > nCtx) throw LlamaException('n_ctx overflow');

          b.token = originalTokenPtr;
          b.embd = nullptr;

          for (var k = 0; k < nTok; ++k) {
            b.token[k] = tokPt[k];
            b.pos[k] = nPast + k;
            b.n_seq_id[k] = 1;

            // FIXED: Use pre-allocated pointers
            b.seq_id[k] = _batchSeqIds[k];
            b.seq_id[k].value = 0;

            b.logits[k] = 0;
          }
          b.logits[nTok - 1] = 1;
          b.n_tokens = nTok;

          if (lib.llama_decode(context, b) != 0) {
            throw LlamaException('llama_decode text chunk #$i failed');
          }
          nPast += nTok;
        }
      }

      _nPos = nPast; // Sync global position

      var produced = 0;
      // Loop to generate response tokens
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

        // FIXED: Use pre-allocated pointer
        b.seq_id[0] = _batchSeqIds[0];
        b.seq_id[0].value = 0;

        b.logits[0] = 1;
        b.n_tokens = 1;

        if (lib.llama_decode(context, b) != 0) {
          throw LlamaException('llama_decode generated token failed');
        }

        ++nPast;
        ++produced;
        _nPos = nPast;
      }
    } finally {
      // Safety reset of batch pointers (optional but good practice)
      if (batch.seq_id != nullptr) {
        final batchCapacity = _contextParams?.nBatch ?? 512;
        for (int i = 0; i < batchCapacity; ++i) {
          batch.seq_id[i] = nullptr;
        }
      }

      // Clean up only the locally allocated resources
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

  /// Disposes of all resources held by this instance
  /// Disposes of all resources held by this instance
  void dispose() {
    if (_isDisposed) return;

    try {
      // 1. Free temporary buffers
      if (_tokens != nullptr) {
        malloc.free(_tokens);
        _tokens = nullptr;
      }
      if (_tokenPtr != nullptr) {
        malloc.free(_tokenPtr);
        _tokenPtr = nullptr;
      }

      // 2. Free Sampler Chain
      if (_smpl != nullptr) {
        lib.llama_sampler_free(_smpl);
        _smpl = nullptr;
      }

      if (_isInitialized) {
        // 3. Free Context and Model first (standard order)
        if (context.address != 0) {
          lib.llama_free(context);
        }
        if (model.address != 0) {
          lib.llama_free_model(model);
        }

        // 4. SAFE BATCH CLEANUP
        // We must detach our custom seq_id pointers from the C struct
        // BEFORE freeing the batch, to prevent llama_batch_free from
        // seeing potentially invalid pointers if we freed them first.
        try {
          if (batch.seq_id != nullptr) {
            final batchCapacity = _contextParams?.nBatch ?? 512;
            for (int i = 0; i < batchCapacity; ++i) {
              // Set the pointer in C memory to NULL so llama_batch_free ignores it
              batch.seq_id[i] = nullptr;
            }
          }

          // Now it's safe to let llama.cpp free the batch arrays
          lib.llama_batch_free(batch);
        } catch (e) {
          // Ignore errors during batch free to ensure we continue cleanup
        }

        // 5. NOW we free our custom sequence ID pointers
        for (final ptr in _batchSeqIds) {
          calloc.free(ptr);
        }
        _batchSeqIds.clear();
      }

      if (_mctx != nullptr) {
        // lib.mtmd_free(_mctx);
        _mctx = nullptr;
      }

      lib.llama_backend_free();
    } finally {
      _isDisposed = true;
      _status = LlamaStatus.disposed;
    }
  }

  /// Clears the current state of the Llama instance
  void clear() {
    if (_isDisposed) {
      throw StateError('Cannot clear disposed instance');
    }

    try {
      if (_tokens != nullptr) {
        malloc.free(_tokens);
        _tokens = nullptr;
      }
      _nPrompt = 0;
      _nPos = 0;
      // FIXED: Clear pending UTF-8 bytes
      _pendingBytes.clear();

      if (_isInitialized && context.address != 0) {
        final mem = lib.llama_get_memory(context);
        lib.llama_memory_clear(mem, true);
      }

      // Reset batch state (seq_id pointers are in _batchSeqIds and don't need freeing here)
      batch.n_tokens = 0;

      _status = LlamaStatus.ready;
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Failed to clear Llama state', e);
    }
  }

  /// Converts a text string to a list of token IDs
  List<int> tokenize(String text, bool addBos) {
    if (_isDisposed) {
      throw StateError('Cannot tokenize with disposed instance');
    }

    if (text.isEmpty) {
      throw ArgumentError('Text cannot be empty');
    }

    try {
      final textPtr = text.toNativeUtf8().cast<Char>();

      try {
        int nTokens = -lib.llama_tokenize(
            vocab, textPtr, text.length, nullptr, 0, addBos, true);

        if (nTokens <= 0) {
          throw LlamaException('Failed to determine token count');
        }

        final tokens = malloc<llama_token>(nTokens);

        try {
          int actualTokens = lib.llama_tokenize(
              vocab, textPtr, text.length, tokens, nTokens, addBos, true);

          if (actualTokens < 0) {
            throw LlamaException('Tokenization failed');
          }
          return List<int>.generate(actualTokens, (i) => tokens[i]);
        } finally {
          malloc.free(tokens);
        }
      } finally {
        malloc.free(textPtr);
      }
    } catch (e) {
      throw LlamaException('Error during tokenization', e);
    }
  }

  /// Generates embeddings for the given prompt.
List<double> getEmbeddings(String prompt,
      {bool addBos = true, bool normalize = true}) {
    if (_isDisposed) {
      throw StateError('Cannot generate embeddings on disposed instance');
    }

    if (prompt.isEmpty) {
      throw ArgumentError('Prompt cannot be empty');
    }

    llama_batch? promptBatch;
    // Track temporary seq_ids specifically for this method
    final List<Pointer<llama_seq_id>> tempSeqIds = [];
    // Track batch size for cleanup
    int batchCapacity = 0;

    try {
      // Tokenize the input text
      List<int> tokens = tokenize(prompt, addBos);
      int nTokens = tokens.length;

      // Check if token count exceeds global batch size preference
      int maxBatch = _contextParams?.nBatch ?? 512;
      if (nTokens > maxBatch) {
        tokens = tokens.sublist(0, maxBatch - 1);
        nTokens = tokens.length;
      }

      // Create a batch for the tokens
      promptBatch = lib.llama_batch_init(nTokens, 0, 1);
      batchCapacity = nTokens;

      // Setup the batch with the tokens
      for (int i = 0; i < nTokens; i++) {
        promptBatch.token[i] = tokens[i];
        promptBatch.pos[i] = i; 
        promptBatch.n_seq_id[i] = 1;
        
        // Allocate specific seq_ids for this embedding batch
        final seqId = calloc<llama_seq_id>();
        seqId.value = 0;
        tempSeqIds.add(seqId);
        
        // Assign to batch
        promptBatch.seq_id[i] = seqId;
        
        promptBatch.logits[i] = i == nTokens - 1 ? 1 : 0; 
      }
      promptBatch.n_tokens = nTokens;

      final mem = lib.llama_get_memory(context);
      lib.llama_memory_clear(mem, true);

      // Process the batch
      bool isEncoderOnly = false;
      isEncoderOnly = lib.llama_model_has_encoder(model) &&
          !lib.llama_model_has_decoder(model);

      if (isEncoderOnly) {
        if (lib.llama_encode(context, promptBatch) != 0) {
          throw LlamaException("Failed to encode prompt for embeddings");
        }
      } else {
        if (lib.llama_decode(context, promptBatch) != 0) {
          throw LlamaException("Failed to decode prompt for embeddings");
        }
      }

      // Get the embeddings
      final int nEmbd = lib.llama_n_embd(model);
      Pointer<Float> embeddingsPtr;

      try {
        // First try sequence embeddings
        embeddingsPtr = lib.llama_get_embeddings_seq(context, 0);
      } catch (e) {
        try {
          // Then try last token embeddings
          embeddingsPtr = lib.llama_get_embeddings_ith(context, nTokens - 1);
        } catch (e) {
          // Finally fall back to default embeddings
          embeddingsPtr = lib.llama_get_embeddings(context);
        }
      }

      if (embeddingsPtr == nullptr) {
        throw LlamaException("Failed to get embeddings");
      }

      // Convert to Dart list
      final List<double> embeddings = List<double>.filled(nEmbd, 0.0);
      for (int i = 0; i < nEmbd; i++) {
        embeddings[i] = embeddingsPtr[i].toDouble();
      }

      // Normalize if requested
      if (normalize) {
        double sum = 0.0;
        for (int i = 0; i < nEmbd; i++) {
          sum += embeddings[i] * embeddings[i];
        }
        final double norm = sqrt(sum);
        if (norm > 0) {
          for (int i = 0; i < nEmbd; i++) {
            embeddings[i] = embeddings[i] / norm;
          }
        }
      }

      return embeddings;
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Error generating embeddings', e);
    } finally {
      if (promptBatch != null) {
        // 1. CRITICAL FIX: Detach our pointers from the batch C-struct
        // before the library tries to free the batch arrays.
        if (promptBatch.seq_id != nullptr) {
          for (int i = 0; i < batchCapacity; i++) {
            promptBatch.seq_id[i] = nullptr;
          }
        }
        
        // 2. Now safe to free the batch structure
        lib.llama_batch_free(promptBatch);
      }
      
      // 3. Finally, free the actual sequence ID memory we allocated
      for (final ptr in tempSeqIds) {
        calloc.free(ptr);
      }
    }
  }

  /// Loads a saved session state
  bool loadSession(String path) {
    final bytes = File(path).readAsBytesSync();
    final hdr = ByteData.sublistView(bytes, 0, 12);
    final magic = hdr.getUint32(0, Endian.little);
    final version = hdr.getUint32(4, Endian.little);
    if (magic != 0x4C4C5346 || version != 1) {
      throw LlamaException('Bad session header');
    }
    _nPos = hdr.getUint32(8, Endian.little);

    final stateBytes = bytes.sublist(12);
    final ptr = malloc<Uint8>(stateBytes.length)
      ..asTypedList(stateBytes.length).setAll(0, stateBytes);

    lib.llama_set_state_data(context, ptr);
    malloc.free(ptr);
    return true;
  }

  /// Saves the current session state
  void saveSession(String path) {
    final size = lib.llama_get_state_size(context);
    final buf = malloc<Uint8>(size);
    lib.llama_copy_state_data(context, buf);

    final bytes = buf.asTypedList(size);

    final header = ByteData(12)
      ..setUint32(0, 0x4C4C5346, Endian.little) // "F S L L" magic
      ..setUint32(4, 1, Endian.little) // version
      ..setUint32(8, _nPos, Endian.little); // position

    final out = BytesBuilder()
      ..add(header.buffer.asUint8List())
      ..add(bytes);

    File(path).writeAsBytesSync(out.toBytes(), flush: true);
    malloc.free(buf);
  }
}
