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
// enum LlamaStatus { initializing, uninitialized, ready, generating, error, disposed }
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
  late Pointer<llama_model> model;
  late Pointer<llama_context> context;
  late Pointer<llama_vocab> vocab;
  late llama_batch batch;

  Pointer<llama_sampler> _smpl = nullptr;
  Pointer<llama_token> _tokens = nullptr;
  Pointer<llama_token> _tokenPtr = nullptr;
  int _nPrompt = 0;
  int _nPredict = 32;
  int _nPos = 0;

  bool _verbos = false;
  bool _isInitialized = false; // Track if initialization is complete

  bool _isDisposed = false;
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

  Pointer<mtmd_context> _mctx = nullptr;
  bool _isVisionEnabled = false;

  /// Creates a new Llama instance with the specified model path and optional parameters.
  ///
  /// Throws [LlamaException] if model loading or initialization fails.
  Llama(String modelPath,
      [ModelParams? modelParamsDart,
      ContextParams? contextParamsDart,
      SamplerParams? samplerParams,
      bool? verbos,
      String? mmprojPath]) {
    try {
      _verbos = verbos ?? false;
      // _verbos = true;
      _validateConfiguration();
      _initializeLlama(modelPath, mmprojPath, modelParamsDart,
          contextParamsDart, samplerParams);

      // Always initialize the batch, even if contextParamsDart is null
      contextParamsDart ??= ContextParams();
      var contextParams = contextParamsDart.get();
      batch = lib.llama_batch_init(contextParams.n_batch, 0, 1);

      _isInitialized = true; // Mark initialization as complete
      _status = LlamaStatus.ready;
    } catch (e) {
      _status = LlamaStatus.error;
      dispose(); // Dispose resources on initialization failure
      throw LlamaException('Failed to initialize Llama', e);
    }
  }

  /// Validates the configuration parameters
  void _validateConfiguration() {
    // Allow nPredict = -1 for unlimited generation,
    // but disallow 0 or other negative values.
    if (_nPredict == 0 || _nPredict < -1) {
      throw ArgumentError('nPredict must be positive or -1 for unlimited');
    }
    // No error if _nPredict > 0 or _nPredict == -1
  }

  static void llamaLogCallbackNull(
      int level, Pointer<Char> text, Pointer<Void> userData) {}

  /// Initializes the Llama instance with the given parameters
  void _initializeLlama(
      String modelPath,
      String? mmprojPath,
      ModelParams? modelParamsDart,
      ContextParams? contextParamsDart,
      SamplerParams? samplerParams) {
    if (_verbos == false) {
      final nullCallbackPointer =
          Pointer.fromFunction<LlamaLogCallback>(Llama.llamaLogCallbackNull);
      lib.llama_log_set(nullCallbackPointer, nullptr);
    }

    lib.llama_backend_init();

    modelParamsDart ??= ModelParams();
    var modelParams = modelParamsDart.get();

    final modelPathPtr = modelPath.toNativeUtf8().cast<Char>();
    Pointer<llama_model> loadedModel = nullptr; // Use a local variable
    try {
      loadedModel = lib.llama_load_model_from_file(modelPathPtr, modelParams);
      if (loadedModel == nullptr) {
        throw LlamaException("Could not load model at $modelPath");
      }
      model = loadedModel; // Assign to the class member after the check
      vocab = lib.llama_model_get_vocab(
          model); // Get the vocab *after* model is loaded.
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

    samplerParams ??= SamplerParams();
    llama_sampler_chain_params sparams =
        lib.llama_sampler_chain_default_params();
    sparams.no_perf = false;
    _smpl = lib.llama_sampler_chain_init(sparams);

    if (samplerParams.greedy) {
      lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_greedy());
    }

    lib.llama_sampler_chain_add(
        _smpl, lib.llama_sampler_init_dist(samplerParams.seed));

    if (samplerParams.softmax) {
      lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_softmax());
    }

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

    lib.llama_sampler_chain_add(
        _smpl,
        lib.llama_sampler_init_mirostat(
            lib.llama_n_vocab(vocab),
            samplerParams.seed,
            samplerParams.mirostatTau,
            samplerParams.mirostatEta,
            samplerParams.mirostatM));

    lib.llama_sampler_chain_add(
        _smpl,
        lib.llama_sampler_init_mirostat_v2(samplerParams.seed,
            samplerParams.mirostat2Tau, samplerParams.mirostat2Eta));

    final grammarStrPtr = samplerParams.grammarStr.toNativeUtf8().cast<Char>();
    final grammarRootPtr =
        samplerParams.grammarRoot.toNativeUtf8().cast<Char>();

    final grammar =
        lib.llama_sampler_init_grammar(vocab, grammarStrPtr, grammarRootPtr);
    if (grammar != nullptr) {
      lib.llama_sampler_chain_add(_smpl, grammar);
    }

    if (grammarStrPtr != nullptr) malloc.free(grammarStrPtr);
    if (grammarRootPtr != nullptr) malloc.free(grammarRootPtr);

    lib.llama_sampler_chain_add(
        _smpl,
        lib.llama_sampler_init_penalties(
          samplerParams.penaltyLastTokens,
          samplerParams.penaltyRepeat,
          samplerParams.penaltyFreq,
          samplerParams.penaltyPresent,
        ));

    final seqBreakers = samplerParams.dryBreakers;
    final numBreakers = seqBreakers.length;
    final seqBreakersPointer = calloc<Pointer<Char>>(numBreakers);

    try {
      for (var i = 0; i < numBreakers; i++) {
        seqBreakersPointer[i] = seqBreakers[i].toNativeUtf8().cast<Char>();
      }

      lib.llama_sampler_chain_add(
          _smpl,
          lib.llama_sampler_init_penalties(
            samplerParams.penaltyLastTokens,
            samplerParams.penaltyRepeat,
            samplerParams.penaltyFreq,
            samplerParams.penaltyPresent,
          ));
    } finally {
      for (var i = 0; i < numBreakers; i++) {
        malloc.free(seqBreakersPointer[i]);
      }
      calloc.free(seqBreakersPointer);
    }

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

  /// Sets the prompt for text generation.
  ///
  /// [prompt] - The input prompt text
  /// [onProgress] - Optional callback for tracking tokenization progress
  ///
  /// Throws [ArgumentError] if prompt is empty
  /// Throws [LlamaException] if tokenization fails
  void setPrompt(String prompt,
      {void Function(int current, int total)? onProgress}) {
    if (prompt.isEmpty) {
      throw ArgumentError('Prompt cannot be empty');
    }
    if (_isDisposed) {
      throw StateError('Cannot set prompt on disposed instance');
    }

    Pointer<Utf8>? promptUtf8Ptr;

    try {
      _status = LlamaStatus.generating;

      if (_nPos == 0) {
        if (context.address != 0) {
          final mem = lib.llama_get_memory(context);
          lib.llama_memory_clear(mem, true);
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

      for (int i = 0; i < _nPrompt; i++) {
        batch.token[i] = _tokens[i];
        batch.pos[i] = _nPos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = calloc<llama_seq_id>()..value = 0;
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

  /// Generates the next token in the sequence.
  ///
  /// Returns a tuple containing the generated text and a boolean indicating if generation is complete.
  /// Throws [LlamaException] if generation fails.
  (String, bool) getNext() {
    if (_isDisposed) {
      throw StateError('Cannot generate text on disposed instance');
    }

    try {
      final nGenerated = _nPos > _nPrompt ? _nPos - _nPrompt : 0;
      if (_nPredict > 0 && nGenerated >= _nPredict) {
        return ("", true);
      }

      if (lib.llama_decode(context, batch) != 0) {
        throw LlamaException("Failed to eval");
      }
      _nPos += batch.n_tokens;
      int newTokenId = lib.llama_sampler_sample(_smpl, context, -1);

      if (lib.llama_token_is_eog(vocab, newTokenId)) {
        return ("", true);
      }

      final buf = malloc<Char>(128);
      try {
        int n = lib.llama_token_to_piece(vocab, newTokenId, buf, 128, 0, true);
        if (n < 0) {
          throw LlamaException("Failed to convert token to piece");
        }
        // String piece = utf8.decode(buf.cast<Uint8>().asTypedList(n));
        String piece = '';
        final bytes = buf.cast<Uint8>().asTypedList(n);
        try {
          piece = utf8.decode(bytes);
        } catch (e) {
          piece = utf8.decode(bytes, allowMalformed: true);
        }

        batch.token[0] = newTokenId;
        batch.pos[0] = _nPos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0].value = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;

        bool isEos = newTokenId == lib.llama_token_eos(vocab);
        return (piece, isEos);
      } finally {
        malloc.free(buf);
      }
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Error generating text', e);
    }
  }

  /// Provides a stream of generated text tokens
  Stream<String> generateText() async* {
    if (_isDisposed) {
      throw StateError('Cannot generate text on disposed instance');
    }

    try {
      while (true) {
        final (text, isDone) = getNext();
        if (isDone) break;
        yield text;
      }
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Error in text generation stream', e);
    }
  }

  /// This is the primary method for multimodal generation. It is a stateless
  /// operation that processes the entire input and generates a response stream.
  ///
  /// [prompt]: The text part of the prompt. It must contain the `<image>` marker
  ///          for each image provided in the `inputs` list.
  /// [inputs]: A list of `LlamaInput` objects (currently `LlamaImage`). The
  ///          number of images must match the number of `<image>` markers.
  ///
  /// Returns a stream of generated text tokens.
  /// Generates a response based on a prompt that includes text and images.
  /// Generates a response from a prompt that combines text and one or more images.
  /// The prompt must contain a `<image>` placeholder for every image in `inputs`.
  Stream<String> generateWithMeda(
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

      // Clear KV cache and reset batch state for a clean run.
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
          if (nPast + nTok > nCtx) throw LlamaException('n_ctx overflow');

          b.token = nullptr;
          b.embd = embd;

          for (var k = 0; k < nTok; ++k) {
            b.pos[k] = nPast + k;
            b.n_seq_id[k] = 1;
            b.seq_id[k] = calloc<llama_seq_id>()..value = 0; // <<< FIX
            b.logits[k] = 0;
          }
          b.logits[nTok - 1] = 1;
          b.n_tokens = nTok;

          if (lib.llama_decode(context, b) != 0) {
            throw LlamaException('llama_decode image chunk #$i failed');
          }
          nPast += nTok;

          for (var k = 0; k < nTok; ++k) {
            calloc.free(b.seq_id[k]);
            b.seq_id[k] = nullptr;
          }
        } else {
          // Text chunk
          final nPtr = malloc<Size>();
          final tokPt = lib.mtmd_input_chunk_get_tokens_text(chunk, nPtr);
          final nTok = nPtr.value;
          malloc.free(nPtr);
          if (nPast + nTok > nCtx) throw LlamaException('n_ctx overflow');

          b.token = originalTokenPtr;
          b.embd = nullptr;

          for (var k = 0; k < nTok; ++k) {
            b.token[k] = tokPt[k];
            b.pos[k] = nPast + k;
            b.n_seq_id[k] = 1;
            b.seq_id[k] = calloc<llama_seq_id>()..value = 0; // <<< FIX
            b.logits[k] = 0;
          }
          b.logits[nTok - 1] = 1;
          b.n_tokens = nTok;

          if (lib.llama_decode(context, b) != 0) {
            throw LlamaException('llama_decode text chunk #$i failed');
          }
          nPast += nTok;

          for (var k = 0; k < nTok; ++k) {
            calloc.free(b.seq_id[k]);
            b.seq_id[k] = nullptr;
          }
        }
      }

      var produced = 0;
      while (nPast < nCtx && (_nPredict == -1 || produced < _nPredict)) {
        b.token = originalTokenPtr;
        b.embd = nullptr;

        final tok = lib.llama_sampler_sample(_smpl, context, -1);
        if (lib.llama_token_is_eog(vocab, tok)) {
          break;
        }

        final buf = malloc<Char>(128);
        try {
          final n = lib.llama_token_to_piece(vocab, tok, buf, 128, 0, false);
          if (n < 0) throw LlamaException('token_to_piece failed ($tok)');
          yield utf8.decode(buf.cast<Uint8>().asTypedList(n));
        } finally {
          malloc.free(buf);
        }

        b.n_tokens = 0;
        b.token[0] = tok;
        b.pos[0] = nPast;
        b.n_seq_id[0] = 1;
        b.seq_id[0] = calloc<llama_seq_id>()..value = 0; // <<< FIX
        b.logits[0] = 1;
        b.n_tokens = 1;

        if (lib.llama_decode(context, b) != 0) {
          throw LlamaException('llama_decode generated token failed');
        }

        calloc.free(b.seq_id[0]);
        b.seq_id[0] = nullptr;

        ++nPast;
        ++produced;
      }
    } finally {
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
  void dispose() {
    if (_isDisposed) return;
    if (_tokens != nullptr) malloc.free(_tokens);
    if (_tokenPtr != nullptr) malloc.free(_tokenPtr);
    if (_smpl != nullptr) lib.llama_sampler_free(_smpl);

    // Only access late fields if initialization was completed
    if (_isInitialized) {
      if (context.address != 0) lib.llama_free(context);
      if (model.address != 0) lib.llama_free_model(model);

      // Free the batch only if it was initialized
      try {
        lib.llama_batch_free(batch);
      } catch (e) {
        // Batch not initialized, ignore
      }
    }

    // if (_mctx != nullptr) lib.mtmd_free(_mctx); // <- crash - double free

    lib.llama_backend_free();

    _isDisposed = true;
    _status = LlamaStatus.disposed;
  }

  /// Clears the current state of the Llama instance
  /// This allows reusing the same instance for a new generation
  /// without creating a new instance
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

      if (_isInitialized && context.address != 0) {
        final mem = lib.llama_get_memory(context);
        lib.llama_memory_clear(mem, true);
      }

      if (batch.seq_id != nullptr) {
        int batchCapacity = _contextParams?.nBatch ?? 0;
        if (batchCapacity > 0) {
          for (int i = 0; i < batchCapacity; ++i) {
            if (batch.seq_id[i] != nullptr) {
              calloc.free(batch.seq_id[i]);
              batch.seq_id[i] = nullptr; // Set to null as batch is reused
            }
          }
        }
      }
      batch.n_tokens = 0;

      _status = LlamaStatus.ready;
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Failed to clear Llama state', e);
    }
  }

  /// Converts a text string to a list of token IDs
  ///
  /// [text] - The input text to tokenize
  /// [addBos] - Whether to add the beginning-of-sequence token
  ///
  /// Returns a List of integer token IDs
  ///
  /// Throws [ArgumentError] if text is empty
  /// Throws [LlamaException] if tokenization fails
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
  ///
  /// [prompt] - The input text for which to generate embeddings.
  /// [addBos] - Whether to add the beginning-of-sequence token.
  /// [normalize] - Whether to normalize the embeddings (default: true).
  ///
  /// Returns a List of floats representing the embedding.
  List<double> getEmbeddings(String prompt,
      {bool addBos = true, bool normalize = true}) {
    if (_isDisposed) {
      throw StateError('Cannot generate embeddings on disposed instance');
    }

    if (prompt.isEmpty) {
      throw ArgumentError('Prompt cannot be empty');
    }

    llama_batch? promptBatch;

    try {
      // Tokenize the input text
      List<int> tokens = tokenize(prompt, addBos);
      int nTokens = tokens.length;

      // Check if token count exceeds batch size
      int batchSize = _contextParams?.nBatch ?? 512;
      if (nTokens > batchSize) {
        tokens = tokens.sublist(0, batchSize - 1);
        nTokens = tokens.length;
      }

      // Create a batch for the tokens
      promptBatch = lib.llama_batch_init(nTokens, 0, 1);

      // Setup the batch with the tokens
      for (int i = 0; i < nTokens; i++) {
        promptBatch.token[i] = tokens[i];
        promptBatch.pos[i] = i; // Use position within sequence
        promptBatch.n_seq_id[i] = 1;
        promptBatch.seq_id[i] = calloc<llama_seq_id>()..value = 0;
        promptBatch.logits[i] =
            i == nTokens - 1 ? 1 : 0; // Set logits flag for last token
      }
      promptBatch.n_tokens = nTokens;

      // Clear the KV cache
      // lib.llama_kv_cache_clear(context);
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
      // Clean up in finally block to ensure it happens even if there's an exception
      if (promptBatch != null) {
        // Free sequence IDs
        for (int i = 0; i < promptBatch.n_tokens; i++) {
          if (promptBatch.seq_id[i] != nullptr) {
            calloc.free(promptBatch.seq_id[i]);
            promptBatch.seq_id[i] = nullptr; // Set to nullptr after freeing
          }
        }
        lib.llama_batch_free(promptBatch);
      }
    }
  }

  /// season management

  /// Loads a saved session state
  /// Returns true if successful, false if session file doesn't exist
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
      ..setUint32(
          0, 0x4C4C5346, Endian.little) // "F S L L" magic, pick anything
      ..setUint32(4, 1, Endian.little) // version
      ..setUint32(8, _nPos, Endian.little); // position

    final out = BytesBuilder()
      ..add(header.buffer.asUint8List())
      ..add(bytes);

    File(path).writeAsBytesSync(out.toBytes(), flush: true);
    malloc.free(buf);
  }

  /// Ensures there is enough space in the context for a new batch of tokens.
  /// If the context is full, it removes older tokens to make space, preserving
  /// the first `_nKeep` tokens if specified.
  // ignore: unused_element
  void _ensureContextHasSpace(int tokensInBatch, int keep) {
    final nCtx = _contextParams?.nCtx ?? 2048;

    if (_nPos + tokensInBatch > nCtx) {
      final tokensToRemove = (_nPos + tokensInBatch) - nCtx;

      if (tokensToRemove <= 0) {
        return;
      }

      final removalStartPos = keep;
      final removalEndPos = keep + tokensToRemove;
      lib.llama_kv_self_seq_rm(context, 0, removalStartPos, removalEndPos);
      lib.llama_kv_self_seq_add(
          context, 0, removalEndPos, _nPos, -tokensToRemove);
      _nPos -= tokensToRemove;
    }
  }
}
