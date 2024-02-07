import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:llama_cpp_dart/src/sampling_context.dart';
import 'package:llama_cpp_dart/src/sampling_params.dart';
import 'model_params.dart';

import 'llama_cpp.dart';
import 'context_params.dart';

/// The `Llama` class provides an interface to interact with the Llama model.
///
/// It includes methods for loading and managing the model and context,
/// as well as utilities for tokenization and decoding.
class Llama {
  static llama_cpp? _lib;

  /// Pointer to the Llama model.
  late Pointer<llama_model> model;

  /// Pointer to the Llama context.
  late Pointer<llama_context> context;

  /// Batch configuration for the Llama model.
  late llama_batch batch;

  /// List of tokens generated from the input prompt.
  List<int> tokensList = [];

  /// Temporary storage for invalid C characters.
  List<int> lastTokens = [];

  /// Length of the ouput. Default is -1.
  int length = -1;

  /// Cursor position in the token list. Default is 0.
  int cursor = 0;

  /// Counter for decoding operations. Default is 0.
  int decode = 0;

  /// set llama.cpp library path
  static String? libraryPath;

  /// Getter for the Llama library.
  ///
  /// Loads the library based on the current platform.
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

  llama_cpp get l => _lib!;

  ContextParams contextParams;
  ModelParams modelParams;
  SamplingParams? samplingParams;

  String loraBase;
  List<(String, double)> loraAdapters;

  /// Constructor for Llama.
  ///
  /// Loads the model and context based on provided model and context parameters.
  Llama(String modelPath,
      [ModelParams? modelParams,
      ContextParams? contextParams,
      SamplingParams? samplingParams,
      // this.samplingParams,
      this.loraBase = "",
      this.loraAdapters = const []])
      : modelParams = modelParams ?? ModelParams(),
        contextParams = contextParams ?? ContextParams(),
        samplingParams = samplingParams ?? SamplingParams() {
    lib.llama_backend_init(false);
    llama_model_params modelParams = this.modelParams.get();

    Pointer<Char> char = modelPath.toNativeUtf8().cast<Char>();
    model = lib.llama_load_model_from_file(char, modelParams);
    malloc.free(char);

    if (model.address == 0) {
      throw Exception("Could not load model at $modelPath");
    }

    llama_context_params contextParams = this.contextParams.get();
    context = lib.llama_new_context_with_model(model, contextParams);
    if (context.address == 0) {
      throw Exception("Could not load context!");
    }

    batch = lib.llama_batch_init(this.contextParams.batch, 0, 1);

    Pointer<Char> cLoraBase = loraBase.toNativeUtf8().cast<Char>();
    for (int i = 0; i < loraAdapters.length; i++) {
      Pointer<Char> loraAdapter =
          loraAdapters[i].$1.toNativeUtf8().cast<Char>();
      double loraScale = loraAdapters[i].$2;
      int err = lib.llama_model_apply_lora_from_file(
          model,
          loraAdapter,
          loraScale,
          loraBase.isNotEmpty ? cLoraBase : nullptr,
          this.contextParams.threads);
      malloc.free(loraAdapter);
      if (err != 0) {
        lib.llama_batch_free(batch);
        lib.llama_free(context);
        lib.llama_free_model(model);
        throw Exception("failed to apply lora adapter");
      }
    }
    malloc.free(cLoraBase);
  }

  /// Releases all resources associated with the Llama instance.
  ///
  /// Frees the memory allocated for the model, context, and batch,
  /// and calls the backend cleanup routines.
  /// This method should be called when the Llama instance is no longer needed.
  dispose() {
    lib.llama_batch_free(batch);

    if (context.address != 0) {
      lib.llama_free(context);
    }

    if (model.address != 0) {
      lib.llama_free_model(model);
    }

    lib.llama_backend_free();
  }

  /// Sets the initial prompt for the model to begin generating text.
  ///
  /// Tokenizes the input prompt and prepares the model for text generation.
  /// An exception is thrown if the required KV cache size exceeds the context's limit.
  /// The function also initializes the batch for token processing.
  setPrompt(String prompt) {
    // context = lib.llama_new_context_with_model(model, contextParams.get());
    tokensList = tokenize(prompt, true);

    if (length != -1) {
      int nCtx = lib.llama_n_ctx(context);
      int nKvReq = tokensList.length + (length - tokensList.length);

      if (nKvReq > nCtx) {
        throw Exception("error: the required KV cache size is not big enough");
      }
    }

    batch.n_tokens = 0;

    for (var i = 0; i < tokensList.length; i++) {
      batchAdd(batch, tokensList[i], i, [0], false);
    }

    batch.logits[batch.n_tokens - 1] = 1;

    if (lib.llama_decode(context, batch) != 0) {
      throw Exception("llama_decode() failed");
    }

    cursor = batch.n_tokens;
  }

  /// Generates and returns the next token in the sequence based on the current context.
  ///
  /// This function handles the selection and decoding of the next token.
  /// Returns a tuple with the generated text and a boolean indicating if the end-of-sequence token is reached.
  /// An exception is thrown if llama_decode fails during processing.
  (String, bool) getNext() {
    Pointer<Int32> newTokenId = calloc.allocate<Int32>(sizeOf<Int32>());
    final int nVocab = lib.llama_n_vocab(model);
    final logits = lib.llama_get_logits_ith(context, batch.n_tokens - 1);

    final Pointer<llama_token_data> candidates = calloc<llama_token_data>(nVocab);
    for (int tokenId = 0; tokenId < nVocab; tokenId++) {
      candidates[tokenId].id = tokenId;
      candidates[tokenId].logit = logits[tokenId];
      candidates[tokenId].p = 0.0;
    }

    final Pointer<llama_token_data_array> candidatesP = calloc<llama_token_data_array>();
    candidatesP.ref.data = candidates;
    candidatesP.ref.size = nVocab;
    candidatesP.ref.sorted = false;

    SamplingContext sampling = SamplingContext(this);
    sampling.params = samplingParams;

    newTokenId.value = candidatesP.ref.data.elementAt(0).ref.id;
    newTokenId.value = sampling.sample(newTokenId, null);
    sampling.accept(newTokenId.value);

    // newTokenId.value = lib.llama_sample_token_greedy(context, candidatesP);
    // lastTokens.add(newTokenId);

    // calloc.free(nativeLastTokens);
    calloc.free(candidates);
    calloc.free(candidatesP);

    sampling.dispose();

    final newTokenStr = tokenToPiece(newTokenId.value);

    batch.n_tokens = 0;
    batchAdd(batch, newTokenId.value, cursor, [0], true);

    decode++;
    cursor++;

    if (lib.llama_decode(context, batch) != 0) {
      throw Exception("failed to evaluate llama!");
    }

    int token = newTokenId.value;
    calloc.free(newTokenId);
    return (newTokenStr, token == lib.llama_token_eos(model));
  }

  /// Asynchronously generates text based on a given prompt.
  ///
  /// This is a generator function that continuously yields generated text.
  /// It continues generating text until an end-of-sequence condition is met.
  /// This is ideal for use cases that require streaming output.
  Stream<String> prompt(String prompt) async* {
    setPrompt(prompt);
    while (true) {
      var (text, done) = getNext();
      yield text;
      if (done) {
        break;
      }
    }
  }

  /// Resets the state of the Llama instance.
  ///
  /// Clears the current token list and temporary data, and flushes the KV cache.
  /// This method should be used to reset the state before starting a new text generation session.
  void clear() {
    tokensList.clear();
    lastTokens.clear();
    lib.llama_kv_cache_clear(context);
    batch.n_tokens = 0;
    cursor = 0;
    decode = 0;
  }

  // Utility methods

  /// Adds a token to the batch for processing.
  ///
  /// Appends a token with its associated position and sequence IDs to the batch.
  /// The 'logits' flag indicates whether logits should be calculated for this token.
  /// This is a utility method used internally during token processing.
  void batchAdd(
      llama_batch batch, int id, int pos, List<int> seqIds, bool logits) {
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seqIds.length;
    for (int i = 0; i < seqIds.length; i++) {
      batch.seq_id[batch.n_tokens][i] = seqIds[i];
    }
    batch.logits[batch.n_tokens] = logits ? 1 : 0;
    batch.n_tokens += 1;
  }

  /// Converts a text string to a list of token IDs.
  ///
  /// This function tokenizes the given string into a sequence of integers representing tokens.
  /// An optional flag 'addBos' indicates whether to prepend a beginning-of-sentence token.
  /// The function handles memory allocation and conversion between Dart strings and native character arrays.
  List<int> tokenize(String text, bool addBos) {
    Pointer<Char> cchar = text.toNativeUtf8().cast<Char>();

    int nUtf8CodeUnits = utf8.encode(text).length;
    int nTokens = nUtf8CodeUnits + (addBos ? 1 : 0) + 1;

    Pointer<llama_token> tokens =
        malloc.allocate<llama_token>(nTokens * sizeOf<llama_token>());

    try {
      int tokenCount = lib.llama_tokenize(
          model, cchar, nUtf8CodeUnits, tokens, nTokens, addBos, false);

      List<int> tokensList = [];
      for (int i = 0; i < tokenCount; i++) {
        tokensList.add(tokens.elementAt(i).value);
      }

      return tokensList;
    } finally {
      malloc.free(tokens);
      malloc.free(cchar);
    }
  }

  /// Converts a token ID to its corresponding string representation.
  ///
  /// This utility function takes a token ID and returns the associated text piece.
  /// It handles the conversion and memory management involved in this process.
  /// This is typically used in decoding the output of the model.
  String tokenToPiece(int token) {
    Pointer<Uint8> result = malloc.allocate<Uint8>(8);
    try {
      int nTokens =
          lib.llama_token_to_piece(model, token, result.cast<Char>(), 8);
      if (nTokens < 0 || nTokens > 8) {
        return '';
      }
      return String.fromCharCodes(result.asTypedList(nTokens));
    } finally {
      malloc.free(result);
    }
  }
}
