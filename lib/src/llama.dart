import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
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
  List<Char> temporaryInvalidCChars = [];

  /// Length of the ouput. Default is -1.
  int length = -1;

  /// Cursor position in the token list. Default is 0.
  int cursor = 0;

  /// Counter for decoding operations. Default is 0.
  int decode = 0;

  /// Getter for the Llama library.
  ///
  /// Loads the library based on the current platform.
  static llama_cpp get lib {
    if (_lib == null) {
      if (Platform.isAndroid) {
        _lib = llama_cpp(DynamicLibrary.open("libllama.so"));
      } else {
        _lib = llama_cpp(DynamicLibrary.process());
      }
    }
    return _lib!;
  }

  /// Constructor for Llama.
  ///
  /// Loads the model and context based on provided model and context parameters.
  Llama(String modelPath, [ModelParams? mparams, ContextParams? cparams]) {
    mparams ??= ModelParams();
    cparams ??= ContextParams();

    lib.llama_backend_init(false);
    llama_model_params modelParams = mparams.get();

    Pointer<Char> char = modelPath.toNativeUtf8().cast<Char>();
    model = lib.llama_load_model_from_file(char, modelParams);
    malloc.free(char);

    if (model.address == 0) {
      throw Exception("Could not load model at $modelPath");
    }

    llama_context_params contextParams = cparams.get();
    context = lib.llama_new_context_with_model(model, contextParams);
    if (context.address == 0) {
      throw Exception("Could not load context!");
    }

    batch = lib.llama_batch_init(cparams.batch, 0, 1);
  }

  /// Releases all allocated resources.
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

  /// Returns information about the model.
  String modelInfo() {
    final Pointer<Char> result = malloc.allocate<Char>(256);
    try {
      int length = lib.llama_model_desc(model, result, 256);
      return result.cast<Utf8>().toDartString(length: length);
    } finally {
      malloc.free(result);
    }
  }

  /// Sets the prompt for the model.
  ///
  /// Tokenizes the input prompt and prepares it for processing.
  setPrompt(String prompt) {
    tokensList = tokenize(prompt, true);
    temporaryInvalidCChars = [];

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

  /// Generates and returns the next token in the sequence.
  (String, bool) getNext() {
    int newTokenId = 0;
    final nVocab = lib.llama_n_vocab(model);
    final logits = lib.llama_get_logits_ith(context, batch.n_tokens - 1);

    final Pointer<llama_token_data> candidates =
        malloc.allocate<llama_token_data>(sizeOf<llama_token_data>() * nVocab);
    for (int i = 0; i < nVocab; i++) {
      candidates.elementAt(i).ref
        ..id = i
        ..logit = logits.elementAt(i).value
        ..p = 0.0;
    }

    final Pointer<llama_token_data_array> candidatesP =
        malloc<llama_token_data_array>();
    candidatesP.ref
      ..data = candidates
      ..size = nVocab
      ..sorted = true;

    newTokenId = lib.llama_sample_token_greedy(context, candidatesP);

    malloc.free(candidates);
    malloc.free(candidatesP);

    if (newTokenId == lib.llama_token_eos(model)) {
      final newTokenStr = tokenToPiece(newTokenId);
      return (newTokenStr, newTokenId == lib.llama_token_eos(model));
    }

    final newTokenStr = tokenToPiece(newTokenId);

    batch.n_tokens = 0;
    batchAdd(batch, newTokenId, cursor, [0], true);

    decode++;
    cursor++;

    if (lib.llama_decode(context, batch) != 0) {
      throw Exception("failed to evaluate llama!");
    }

    return (newTokenStr, newTokenId == lib.llama_token_eos(model));
  }

  /// Stream of generated text based on the given prompt.
  ///
  /// Continuously yields generated text until the end of the sequence.
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

  /// Clears the current token list and temporary data.
  void clear() {
    tokensList.clear();
    temporaryInvalidCChars.clear();
    lib.llama_kv_cache_clear(context);
  }

  // Utility methods

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
