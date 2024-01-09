import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'llama_cpp.dart';

typedef LlamaLogCallback = Void Function(
    Int32 level, Pointer<Char> text, Pointer<Void> userData);

void llamaLogCallbackLogTee(
    int level, Pointer<Char> text, Pointer<Void> userData) {}

class ModelManager {
  late Pointer<llama_model> model;
  late Pointer<llama_context> context;
  late llama_batch batch;
  late llama_cpp lib;

  bool disableLog = false;

  int nLen = (512 / 4).round();
  int nCur = 0;
  int threads = 10;
  int nDecode = 0;

  int seed = 1234;
  int contextSize = 2048;

  void loadModel(String path) {
    lib = llama_cpp(DynamicLibrary.process());

    if (disableLog) {
      final Pointer<NativeFunction<LlamaLogCallback>> callbackPointer =
          Pointer.fromFunction<LlamaLogCallback>(llamaLogCallbackLogTee);
      lib.llama_log_set(callbackPointer, nullptr);
    }

    batch = lib.llama_batch_init(512, 0, 1);

    lib.llama_backend_init(false);
    llama_model_params modelParams = lib.llama_model_default_params();

    final pathUtf8 = path.toNativeUtf8();
    model = lib.llama_load_model_from_file(pathUtf8.cast<Char>(), modelParams);
    malloc.free(pathUtf8);

    if (model.address == 0) {
      throw Exception("Model loading failed");
    }
  }

  void unloadModel() {
    lib.llama_batch_free(batch);
    lib.llama_free_model(model);
    lib.llama_backend_free();
  }

  void setup(String prompt) {
    List<int> tokensList = tokenize(prompt, true);

    final nCtx = lib.llama_n_ctx(context);
    final nKvReq = tokensList.length + (nLen - tokensList.length);

    if (nKvReq > nCtx) {
      throw Exception('The required KV cache size is not big enough');
    }

    batch.n_tokens = 0;

    for (var i = 0; i < tokensList.length; i++) {
      batchAdd(batch, tokensList[i], i, [0], false);
    }
    batch.logits[batch.n_tokens - 1] = 1;

    if (lib.llama_decode(context, batch) != 0) {
      throw Exception('Error: decode failed');
    }

    nCur = batch.n_tokens;
  }

  String getNext() {
    int newTokenId = 0;
    final nVocab = lib.llama_n_vocab(model);
    final logits = lib.llama_get_logits_ith(context, batch.n_tokens - 1);

    final Pointer<llama_token_data> candidates =
        malloc.allocate<llama_token_data>(sizeOf<llama_token_data>() * nVocab);
    for (int tokenId = 0; tokenId < nVocab; tokenId++) {
      candidates.elementAt(tokenId).ref
        ..id = tokenId
        ..logit = logits.elementAt(tokenId).value
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

    if (newTokenId == lib.llama_token_eos(model) || nCur == nLen) {
      final newTokenStr = tokenToPiece(newTokenId);
      return newTokenStr;
    }

    final newTokenStr = tokenToPiece(newTokenId);

    batch.n_tokens = 0;
    batchAdd(batch, newTokenId, nCur, [0], true);

    nDecode++;
    nCur++;

    if (lib.llama_decode(context, batch) != 0) {
      throw Exception("failed to evaluate llama!");
    }

    return newTokenStr;
  }

  void createContext() {
    llama_context_params ctxParams = lib.llama_context_default_params();
    ctxParams.seed = seed;
    ctxParams.n_ctx = contextSize;
    ctxParams.n_threads = threads;
    ctxParams.n_threads_batch = threads;

    context = lib.llama_new_context_with_model(model, ctxParams);
    if (context.address == 0) {
      throw Exception("Context loading failed");
    }
  }

  void clearContext() {
    lib.llama_kv_cache_clear(context);
    lib.llama_free(context);
  }

  String modelInfo() {
    final Pointer<Char> result = malloc.allocate<Char>(256);
    try {
      int nChars = lib.llama_model_desc(model, result, 256);
      String dartString = result.cast<Utf8>().toDartString(length: nChars);
      return dartString;
    } finally {
      malloc.free(result);
    }
  }

  // utils
  List<int> tokenize(String text, bool addBos) {
    final Pointer<Char> utf8Text = text.toNativeUtf8().cast<Char>();
    final nUtf8CodeUnits = utf8.encode(text).length;
    final nTokens = nUtf8CodeUnits + (addBos ? 1 : 0) + 1;

    final Pointer<Int32> tokens =
        malloc.allocate<Int32>(nTokens * sizeOf<Int32>());

    try {
      final tokenCount = lib.llama_tokenize(
          model, utf8Text, nUtf8CodeUnits, tokens, nTokens, addBos, false);

      final Pointer<Int32> dartTokens =
          malloc.allocate<Int32>(tokenCount * sizeOf<Int32>());
      for (int i = 0; i < tokenCount; i++) {
        dartTokens.elementAt(i).value = tokens.elementAt(i).value;
      }

      var dartTokenList =
          List.generate(tokenCount, (i) => dartTokens.elementAt(i).value);
      return dartTokenList;
    } finally {
      malloc.free(tokens);
      malloc.free(utf8Text);
    }
  }

  void batchAdd(
      llama_batch batch, int id, int pos, List<int> seqIds, bool logits) {
    final currentIndex = batch.n_tokens;

    // Accessing and setting values in the struct array fields
    batch.token[currentIndex] = id;
    batch.pos[currentIndex] = pos;
    batch.n_seq_id[currentIndex] = seqIds.length;

    // Handling the array of seq_ids
    for (int i = 0; i < seqIds.length; i++) {
      batch.seq_id[currentIndex][i] = seqIds[i];
    }

    batch.logits[currentIndex] = logits ? 1 : 0;

    batch.n_tokens++;
  }

  String tokenToPiece(int token) {
    final Pointer<Uint8> result = malloc.allocate<Uint8>(8);
    try {
      int nTokens =
          lib.llama_token_to_piece(model, token, result.cast<Char>(), 8);

      if (nTokens < 0) {
        final Pointer<Uint8> newResult = malloc.allocate<Uint8>(-nTokens);
        try {
          int nNewTokens = lib.llama_token_to_piece(
              model, token, newResult.cast<Char>(), -nTokens);
          return String.fromCharCodes(newResult.asTypedList(nNewTokens));
        } finally {
          malloc.free(newResult);
        }
      } else {
        return String.fromCharCodes(result.asTypedList(nTokens));
      }
    } finally {
      malloc.free(result);
    }
  }
}
