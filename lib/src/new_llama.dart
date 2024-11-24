import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'sampler_params.dart';
import 'model_params.dart';

import 'llama_cpp.dart';
import 'context_params.dart';

class NewLlama {
  static llama_cpp? _lib;
  late Pointer<llama_model> model;
  late Pointer<llama_context> context;
  late llama_batch batch;

  Pointer<llama_sampler> _smpl = nullptr;
  Pointer<llama_token> _tokens = nullptr;
  Pointer<llama_token> _tokenPtr = nullptr;
  int _n_prompt = 0;
  int _n_predict = 32;
  int _nPos = 0;

  static String? libraryPath = Platform.isAndroid ? "libllama.so" : null;

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

  NewLlama(String modelPath) {
    lib.llama_backend_init();

    ModelParams modelParamsDart = ModelParams();
    var modelParams = modelParamsDart.get();

    final modelPathPtr = modelPath.toNativeUtf8().cast<Char>();
    model = lib.llama_load_model_from_file(modelPathPtr, modelParams);
    malloc.free(modelPathPtr);

    if (model.address == 0) {
      throw Exception("Could not load model at $modelPath");
    }

    ContextParams contextParamsDart = ContextParams();
    contextParamsDart.nCtx = 512;
    contextParamsDart.nBatch = 512;
    contextParamsDart.noPerfTimings = false;
    var contextParams = contextParamsDart.get();

    context = lib.llama_new_context_with_model(model, contextParams);
    if (context.address == 0) {
      throw Exception("Could not load context!");
    }

    SamplerParams samplerParams = SamplerParams();

    // Initialize sampler chain
    llama_sampler_chain_params sparams =
        lib.llama_sampler_chain_default_params();
    sparams.no_perf = false;
    _smpl = lib.llama_sampler_chain_init(sparams);

    // Add samplers based on params
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
            lib.llama_n_vocab(model),
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
    lib.llama_sampler_chain_add(_smpl,
        lib.llama_sampler_init_grammar(model, grammarStrPtr, grammarRootPtr));
    calloc.free(grammarStrPtr);
    calloc.free(grammarRootPtr);

    lib.llama_sampler_chain_add(
        _smpl,
        lib.llama_sampler_init_penalties(
            lib.llama_n_vocab(model),
            lib.llama_token_eos(model),
            lib.llama_token_nl(model),
            samplerParams.penaltyLastTokens,
            samplerParams.penaltyRepeat,
            samplerParams.penaltyFreq,
            samplerParams.penaltyPresent,
            samplerParams.penaltyNewline,
            samplerParams.ignoreEOS));

    // Add DRY sampler
    final seqBreakers = samplerParams.dryBreakers;
    final numBreakers = seqBreakers.length;
    final seqBreakersPointer = calloc<Pointer<Char>>(numBreakers);

    try {
      for (var i = 0; i < numBreakers; i++) {
        seqBreakersPointer[i] = seqBreakers[i].toNativeUtf8().cast<Char>();
      }

      lib.llama_sampler_chain_add(
          _smpl,
          lib.llama_sampler_init_dry(
              model,
              samplerParams.dryPenalty,
              samplerParams.dryMultiplier,
              samplerParams.dryAllowedLen,
              samplerParams.dryLookback,
              seqBreakersPointer,
              numBreakers));
    } finally {
      // Clean up DRY sampler allocations
      for (var i = 0; i < numBreakers; i++) {
        calloc.free(seqBreakersPointer[i]);
      }
      calloc.free(seqBreakersPointer);
    }

    lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_infill(model));

    /*
    // samplers
    llama_sampler_chain_params sparams =
        lib.llama_sampler_chain_default_params();
    sparams.no_perf = false;
    _smpl = lib.llama_sampler_chain_init(sparams);

    //
    lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_greedy());
    lib.llama_sampler_chain_add(
        _smpl, lib.llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_softmax());

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_top_k(40));

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_top_p(0.95, 1));

    /// @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
    lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_min_p(0.05, 1));

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_typical(1.00, 1));

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_temp(0.80));

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    // lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_temp_ext(0.80));

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    lib.llama_sampler_chain_add(
        _smpl, lib.llama_sampler_init_xtc(1.0, 0.1, 1, 1));

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    lib.llama_sampler_chain_add(
        _smpl,
        lib.llama_sampler_init_mirostat(
            lib.llama_n_vocab(model), LLAMA_DEFAULT_SEED, 5.00, 0.10, 100));

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    lib.llama_sampler_chain_add(_smpl,
        lib.llama_sampler_init_mirostat_v2(LLAMA_DEFAULT_SEED, 5.00, 0.10));

    final grammarStrPtr = "".toNativeUtf8().cast<Char>();
    final grammarRootPtr = "".toNativeUtf8().cast<Char>();
    lib.llama_sampler_chain_add(_smpl,
        lib.llama_sampler_init_grammar(model, grammarStrPtr, grammarRootPtr));

    lib.llama_sampler_chain_add(
        _smpl,
        lib.llama_sampler_init_penalties(
            lib.llama_n_vocab(model), // llama_n_vocab()
            lib.llama_token_eos(model), // llama_token_eos()
            lib.llama_token_nl(model), // llama_token_nl()
            64, // last n tokens to penalize (0 = disable penalty, -1 = context size)
            1.00, // 1.0 = disabled
            0.00, // 0.0 = disabled
            0.00, // 0.0 = disabled
            false, // consider newlines as a repeatable token
            false)); // ignore the end-of-sequence token

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    final seqBreakers = ["\n", ":", "\"", "*"];
    final numBreakers = seqBreakers.length;
    final seqBreakersPointer = calloc<Pointer<Char>>(numBreakers);
    for (var i = 0; i < numBreakers; i++) {
      seqBreakersPointer[i] = seqBreakers[i].toNativeUtf8().cast<Char>();
    }

    lib.llama_sampler_chain_add(
        _smpl,
        lib.llama_sampler_init_dry(
            model,
            0.0, // DRY repetition penalty for tokens extending repetition:
            1.75, // multiplier * base ^ (length of sequence before token - allowed length)
            2, // tokens extending repetitions beyond this receive penalty
            -1, // how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
            seqBreakersPointer, // default sequence breakers for DRY
            numBreakers));

    // @TODO
    // lib.llama_sampler_init_logit_bias(
    //    lib.llama_n_vocab(model), n_logit_bias, logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    lib.llama_sampler_chain_add(_smpl, lib.llama_sampler_init_infill(model));
    */

    _tokenPtr = malloc<llama_token>();
  }

  void setPrompt(String prompt) {
    // Free previous tokens if they exist
    if (_tokens.address != 0) {
      malloc.free(_tokens);
    }

    final promptPtr = prompt.toNativeUtf8().cast<Char>();
    _n_prompt = -lib.llama_tokenize(
        model, promptPtr, prompt.length, nullptr, 0, true, true);

    _tokens = malloc<llama_token>(_n_prompt);
    if (lib.llama_tokenize(
            model, promptPtr, prompt.length, _tokens, _n_prompt, true, true) <
        0) {
      malloc.free(promptPtr);
      throw Exception("Failed to tokenize prompt");
    }
    malloc.free(promptPtr);

    // Print prompt tokens
    for (int i = 0; i < _n_prompt; i++) {
      final buf = malloc<Char>(128);
      int n = lib.llama_token_to_piece(model, _tokens[i], buf, 128, 0, true);
      if (n < 0) {
        malloc.free(buf);
        throw Exception("Failed to convert token to piece");
      }
      String piece = String.fromCharCodes(buf.cast<Uint8>().asTypedList(n));
      stdout.write(piece);
      malloc.free(buf);
    }

    batch = lib.llama_batch_get_one(_tokens, _n_prompt);
    _nPos = 0;
  }

  (String, bool) getNext() {
    if (_nPos + batch.n_tokens >= _n_prompt + _n_predict) {
      return ("", true);
    }

    if (lib.llama_decode(context, batch) != 0) {
      throw Exception("Failed to eval");
    }

    _nPos += batch.n_tokens;

    // Sample next token
    int newTokenId = lib.llama_sampler_sample(_smpl, context, -1);

    if (lib.llama_token_is_eog(model, newTokenId)) {
      return ("", true);
    }

    final buf = malloc<Char>(128);
    int n = lib.llama_token_to_piece(model, newTokenId, buf, 128, 0, true);
    if (n < 0) {
      malloc.free(buf);
      throw Exception("Failed to convert token to piece");
    }

    String piece = String.fromCharCodes(buf.cast<Uint8>().asTypedList(n));
    malloc.free(buf);

    // Prepare next batch
    _tokenPtr.value = newTokenId;
    batch = lib.llama_batch_get_one(_tokenPtr, 1);

    bool isEos = newTokenId == lib.llama_token_eos(model);
    return (piece, isEos);
  }

  void dispose() {
    if (_tokens.address != 0) malloc.free(_tokens);
    if (_tokenPtr.address != 0) malloc.free(_tokenPtr);
    lib.llama_sampler_free(_smpl);
    if (context.address != 0) lib.llama_free(context);
    if (model.address != 0) lib.llama_free_model(model);
    lib.llama_backend_free();
  }
}
