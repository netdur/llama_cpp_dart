import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../llama_cpp.dart';
import '../sampler_params.dart';

class SamplerFactory {
  static Pointer<llama_sampler> build({
    required llama_cpp lib,
    required Pointer<llama_vocab> vocab,
    required SamplerParams params,
  }) {
    final sparams = lib.llama_sampler_chain_default_params();
    final smpl = lib.llama_sampler_chain_init(sparams);

    if (params.greedy) {
      lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_greedy());
      return smpl;
    }

    final grammarStrPtr = params.grammarStr.toNativeUtf8().cast<Char>();
    final grammarRootPtr = params.grammarRoot.toNativeUtf8().cast<Char>();

    if (params.grammarStr.isNotEmpty) {
      final grammar =
          lib.llama_sampler_init_grammar(vocab, grammarStrPtr, grammarRootPtr);
      if (grammar != nullptr) lib.llama_sampler_chain_add(smpl, grammar);
    }

    malloc.free(grammarStrPtr);
    malloc.free(grammarRootPtr);

    lib.llama_sampler_chain_add(
      smpl,
      lib.llama_sampler_init_penalties(
        params.penaltyLastTokens,
        params.penaltyRepeat,
        params.penaltyFreq,
        params.penaltyPresent,
      ),
    );

    lib.llama_sampler_chain_add(
        smpl, lib.llama_sampler_init_top_k(params.topK));
    lib.llama_sampler_chain_add(
        smpl, lib.llama_sampler_init_top_p(params.topP, 1));
    lib.llama_sampler_chain_add(
        smpl, lib.llama_sampler_init_min_p(params.minP, 1));
    lib.llama_sampler_chain_add(
        smpl, lib.llama_sampler_init_temp(params.temp));
    lib.llama_sampler_chain_add(
        smpl, lib.llama_sampler_init_dist(params.seed));

    return smpl;
  }
}
