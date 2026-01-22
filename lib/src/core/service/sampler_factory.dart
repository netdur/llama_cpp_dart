import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../llama_cpp.dart';
import '../sampler_params.dart';

class SamplerFactory {
  static Pointer<llama_sampler> build({
    required llama_cpp lib,
    required Pointer<llama_vocab> vocab,
    required Pointer<llama_model> model,
    required SamplerParams params,
  }) {
    final sparams = lib.llama_sampler_chain_default_params();
    sparams.no_perf = false;
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

    if (params.dryMultiplier > 0.0) {
      try {
        final breakers = params.dryBreakers;
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
          smpl,
          lib.llama_sampler_init_dry(
            vocab,
            nCtxTrain,
            params.dryMultiplier,
            params.dryBase,
            params.dryAllowedLen,
            params.dryPenaltyLastN,
            breakersPtr,
            breakerCount,
          ),
        );

        for (var ptr in allocatedStrings) {
          malloc.free(ptr);
        }
        malloc.free(breakersPtr);
      } catch (_) {}
    }

    if (params.mirostat == 2) {
      lib.llama_sampler_chain_add(
          smpl,
          lib.llama_sampler_init_mirostat_v2(
              params.seed, params.mirostatTau, params.mirostatEta));
    } else if (params.mirostat == 1) {
      lib.llama_sampler_chain_add(
          smpl,
          lib.llama_sampler_init_mirostat(
              lib.llama_n_vocab(vocab),
              params.seed,
              params.mirostatTau,
              params.mirostatEta,
              params.mirostatM));
    } else {
      lib.llama_sampler_chain_add(
          smpl, lib.llama_sampler_init_top_k(params.topK));

      lib.llama_sampler_chain_add(
          smpl, lib.llama_sampler_init_top_p(params.topP, 1));

      lib.llama_sampler_chain_add(
          smpl, lib.llama_sampler_init_min_p(params.minP, 1));

      lib.llama_sampler_chain_add(
          smpl, lib.llama_sampler_init_typical(params.typical, 1));

      if (params.dynatempRange > 0.0) {
        try {
          lib.llama_sampler_chain_add(
              smpl,
              lib.llama_sampler_init_temp_ext(
                  params.temp, params.dynatempRange, params.dynatempExponent));
        } catch (_) {
          lib.llama_sampler_chain_add(
              smpl, lib.llama_sampler_init_temp(params.temp));
        }
      } else {
        lib.llama_sampler_chain_add(
            smpl, lib.llama_sampler_init_temp(params.temp));
      }

      if (params.xtcProbability > 0.0) {
        try {
          lib.llama_sampler_chain_add(
              smpl,
              lib.llama_sampler_init_xtc(params.xtcProbability,
                  params.xtcThreshold, 1, params.seed));
        } catch (_) {}
      }
    }

    lib.llama_sampler_chain_add(
        smpl, lib.llama_sampler_init_dist(params.seed));

    return smpl;
  }
}
