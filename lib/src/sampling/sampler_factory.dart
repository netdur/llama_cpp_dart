import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import '../model/model.dart';
import 'sampler.dart';
import 'sampler_params.dart';

/// Builds a [Sampler] from declarative [SamplerParams].
///
/// Default chain order, mirroring llama.cpp's recommended layout:
///
///   logit_bias → penalties → dry → top-n-sigma → top-k → typical-p →
///   top-p → min-p → xtc → (dynamic-)temperature → grammar → infill →
///   mirostat | adaptive-p | dist
///
/// Stages whose config is disabled are skipped. `greedy` short-circuits to
/// argmax sampling and ignores everything else.
///
/// Pass [model] when the chain may use grammar / DRY / infill / logit-bias /
/// mirostat-v1 — those samplers need the vocab or `n_ctx_train`. A
/// [StateError] is thrown if a stage is requested without the model.
final class SamplerFactory {
  SamplerFactory._();

  static Sampler build(SamplerParams params, {LlamaModel? model}) {
    final b = LlamaLibrary.bindings;
    final cp = b.llama_sampler_chain_default_params();
    final chain = b.llama_sampler_chain_init(cp);

    if (params.greedy) {
      b.llama_sampler_chain_add(chain, b.llama_sampler_init_greedy());
      return Sampler(chain, params);
    }

    Pointer<llama_vocab> requireVocab(String stage) {
      if (model == null) {
        throw StateError(
          'SamplerFactory.build: $stage requires a model (vocab access). '
          'Pass `model:` when building the sampler.',
        );
      }
      return model.vocab.pointer;
    }

    if (params.logitBias.isNotEmpty) {
      final nVocab = model == null
          ? throw StateError(
              'SamplerFactory.build: logit_bias requires a model (n_vocab).')
          : b.llama_vocab_n_tokens(model.vocab.pointer);
      final arr = calloc<llama_logit_bias>(params.logitBias.length);
      for (var i = 0; i < params.logitBias.length; i++) {
        arr[i].token = params.logitBias[i].token;
        arr[i].bias = params.logitBias[i].bias;
      }
      // llama copies the array; safe to free immediately.
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_logit_bias(nVocab, params.logitBias.length, arr),
      );
      calloc.free(arr);
    }

    final hasPenalty = params.repeatPenalty != 1.0 ||
        params.frequencyPenalty != 0.0 ||
        params.presencePenalty != 0.0;
    if (hasPenalty) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_penalties(
          params.penaltyLastN,
          params.repeatPenalty,
          params.frequencyPenalty,
          params.presencePenalty,
        ),
      );
    }

    if (params.dry.enabled) {
      final vocab = requireVocab('DRY sampler');
      final m = model!;
      final breakers = params.dry.seqBreakers;
      final breakerPtrs =
          calloc<Pointer<Char>>(breakers.isEmpty ? 1 : breakers.length);
      final allocated = <Pointer<Utf8>>[];
      for (var i = 0; i < breakers.length; i++) {
        final s = breakers[i].toNativeUtf8(allocator: calloc);
        allocated.add(s);
        breakerPtrs[i] = s.cast();
      }
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_dry(
          vocab,
          m.trainCtx,
          params.dry.multiplier,
          params.dry.base,
          params.dry.allowedLength,
          params.dry.penaltyLastN,
          breakerPtrs,
          breakers.length,
        ),
      );
      // llama_sampler_init_dry copies the strings internally.
      for (final p in allocated) {
        calloc.free(p);
      }
      calloc.free(breakerPtrs);
    }

    if (params.topNSigma > 0) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_top_n_sigma(params.topNSigma),
      );
    }

    if (params.topK > 0) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_top_k(params.topK),
      );
    }

    if (params.typicalP < 1.0) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_typical(params.typicalP, params.minKeep),
      );
    }

    if (params.topP < 1.0) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_top_p(params.topP, params.minKeep),
      );
    }

    if (params.minP > 0.0) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_min_p(params.minP, params.minKeep),
      );
    }

    if (params.xtc.enabled) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_xtc(
          params.xtc.probability,
          params.xtc.threshold,
          params.xtc.minKeep,
          params.xtc.seed,
        ),
      );
    }

    if (params.mirostat.enabled) {
      // Mirostat is terminal — no temp / dist after it.
      _addMirostat(chain, params, model);
      if (params.grammar.enabled) {
        _addGrammar(chain, params.grammar, requireVocab('grammar'));
      }
      if (params.infill) {
        b.llama_sampler_chain_add(
          chain,
          b.llama_sampler_init_infill(requireVocab('infill')),
        );
      }
      return Sampler(chain, params);
    }

    if (params.temperature <= 0.0) {
      // Temp <= 0 — pick argmax after the filters.
      if (params.grammar.enabled) {
        _addGrammar(chain, params.grammar, requireVocab('grammar'));
      }
      if (params.infill) {
        b.llama_sampler_chain_add(
          chain,
          b.llama_sampler_init_infill(requireVocab('infill')),
        );
      }
      b.llama_sampler_chain_add(chain, b.llama_sampler_init_greedy());
      return Sampler(chain, params);
    }

    if (params.dynamicTemp.enabled) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_temp_ext(
          params.temperature,
          params.dynamicTemp.range,
          params.dynamicTemp.exponent,
        ),
      );
    } else {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_temp(params.temperature),
      );
    }

    if (params.grammar.enabled) {
      _addGrammar(chain, params.grammar, requireVocab('grammar'));
    }

    if (params.infill) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_infill(requireVocab('infill')),
      );
    }

    if (params.adaptiveP.enabled) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_adaptive_p(
          params.adaptiveP.target,
          params.adaptiveP.decay,
          params.adaptiveP.seed,
        ),
      );
    } else {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_dist(params.seed),
      );
    }

    return Sampler(chain, params);
  }

  static void _addMirostat(
    Pointer<llama_sampler> chain,
    SamplerParams params,
    LlamaModel? model,
  ) {
    final b = LlamaLibrary.bindings;
    switch (params.mirostat.version) {
      case MirostatVersion.v1:
        if (model == null) {
          throw StateError(
            'SamplerFactory.build: Mirostat v1 requires a model (n_vocab).',
          );
        }
        final nVocab = b.llama_vocab_n_tokens(model.vocab.pointer);
        b.llama_sampler_chain_add(
          chain,
          b.llama_sampler_init_mirostat(
            nVocab,
            params.seed,
            params.mirostat.tau,
            params.mirostat.eta,
            params.mirostat.m,
          ),
        );
        break;
      case MirostatVersion.v2:
        b.llama_sampler_chain_add(
          chain,
          b.llama_sampler_init_mirostat_v2(
            params.seed,
            params.mirostat.tau,
            params.mirostat.eta,
          ),
        );
        break;
      case MirostatVersion.off:
        return;
    }
  }

  static void _addGrammar(
    Pointer<llama_sampler> chain,
    GrammarConfig cfg,
    Pointer<llama_vocab> vocab,
  ) {
    final b = LlamaLibrary.bindings;
    final grammarPtr = cfg.grammar!.toNativeUtf8(allocator: calloc);
    final rootPtr = cfg.root.toNativeUtf8(allocator: calloc);
    try {
      if (!cfg.lazy) {
        b.llama_sampler_chain_add(
          chain,
          b.llama_sampler_init_grammar(
            vocab,
            grammarPtr.cast(),
            rootPtr.cast(),
          ),
        );
        return;
      }

      final patterns = cfg.triggerPatterns;
      final patternPtrs =
          calloc<Pointer<Char>>(patterns.isEmpty ? 1 : patterns.length);
      final allocated = <Pointer<Utf8>>[];
      for (var i = 0; i < patterns.length; i++) {
        final s = patterns[i].toNativeUtf8(allocator: calloc);
        allocated.add(s);
        patternPtrs[i] = s.cast();
      }

      final tokens = cfg.triggerTokens;
      final tokenPtr = calloc<llama_token>(tokens.isEmpty ? 1 : tokens.length);
      for (var i = 0; i < tokens.length; i++) {
        tokenPtr[i] = tokens[i];
      }

      try {
        b.llama_sampler_chain_add(
          chain,
          b.llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammarPtr.cast(),
            rootPtr.cast(),
            patternPtrs,
            patterns.length,
            tokenPtr,
            tokens.length,
          ),
        );
      } finally {
        for (final p in allocated) {
          calloc.free(p);
        }
        calloc.free(patternPtrs);
        calloc.free(tokenPtr);
      }
    } finally {
      calloc.free(grammarPtr);
      calloc.free(rootPtr);
    }
  }
}
