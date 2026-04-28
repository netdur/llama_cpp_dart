import '../ffi/library_loader.dart';
import 'sampler.dart';
import 'sampler_params.dart';

/// Builds a [Sampler] from declarative [SamplerParams].
///
/// The chain order follows llama.cpp's recommended layout:
///   penalties → top-k → typical-p → top-p → min-p → temperature → dist
///
/// Disabled stages are skipped. `greedy` short-circuits to argmax sampling.
final class SamplerFactory {
  SamplerFactory._();

  static Sampler build(SamplerParams params) {
    final b = LlamaLibrary.bindings;
    final cp = b.llama_sampler_chain_default_params();
    final chain = b.llama_sampler_chain_init(cp);

    if (params.greedy) {
      b.llama_sampler_chain_add(chain, b.llama_sampler_init_greedy());
      return Sampler(chain, params);
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

    if (params.topK > 0) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_top_k(params.topK),
      );
    }

    if (params.typicalP < 1.0) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_typical(params.typicalP, 1),
      );
    }

    if (params.topP < 1.0) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_top_p(params.topP, 1),
      );
    }

    if (params.minP > 0.0) {
      b.llama_sampler_chain_add(
        chain,
        b.llama_sampler_init_min_p(params.minP, 1),
      );
    }

    if (params.temperature <= 0.0) {
      // Temp <= 0 means greedy after the filters — pick argmax.
      b.llama_sampler_chain_add(chain, b.llama_sampler_init_greedy());
      return Sampler(chain, params);
    }

    b.llama_sampler_chain_add(
      chain,
      b.llama_sampler_init_temp(params.temperature),
    );

    b.llama_sampler_chain_add(
      chain,
      b.llama_sampler_init_dist(params.seed),
    );

    return Sampler(chain, params);
  }
}
