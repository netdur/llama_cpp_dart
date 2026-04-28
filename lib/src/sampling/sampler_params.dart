/// Special seed value meaning "let the runtime pick a random seed."
const int defaultSeed = 0xFFFFFFFF;

/// Declarative configuration for sampling.
///
/// Immutable, JSON-friendly. Holds no native memory.
final class SamplerParams {
  /// Random seed. [defaultSeed] (0xFFFFFFFF) means runtime-chosen.
  final int seed;

  /// Sampling temperature. `0.0` collapses to greedy.
  final double temperature;

  /// Top-K sampling. `0` disables.
  final int topK;

  /// Top-P (nucleus) sampling. `1.0` disables.
  final double topP;

  /// Minimum P sampling. `0.0` disables.
  final double minP;

  /// Locally-typical sampling. `1.0` disables.
  final double typicalP;

  /// Repeat penalty (multiplicative). `1.0` disables.
  final double repeatPenalty;

  /// Window size for repeat / frequency / presence penalties.
  final int penaltyLastN;

  /// Frequency penalty (additive, OpenAI-style). `0.0` disables.
  final double frequencyPenalty;

  /// Presence penalty (additive, OpenAI-style). `0.0` disables.
  final double presencePenalty;

  /// Force argmax sampling. Overrides every other knob.
  final bool greedy;

  const SamplerParams({
    this.seed = defaultSeed,
    this.temperature = 0.8,
    this.topK = 40,
    this.topP = 0.95,
    this.minP = 0.05,
    this.typicalP = 1.0,
    this.repeatPenalty = 1.0,
    this.penaltyLastN = 64,
    this.frequencyPenalty = 0.0,
    this.presencePenalty = 0.0,
    this.greedy = false,
  });

  /// Argmax sampling (deterministic).
  static const SamplerParams greedyDefault = SamplerParams(greedy: true);

  SamplerParams copyWith({
    int? seed,
    double? temperature,
    int? topK,
    double? topP,
    double? minP,
    double? typicalP,
    double? repeatPenalty,
    int? penaltyLastN,
    double? frequencyPenalty,
    double? presencePenalty,
    bool? greedy,
  }) {
    return SamplerParams(
      seed: seed ?? this.seed,
      temperature: temperature ?? this.temperature,
      topK: topK ?? this.topK,
      topP: topP ?? this.topP,
      minP: minP ?? this.minP,
      typicalP: typicalP ?? this.typicalP,
      repeatPenalty: repeatPenalty ?? this.repeatPenalty,
      penaltyLastN: penaltyLastN ?? this.penaltyLastN,
      frequencyPenalty: frequencyPenalty ?? this.frequencyPenalty,
      presencePenalty: presencePenalty ?? this.presencePenalty,
      greedy: greedy ?? this.greedy,
    );
  }

  Map<String, Object?> toJson() => {
        'seed': seed,
        'temperature': temperature,
        'topK': topK,
        'topP': topP,
        'minP': minP,
        'typicalP': typicalP,
        'repeatPenalty': repeatPenalty,
        'penaltyLastN': penaltyLastN,
        'frequencyPenalty': frequencyPenalty,
        'presencePenalty': presencePenalty,
        'greedy': greedy,
      };

  factory SamplerParams.fromJson(Map<String, Object?> json) => SamplerParams(
        seed: (json['seed'] as int?) ?? defaultSeed,
        temperature: (json['temperature'] as num?)?.toDouble() ?? 0.8,
        topK: (json['topK'] as int?) ?? 40,
        topP: (json['topP'] as num?)?.toDouble() ?? 0.95,
        minP: (json['minP'] as num?)?.toDouble() ?? 0.05,
        typicalP: (json['typicalP'] as num?)?.toDouble() ?? 1.0,
        repeatPenalty: (json['repeatPenalty'] as num?)?.toDouble() ?? 1.0,
        penaltyLastN: (json['penaltyLastN'] as int?) ?? 64,
        frequencyPenalty: (json['frequencyPenalty'] as num?)?.toDouble() ?? 0.0,
        presencePenalty: (json['presencePenalty'] as num?)?.toDouble() ?? 0.0,
        greedy: (json['greedy'] as bool?) ?? false,
      );
}
