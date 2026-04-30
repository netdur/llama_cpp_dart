/// Special seed value meaning "let the runtime pick a random seed."
const int defaultSeed = 0xFFFFFFFF;

/// Mirostat algorithm version.
enum MirostatVersion {
  /// Disabled.
  off,

  /// Mirostat 1.0 (`llama_sampler_init_mirostat`).
  v1,

  /// Mirostat 2.0 (`llama_sampler_init_mirostat_v2`).
  v2,
}

/// Mirostat sampler configuration. Replaces top-k/top-p/min-p/typical/temp/dist
/// when [version] is not [MirostatVersion.off] — Mirostat is a terminal sampler.
final class MirostatConfig {
  final MirostatVersion version;

  /// Target cross-entropy. Higher = more surprising / diverse text.
  final double tau;

  /// Learning rate for the running surprisal estimate.
  final double eta;

  /// (Mirostat v1 only) Number of tokens used to estimate `s_hat`.
  final int m;

  const MirostatConfig({
    this.version = MirostatVersion.off,
    this.tau = 5.0,
    this.eta = 0.1,
    this.m = 100,
  });

  static const MirostatConfig disabled = MirostatConfig();

  bool get enabled => version != MirostatVersion.off;

  Map<String, Object?> toJson() => {
        'version': version.name,
        'tau': tau,
        'eta': eta,
        'm': m,
      };

  factory MirostatConfig.fromJson(Map<String, Object?> json) => MirostatConfig(
        version: MirostatVersion.values.firstWhere(
          (e) => e.name == json['version'],
          orElse: () => MirostatVersion.off,
        ),
        tau: (json['tau'] as num?)?.toDouble() ?? 5.0,
        eta: (json['eta'] as num?)?.toDouble() ?? 0.1,
        m: (json['m'] as int?) ?? 100,
      );
}

/// Dynamic temperature (entropy-aware) configuration. Activated when
/// [range] > 0; replaces the static `temperature` stage.
final class DynamicTempConfig {
  /// Width of the temperature range; final temp varies in
  /// `[temperature - range, temperature + range]`.
  final double range;

  /// Curve exponent; higher values flatten the response.
  final double exponent;

  const DynamicTempConfig({
    this.range = 0.0,
    this.exponent = 1.0,
  });

  static const DynamicTempConfig disabled = DynamicTempConfig();

  bool get enabled => range > 0;

  Map<String, Object?> toJson() => {
        'range': range,
        'exponent': exponent,
      };

  factory DynamicTempConfig.fromJson(Map<String, Object?> json) =>
      DynamicTempConfig(
        range: (json['range'] as num?)?.toDouble() ?? 0.0,
        exponent: (json['exponent'] as num?)?.toDouble() ?? 1.0,
      );
}

/// XTC (eXclude Top Choices) sampler configuration. Disabled when
/// [probability] <= 0.
final class XtcConfig {
  /// Probability of triggering the truncation step. `0` disables.
  final double probability;

  /// Threshold below which tokens are excluded.
  final double threshold;

  /// Minimum number of tokens kept after truncation.
  final int minKeep;

  /// RNG seed for the trigger draw.
  final int seed;

  const XtcConfig({
    this.probability = 0.0,
    this.threshold = 0.1,
    this.minKeep = 1,
    this.seed = defaultSeed,
  });

  static const XtcConfig disabled = XtcConfig();

  bool get enabled => probability > 0;

  Map<String, Object?> toJson() => {
        'probability': probability,
        'threshold': threshold,
        'minKeep': minKeep,
        'seed': seed,
      };

  factory XtcConfig.fromJson(Map<String, Object?> json) => XtcConfig(
        probability: (json['probability'] as num?)?.toDouble() ?? 0.0,
        threshold: (json['threshold'] as num?)?.toDouble() ?? 0.1,
        minKeep: (json['minKeep'] as int?) ?? 1,
        seed: (json['seed'] as int?) ?? defaultSeed,
      );
}

/// DRY (Don't Repeat Yourself) sampler configuration. Disabled when
/// [multiplier] <= 0.
final class DryConfig {
  /// Penalty multiplier. `0` disables.
  final double multiplier;

  /// Penalty base.
  final double base;

  /// Minimum match length before a penalty is applied.
  final int allowedLength;

  /// Window size in tokens; `-1` means context size.
  final int penaltyLastN;

  /// Strings that break sequence matching (e.g. `'\n'`, `':'`).
  final List<String> seqBreakers;

  const DryConfig({
    this.multiplier = 0.0,
    this.base = 1.75,
    this.allowedLength = 2,
    this.penaltyLastN = -1,
    this.seqBreakers = const [],
  });

  static const DryConfig disabled = DryConfig();

  bool get enabled => multiplier > 0;

  Map<String, Object?> toJson() => {
        'multiplier': multiplier,
        'base': base,
        'allowedLength': allowedLength,
        'penaltyLastN': penaltyLastN,
        'seqBreakers': seqBreakers,
      };

  factory DryConfig.fromJson(Map<String, Object?> json) => DryConfig(
        multiplier: (json['multiplier'] as num?)?.toDouble() ?? 0.0,
        base: (json['base'] as num?)?.toDouble() ?? 1.75,
        allowedLength: (json['allowedLength'] as int?) ?? 2,
        penaltyLastN: (json['penaltyLastN'] as int?) ?? -1,
        seqBreakers: ((json['seqBreakers'] as List?)?.cast<String>()) ??
            const <String>[],
      );
}

/// Adaptive-P sampler configuration. Disabled when [target] < 0.
///
/// adaptive-p is a terminal sampler — when enabled, the chain skips `dist`.
final class AdaptivePConfig {
  /// Target probability in `[0.0, 1.0]`. Negative disables.
  final double target;

  /// EMA decay; history ≈ `1 / (1 - decay)` tokens.
  final double decay;

  /// RNG seed.
  final int seed;

  const AdaptivePConfig({
    this.target = -1.0,
    this.decay = 0.96,
    this.seed = defaultSeed,
  });

  static const AdaptivePConfig disabled = AdaptivePConfig();

  bool get enabled => target >= 0;

  Map<String, Object?> toJson() => {
        'target': target,
        'decay': decay,
        'seed': seed,
      };

  factory AdaptivePConfig.fromJson(Map<String, Object?> json) => AdaptivePConfig(
        target: (json['target'] as num?)?.toDouble() ?? -1.0,
        decay: (json['decay'] as num?)?.toDouble() ?? 0.96,
        seed: (json['seed'] as int?) ?? defaultSeed,
      );
}

/// GBNF grammar constraint. Disabled when [grammar] is null or empty.
///
/// When [triggerPatterns] or [triggerTokens] is non-empty the grammar is
/// installed in lazy mode: it only activates after a trigger fires.
final class GrammarConfig {
  /// GBNF grammar source. Null/empty disables this stage.
  final String? grammar;

  /// Start symbol.
  final String root;

  /// Lazy-trigger regex patterns (matched from the start of generation).
  final List<String> triggerPatterns;

  /// Lazy-trigger tokens.
  final List<int> triggerTokens;

  const GrammarConfig({
    this.grammar,
    this.root = 'root',
    this.triggerPatterns = const [],
    this.triggerTokens = const [],
  });

  static const GrammarConfig disabled = GrammarConfig();

  bool get enabled => grammar != null && grammar!.isNotEmpty;
  bool get lazy => triggerPatterns.isNotEmpty || triggerTokens.isNotEmpty;

  Map<String, Object?> toJson() => {
        'grammar': grammar,
        'root': root,
        'triggerPatterns': triggerPatterns,
        'triggerTokens': triggerTokens,
      };

  factory GrammarConfig.fromJson(Map<String, Object?> json) => GrammarConfig(
        grammar: json['grammar'] as String?,
        root: (json['root'] as String?) ?? 'root',
        triggerPatterns:
            ((json['triggerPatterns'] as List?)?.cast<String>()) ?? const [],
        triggerTokens:
            ((json['triggerTokens'] as List?)?.cast<int>()) ?? const [],
      );
}

/// One token-level logit bias entry.
final class LogitBiasEntry {
  final int token;
  final double bias;
  const LogitBiasEntry(this.token, this.bias);

  Map<String, Object?> toJson() => {'token': token, 'bias': bias};

  factory LogitBiasEntry.fromJson(Map<String, Object?> json) => LogitBiasEntry(
        json['token']! as int,
        (json['bias'] as num).toDouble(),
      );
}

/// Declarative configuration for sampling.
///
/// Immutable, JSON-friendly. Holds no native memory.
final class SamplerParams {
  /// Random seed. [defaultSeed] (0xFFFFFFFF) means runtime-chosen.
  final int seed;

  /// Sampling temperature. `0.0` collapses to greedy (after filters).
  final double temperature;

  /// Top-K sampling. `0` disables.
  final int topK;

  /// Top-P (nucleus) sampling. `1.0` disables.
  final double topP;

  /// Minimum P sampling. `0.0` disables.
  final double minP;

  /// Locally-typical sampling. `1.0` disables.
  final double typicalP;

  /// Top-n-sigma sampling. `<= 0` disables.
  final double topNSigma;

  /// Minimum tokens kept by top-p / min-p / typical / xtc filters. Most
  /// llama.cpp call sites use `1`.
  final int minKeep;

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

  /// Enable infill (fill-in-the-middle) sampler. Goes after top-k/top-p.
  final bool infill;

  /// Mirostat configuration (terminal sampler when enabled).
  final MirostatConfig mirostat;

  /// Dynamic temperature configuration.
  final DynamicTempConfig dynamicTemp;

  /// XTC sampler configuration.
  final XtcConfig xtc;

  /// DRY sampler configuration.
  final DryConfig dry;

  /// Adaptive-P sampler configuration (terminal sampler when enabled).
  final AdaptivePConfig adaptiveP;

  /// Grammar / GBNF constraint.
  final GrammarConfig grammar;

  /// Token-level logit biases applied at the start of the chain.
  final List<LogitBiasEntry> logitBias;

  const SamplerParams({
    this.seed = defaultSeed,
    this.temperature = 0.8,
    this.topK = 40,
    this.topP = 0.95,
    this.minP = 0.05,
    this.typicalP = 1.0,
    this.topNSigma = 0.0,
    this.minKeep = 1,
    this.repeatPenalty = 1.0,
    this.penaltyLastN = 64,
    this.frequencyPenalty = 0.0,
    this.presencePenalty = 0.0,
    this.greedy = false,
    this.infill = false,
    this.mirostat = MirostatConfig.disabled,
    this.dynamicTemp = DynamicTempConfig.disabled,
    this.xtc = XtcConfig.disabled,
    this.dry = DryConfig.disabled,
    this.adaptiveP = AdaptivePConfig.disabled,
    this.grammar = GrammarConfig.disabled,
    this.logitBias = const [],
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
    double? topNSigma,
    int? minKeep,
    double? repeatPenalty,
    int? penaltyLastN,
    double? frequencyPenalty,
    double? presencePenalty,
    bool? greedy,
    bool? infill,
    MirostatConfig? mirostat,
    DynamicTempConfig? dynamicTemp,
    XtcConfig? xtc,
    DryConfig? dry,
    AdaptivePConfig? adaptiveP,
    GrammarConfig? grammar,
    List<LogitBiasEntry>? logitBias,
  }) {
    return SamplerParams(
      seed: seed ?? this.seed,
      temperature: temperature ?? this.temperature,
      topK: topK ?? this.topK,
      topP: topP ?? this.topP,
      minP: minP ?? this.minP,
      typicalP: typicalP ?? this.typicalP,
      topNSigma: topNSigma ?? this.topNSigma,
      minKeep: minKeep ?? this.minKeep,
      repeatPenalty: repeatPenalty ?? this.repeatPenalty,
      penaltyLastN: penaltyLastN ?? this.penaltyLastN,
      frequencyPenalty: frequencyPenalty ?? this.frequencyPenalty,
      presencePenalty: presencePenalty ?? this.presencePenalty,
      greedy: greedy ?? this.greedy,
      infill: infill ?? this.infill,
      mirostat: mirostat ?? this.mirostat,
      dynamicTemp: dynamicTemp ?? this.dynamicTemp,
      xtc: xtc ?? this.xtc,
      dry: dry ?? this.dry,
      adaptiveP: adaptiveP ?? this.adaptiveP,
      grammar: grammar ?? this.grammar,
      logitBias: logitBias ?? this.logitBias,
    );
  }

  Map<String, Object?> toJson() => {
        'seed': seed,
        'temperature': temperature,
        'topK': topK,
        'topP': topP,
        'minP': minP,
        'typicalP': typicalP,
        'topNSigma': topNSigma,
        'minKeep': minKeep,
        'repeatPenalty': repeatPenalty,
        'penaltyLastN': penaltyLastN,
        'frequencyPenalty': frequencyPenalty,
        'presencePenalty': presencePenalty,
        'greedy': greedy,
        'infill': infill,
        'mirostat': mirostat.toJson(),
        'dynamicTemp': dynamicTemp.toJson(),
        'xtc': xtc.toJson(),
        'dry': dry.toJson(),
        'adaptiveP': adaptiveP.toJson(),
        'grammar': grammar.toJson(),
        'logitBias': logitBias.map((e) => e.toJson()).toList(),
      };

  factory SamplerParams.fromJson(Map<String, Object?> json) => SamplerParams(
        seed: (json['seed'] as int?) ?? defaultSeed,
        temperature: (json['temperature'] as num?)?.toDouble() ?? 0.8,
        topK: (json['topK'] as int?) ?? 40,
        topP: (json['topP'] as num?)?.toDouble() ?? 0.95,
        minP: (json['minP'] as num?)?.toDouble() ?? 0.05,
        typicalP: (json['typicalP'] as num?)?.toDouble() ?? 1.0,
        topNSigma: (json['topNSigma'] as num?)?.toDouble() ?? 0.0,
        minKeep: (json['minKeep'] as int?) ?? 1,
        repeatPenalty: (json['repeatPenalty'] as num?)?.toDouble() ?? 1.0,
        penaltyLastN: (json['penaltyLastN'] as int?) ?? 64,
        frequencyPenalty: (json['frequencyPenalty'] as num?)?.toDouble() ?? 0.0,
        presencePenalty: (json['presencePenalty'] as num?)?.toDouble() ?? 0.0,
        greedy: (json['greedy'] as bool?) ?? false,
        infill: (json['infill'] as bool?) ?? false,
        mirostat: json['mirostat'] is Map
            ? MirostatConfig.fromJson(
                (json['mirostat'] as Map).cast<String, Object?>())
            : MirostatConfig.disabled,
        dynamicTemp: json['dynamicTemp'] is Map
            ? DynamicTempConfig.fromJson(
                (json['dynamicTemp'] as Map).cast<String, Object?>())
            : DynamicTempConfig.disabled,
        xtc: json['xtc'] is Map
            ? XtcConfig.fromJson((json['xtc'] as Map).cast<String, Object?>())
            : XtcConfig.disabled,
        dry: json['dry'] is Map
            ? DryConfig.fromJson((json['dry'] as Map).cast<String, Object?>())
            : DryConfig.disabled,
        adaptiveP: json['adaptiveP'] is Map
            ? AdaptivePConfig.fromJson(
                (json['adaptiveP'] as Map).cast<String, Object?>())
            : AdaptivePConfig.disabled,
        grammar: json['grammar'] is Map
            ? GrammarConfig.fromJson(
                (json['grammar'] as Map).cast<String, Object?>())
            : GrammarConfig.disabled,
        logitBias: ((json['logitBias'] as List?) ?? const [])
            .cast<Map<String, Object?>>()
            .map(LogitBiasEntry.fromJson)
            .toList(growable: false),
      );
}
