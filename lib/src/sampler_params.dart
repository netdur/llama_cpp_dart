class SamplerParams {
  // Basic samplers
  bool greedy = false;
  int seed = 0; // For distribution sampler
  bool softmax = true;

  // Top-K sampling
  /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  int topK = 40;

  // Top-P (nucleus) sampling
  /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  double topP = 0.95;
  int topPKeep = 1;

  // Min-P sampling
  /// @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
  double minP = 0.05;
  int minPKeep = 1;

  // Typical sampling
  /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666
  double typical = 1.00;
  int typicalKeep = 1;

  // Temperature
  /// @details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
  double temp = 0.80;

  // XTC sampling
  /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
  double xtcTemperature = 1.0;
  double xtcStartValue = 0.1;
  int xtcKeep = 1;
  int xtcLength = 1;

  // Mirostat 1.0
  /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
  /// @param tau The target cross-entropy (or surprise) value you want to achieve for the generated text
  /// @param eta The learning rate used to update `mu` based on the error between target and observed surprisal
  /// @param m The number of tokens considered in the estimation of `s_hat`
  double mirostatTau = 5.00;
  double mirostatEta = 0.10;
  int mirostatM = 100;

  // Mirostat 2.0
  /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966
  double mirostat2Tau = 5.00;
  double mirostat2Eta = 0.10;

  // Grammar
  String grammarStr = "";
  String grammarRoot = "";

  // Penalties
  /// @details Token penalties configuration
  int penaltyLastTokens =
      64; // last n tokens to penalize (0 = disable penalty, -1 = context size)
  double penaltyRepeat = 1.00; // 1.0 = disabled
  double penaltyFreq = 0.00; // 0.0 = disabled
  double penaltyPresent = 0.00; // 0.0 = disabled
  bool penaltyNewline = false; // consider newlines as repeatable token
  bool ignoreEOS = false; // ignore end-of-sequence token

  // DRY sampler
  /// @details DRY sampler, designed by p-e-w, described in: https://github.com/oobabooga/text-generation-webui/pull/5677
  double dryPenalty =
      0.0; // DRY repetition penalty for tokens extending repetition
  double dryMultiplier =
      1.75; // multiplier * base ^ (length of sequence before token - allowed length)
  int dryAllowedLen =
      2; // tokens extending repetitions beyond this receive penalty
  int dryLookback = -1; // how many tokens to scan (-1 = context size)
  List<String> dryBreakers = ["\n", ":", "\"", "*"];

  SamplerParams();

  SamplerParams.fromJson(Map<String, dynamic> json) {
    greedy = json['greedy'] ?? greedy;
    seed = json['seed'] ?? seed;
    softmax = json['softmax'] ?? softmax;
    topK = json['topK'] ?? topK;
    topP = json['topP'] ?? topP;
    topPKeep = json['topPKeep'] ?? topPKeep;
    minP = json['minP'] ?? minP;
    minPKeep = json['minPKeep'] ?? minPKeep;
    typical = json['typical'] ?? typical;
    typicalKeep = json['typicalKeep'] ?? typicalKeep;
    temp = json['temp'] ?? temp;
    xtcTemperature = json['xtcTemperature'] ?? xtcTemperature;
    xtcStartValue = json['xtcStartValue'] ?? xtcStartValue;
    xtcKeep = json['xtcKeep'] ?? xtcKeep;
    xtcLength = json['xtcLength'] ?? xtcLength;
    mirostatTau = json['mirostatTau'] ?? mirostatTau;
    mirostatEta = json['mirostatEta'] ?? mirostatEta;
    mirostatM = json['mirostatM'] ?? mirostatM;
    mirostat2Tau = json['mirostat2Tau'] ?? mirostat2Tau;
    mirostat2Eta = json['mirostat2Eta'] ?? mirostat2Eta;
    grammarStr = json['grammarStr'] ?? grammarStr;
    grammarRoot = json['grammarRoot'] ?? grammarRoot;
    penaltyLastTokens = json['penaltyLastTokens'] ?? penaltyLastTokens;
    penaltyRepeat = json['penaltyRepeat'] ?? penaltyRepeat;
    penaltyFreq = json['penaltyFreq'] ?? penaltyFreq;
    penaltyPresent = json['penaltyPresent'] ?? penaltyPresent;
    penaltyNewline = json['penaltyNewline'] ?? penaltyNewline;
    ignoreEOS = json['ignoreEOS'] ?? ignoreEOS;
    dryPenalty = json['dryPenalty'] ?? dryPenalty;
    dryMultiplier = json['dryMultiplier'] ?? dryMultiplier;
    dryAllowedLen = json['dryAllowedLen'] ?? dryAllowedLen;
    dryLookback = json['dryLookback'] ?? dryLookback;
    dryBreakers = List<String>.from(json['dryBreakers'] ?? dryBreakers);
  }

  Map<String, dynamic> toJson() {
    return {
      'greedy': greedy,
      'seed': seed,
      'softmax': softmax,
      'topK': topK,
      'topP': topP,
      'topPKeep': topPKeep,
      'minP': minP,
      'minPKeep': minPKeep,
      'typical': typical,
      'typicalKeep': typicalKeep,
      'temp': temp,
      'xtcTemperature': xtcTemperature,
      'xtcStartValue': xtcStartValue,
      'xtcKeep': xtcKeep,
      'xtcLength': xtcLength,
      'mirostatTau': mirostatTau,
      'mirostatEta': mirostatEta,
      'mirostatM': mirostatM,
      'mirostat2Tau': mirostat2Tau,
      'mirostat2Eta': mirostat2Eta,
      'grammarStr': grammarStr,
      'grammarRoot': grammarRoot,
      'penaltyLastTokens': penaltyLastTokens,
      'penaltyRepeat': penaltyRepeat,
      'penaltyFreq': penaltyFreq,
      'penaltyPresent': penaltyPresent,
      'penaltyNewline': penaltyNewline,
      'ignoreEOS': ignoreEOS,
      'dryPenalty': dryPenalty,
      'dryMultiplier': dryMultiplier,
      'dryAllowedLen': dryAllowedLen,
      'dryLookback': dryLookback,
      'dryBreakers': dryBreakers,
    };
  }
}
