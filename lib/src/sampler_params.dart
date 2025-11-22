class SamplerParams {
  // --- Core Llama.cpp Samplers ---
  
  // 1. Temperature & Dynamic Temp
  double temp = 0.80;
  double dynatempRange = 0.0;     // 0.0 = disabled
  double dynatempExponent = 1.0;

  // 2. Top-K & Top-P & Min-P
  int topK = 40;
  double topP = 0.95;
  double minP = 0.05;

  // 3. Tail Free / Typical / Top-N Sigma
  double typical = 1.00;
  double topNSigma = -1.0;        // -1.0 = disabled

  // 4. XTC (Exclude Top Choices)
  // Replaces your custom xtcTemperature/xtcStartValue
  double xtcProbability = 0.0;    // 0.0 = disabled
  double xtcThreshold = 0.1;      // > 0.5 disables XTC usually
  
  // 5. Mirostat
  // 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0
  int mirostat = 0; 
  double mirostatTau = 5.00;
  double mirostatEta = 0.10;
  int mirostatM = 100;

  // 6. Penalties
  int penaltyLastTokens = 64;     // repeat_last_n
  double penaltyRepeat = 1.00;    // repeat_penalty
  double penaltyFreq = 0.00;      // frequency_penalty
  double penaltyPresent = 0.00;   // presence_penalty
  bool penaltyNewline = false;    // penalize_nl
  bool ignoreEOS = false;         // ignore_eos

  // 7. DRY (Do Not Repeat Yourself)
  double dryMultiplier = 0.0;     // 0.0 = disabled
  double dryBase = 1.75;          // exponential base
  int dryAllowedLen = 2;          // allowed repetition length
  int dryPenaltyLastN = -1;       // -1 = context size
  List<String> dryBreakers = ["\n", ":", "\"", "*"]; // sequence breakers

  // 8. Grammar
  String grammarStr = "";
  String grammarRoot = "";

  // --- App-Level / Non-Standard Extras ---
  
  // Greedy decoding (equivalent to topK = 1)
  bool greedy = false; 
  
  // Softmax post-processing (Application level)
  bool softmax = true;
  
  // Seed is often generation-level, but kept here for convenience
  int seed = 0xFFFFFFFF; // -1 or maxint usually means random in llama.cpp

  // Legacy / Non-Standard Keep parameters
  // (These are likely from Oobabooga or other UIs, not core llama.cpp)
  int topPKeep = 1;
  int minPKeep = 1;
  int typicalKeep = 1;
  int xtcKeep = 1;
  int xtcLength = 1; // Not standard XTC param

  SamplerParams();

  SamplerParams.fromJson(Map<String, dynamic> json) {
    temp = (json['temp'] ?? 0.8).toDouble();
    dynatempRange = (json['dynatempRange'] ?? 0.0).toDouble();
    dynatempExponent = (json['dynatempExponent'] ?? 1.0).toDouble();
    
    topK = json['topK'] ?? 40;
    topP = (json['topP'] ?? 0.95).toDouble();
    minP = (json['minP'] ?? 0.05).toDouble();
    
    typical = (json['typical'] ?? 1.0).toDouble();
    topNSigma = (json['topNSigma'] ?? -1.0).toDouble();
    
    // Mapping old XTC fields if present to new standard ones
    if (json['xtcTemperature'] != null) {
      xtcProbability = json['xtcTemperature'];
    } else {
      xtcProbability = (json['xtcProbability'] ?? 0.0).toDouble();
    }
    
    if (json['xtcStartValue'] != null) {
      xtcThreshold = json['xtcStartValue'];
    } else {
      xtcThreshold = (json['xtcThreshold'] ?? 0.1).toDouble();
    }
    
    mirostat = json['mirostat'] ?? 0;
    // Map legacy mirostat2Tau to standard mirostatTau if set
    if (json['mirostat2Tau'] != null && mirostat == 2) {
       mirostatTau = json['mirostat2Tau'];
    } else {
       mirostatTau = (json['mirostatTau'] ?? 5.0).toDouble();
    }
    
    if (json['mirostat2Eta'] != null && mirostat == 2) {
       mirostatEta = json['mirostat2Eta'];
    } else {
       mirostatEta = (json['mirostatEta'] ?? 0.1).toDouble();
    }
    mirostatM = json['mirostatM'] ?? 100;
    
    penaltyLastTokens = json['penaltyLastTokens'] ?? 64;
    penaltyRepeat = (json['penaltyRepeat'] ?? 1.0).toDouble();
    penaltyFreq = (json['penaltyFreq'] ?? 0.0).toDouble();
    penaltyPresent = (json['penaltyPresent'] ?? 0.0).toDouble();
    penaltyNewline = json['penaltyNewline'] ?? false;
    ignoreEOS = json['ignoreEOS'] ?? false;
    
    dryMultiplier = (json['dryMultiplier'] ?? 0.0).toDouble();
    // Map legacy dryPenalty to dryBase if needed, or just load dryBase
    if (json['dryPenalty'] != null && dryMultiplier > 0) {
       // Heuristic mapping if using old format
       dryBase = (json['dryPenalty'] ?? 1.75).toDouble(); 
    } else {
       dryBase = (json['dryBase'] ?? 1.75).toDouble();
    }
    dryAllowedLen = json['dryAllowedLen'] ?? 2;
    dryPenaltyLastN = json['dryPenaltyLastN'] ?? (json['dryLookback'] ?? -1);
    if (json['dryBreakers'] != null) {
      dryBreakers = List<String>.from(json['dryBreakers']);
    }

    grammarStr = json['grammarStr'] ?? "";
    grammarRoot = json['grammarRoot'] ?? "";
    
    greedy = json['greedy'] ?? false;
    seed = json['seed'] ?? 0xFFFFFFFF;
    
    // Legacy fields
    topPKeep = json['topPKeep'] ?? 1;
    minPKeep = json['minPKeep'] ?? 1;
  }

  Map<String, dynamic> toJson() => {
    'temp': temp,
    'dynatempRange': dynatempRange,
    'dynatempExponent': dynatempExponent,
    'topK': topK,
    'topP': topP,
    'minP': minP,
    'typical': typical,
    'topNSigma': topNSigma,
    'xtcProbability': xtcProbability,
    'xtcThreshold': xtcThreshold,
    'mirostat': mirostat,
    'mirostatTau': mirostatTau,
    'mirostatEta': mirostatEta,
    'mirostatM': mirostatM,
    'penaltyLastTokens': penaltyLastTokens,
    'penaltyRepeat': penaltyRepeat,
    'penaltyFreq': penaltyFreq,
    'penaltyPresent': penaltyPresent,
    'penaltyNewline': penaltyNewline,
    'ignoreEOS': ignoreEOS,
    'dryMultiplier': dryMultiplier,
    'dryBase': dryBase,
    'dryAllowedLen': dryAllowedLen,
    'dryPenaltyLastN': dryPenaltyLastN,
    'dryBreakers': dryBreakers,
    'grammarStr': grammarStr,
    'grammarRoot': grammarRoot,
    'greedy': greedy,
    'seed': seed,
  };
}