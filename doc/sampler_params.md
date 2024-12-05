# SamplerParams Class Documentation

A class that manages various sampling parameters and strategies for text generation models. This class provides comprehensive control over different sampling methods, penalties, and configurations that affect the text generation process.

## Basic Sampling Parameters

### Greedy and Basic Settings
- `greedy` (`bool`, default: false): Enables greedy sampling
- `seed` (`int`, default: 0): Seed for random number generation in distribution sampling
- `softmax` (`bool`, default: true): Whether to use softmax for token distribution

### Temperature Sampling
- `temp` (`double`, default: 0.80): Controls randomness in sampling
  - When t ≤ 0.0, keeps maximum logit at original value, sets others to -inf
  - Higher values increase randomness, lower values make sampling more deterministic

## Advanced Sampling Methods

### Top-K Sampling
- `topK` (`int`, default: 40): Limits sampling to K most likely tokens
- Reference: "The Curious Case of Neural Text Degeneration" (https://arxiv.org/abs/1904.09751)

### Top-P (Nucleus) Sampling
- `topP` (`double`, default: 0.95): Cumulative probability threshold
- `topPKeep` (`int`, default: 1): Number of tokens to keep in Top-P sampling
- Reference: Same paper as Top-K sampling

### Min-P Sampling
- `minP` (`double`, default: 0.05): Minimum probability threshold
- `minPKeep` (`int`, default: 1): Number of tokens to keep
- Reference: Implementation from llama.cpp (PR #3841)

### Typical Sampling
- `typical` (`double`, default: 1.00): Typicality parameter
- `typicalKeep` (`int`, default: 1): Number of tokens to keep
- Reference: "Locally Typical Sampling" (https://arxiv.org/abs/2202.00666)

### XTC Sampling
- `xtcTemperature` (`double`, default: 1.0): Temperature parameter for XTC
- `xtcStartValue` (`double`, default: 0.1): Starting value
- `xtcKeep` (`int`, default: 1): Number of tokens to keep
- `xtcLength` (`int`, default: 1): Length parameter
- Reference: Implementation from text-generation-webui (PR #6335)

## Mirostat Sampling

### Mirostat 1.0
- `mirostatTau` (`double`, default: 5.00): Target cross-entropy
- `mirostatEta` (`double`, default: 0.10): Learning rate
- `mirostatM` (`int`, default: 100): Token window size
- Reference: https://arxiv.org/abs/2007.14966

### Mirostat 2.0
- `mirostat2Tau` (`double`, default: 5.00): Target cross-entropy
- `mirostat2Eta` (`double`, default: 0.10): Learning rate
- Reference: Same paper as Mirostat 1.0

## Grammar Control
- `grammarStr` (`String`, default: ""): Grammar specification string
- `grammarRoot` (`String`, default: ""): Root rule for grammar

## Penalty Configuration

### Token Penalties
- `penaltyLastTokens` (`int`, default: 64): Number of last tokens to penalize
  - 0: Disable penalty
  - -1: Use full context size
- `penaltyRepeat` (`double`, default: 1.00): Repetition penalty (1.0 = disabled)
- `penaltyFreq` (`double`, default: 0.00): Frequency penalty (0.0 = disabled)
- `penaltyPresent` (`double`, default: 0.00): Presence penalty (0.0 = disabled)
- `penaltyNewline` (`bool`, default: false): Consider newlines as repeatable tokens
- `ignoreEOS` (`bool`, default: false): Ignore end-of-sequence token

### DRY Sampler
- `dryPenalty` (`double`, default: 0.0): Penalty for repetition
- `dryMultiplier` (`double`, default: 1.75): Base multiplier
- `dryAllowedLen` (`int`, default: 2): Allowed repetition length
- `dryLookback` (`int`, default: -1): Lookback window (-1 = context size)
- `dryBreakers` (`List<String>`, default: ["\n", ":", "\"", "*"]): Tokens that break repetition
- Reference: Implementation by p-e-w (text-generation-webui PR #5677)

## Methods

### `SamplerParams()`
Creates a new instance with default values.

### `SamplerParams.fromJson(Map<String, dynamic> json)`
Creates an instance from a JSON map.

### `Map<String, dynamic> toJson()`
Converts the instance to a JSON map.