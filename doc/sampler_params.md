# SamplerParams Class Documentation

A class that manages sampling strategies for text generation. It controls how the model selects the next token from the predicted probabilities. This class is strictly aligned with the `llama.cpp` C-API to ensure consistent behavior.

## Basic Sampling

### Temperature & Entropy
- `temp` (`double`, default: 0.80): Controls randomness. Lower values (e.g., 0.1) make the model deterministic; higher values (e.g., 1.0) make it creative.
- `dynatempRange` (`double`, default: 0.0): Dynamic Temperature range. If > 0, temperature varies based on the entropy of the distribution.
- `dynatempExponent` (`double`, default: 1.0): Controls the curve of the dynamic temperature adjustment.

### Top-K & Nucleus
- `topK` (`int`, default: 40): Keeps only the K most likely tokens.
- `topP` (`double`, default: 0.95): Nucleus sampling. Keeps the top tokens whose cumulative probability adds up to P.
- `minP` (`double`, default: 0.05): Minimum P sampling. Discards tokens with probability smaller than `minP * probability_of_most_likely_token`.

### Advanced / Tail Free
- `typical` (`double`, default: 1.00): Locally Typical Sampling.
- `topNSigma` (`double`, default: -1.0): Top-n-sigma sampling. -1 disables it.

## Specialized Samplers

### XTC (Exclude Top Choices)
Removes "cliché" tokens (the absolute top choices) to force the model to be more creative, while keeping the rest of the distribution.
- `xtcProbability` (`double`, default: 0.0): Chance to trigger XTC (0.0 = disabled).
- `xtcThreshold` (`double`, default: 0.1): Probability threshold. If a token is > 0.5 (usually), XTC is disabled to prevent ruining logic.

### DRY (Do Not Repeat Yourself)
A modern repetition penalty that works better than standard penalties for long context.
- `dryMultiplier` (`double`, default: 0.0): Penalty strength (0.0 = disabled).
- `dryBase` (`double`, default: 1.75): Exponential base.
- `dryAllowedLen` (`int`, default: 2): Allowed repetition length before penalty kicks in.
- `dryPenaltyLastN` (`int`, default: -1): How far back to scan (-1 = full context).
- `dryBreakers` (`List<String>`): List of characters that break a repetition sequence (defaults to newline, colon, quotes).

### Mirostat
An algorithm that actively maintains the "surprise" value of text at a target level.
- `mirostat` (`int`, default: 0): Mode flag.
  - `0`: Disabled.
  - `1`: Mirostat v1.
  - `2`: Mirostat v2 (Recommended).
- `mirostatTau` (`double`, default: 5.00): Target entropy (surprise).
- `mirostatEta` (`double`, default: 0.10): Learning rate.
- `mirostatM` (`int`, default: 100): Token window size (v1 only).

## Standard Penalties
Legacy repetition penalties.
- `penaltyLastTokens` (`int`, default: 64): Lookback window (`repeat_last_n`).
- `penaltyRepeat` (`double`, default: 1.00): Strict repetition penalty.
- `penaltyFreq` (`double`, default: 0.00): Frequency penalty (OpenAI style).
- `penaltyPresent` (`double`, default: 0.00): Presence penalty (OpenAI style).
- `penaltyNewline` (`bool`, default: false): If true, newlines are penalized.
- `ignoreEOS` (`bool`, default: false): If true, the End-of-Sequence token is banned (forces generation to continue).

## Grammar
- `grammarStr` (`String`): GBNF grammar string to constrain output (e.g., force JSON).
- `grammarRoot` (`String`): Root rule name.

## App-Level Settings
- `greedy` (`bool`, default: false): If true, disables all probabilistic sampling and selects the token with the highest logit.
- `seed` (`int`, default: 0xFFFFFFFF): Random seed. -1 or MaxInt usually indicates random.

## Methods

### `SamplerParams()`
Creates a new instance with default values.

### `SamplerParams.fromJson(Map<String, dynamic> json)`
Creates an instance from a JSON map. Automatically maps legacy fields (like `xtcTemperature`) to new standard names.

### `Map<String, dynamic> toJson()`
Converts the instance to a JSON map.

## Example Usage

### Creative Writing (Story)
```dart
final params = SamplerParams()
  ..temp = 0.9
  ..minP = 0.05
  ..xtcProbability = 0.5 // Remove clichés
  ..xtcThreshold = 0.1;
```

### Logical / Coding (Strict)
```dart
final params = SamplerParams()
  ..temp = 0.1
  ..topK = 10
  ..penaltyRepeat = 1.1; // Slight penalty to prevent loops
```

### Force JSON Output
```dart
final params = SamplerParams()
  ..temp = 0.1
  ..grammarStr = r'''root ::= object ...'''; // GBNF string
```