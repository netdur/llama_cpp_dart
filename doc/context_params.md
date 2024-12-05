# ContextParams Class Documentation

A class that manages configuration settings for the Llama model context, controlling various aspects of model execution including threading, batching, and performance optimizations.

## Core Configuration

### Context and Batch Settings
- `nPredit` (`int`, default: 32): Maximum tokens to predict/generate
- `nCtx` (`int`, default: 512): Text context size (0 = use model default)
- `nBatch` (`int`, default: 512): Logical maximum batch size for llama_decode
- `nUbatch` (`int`, default: 512): Physical maximum batch size
- `nSeqMax` (`int`, default: 1): Maximum number of sequences

### Threading
- `nThreads` (`int`, default: 8): Number of threads for generation
- `nThreadsBatch` (`int`, default: 8): Number of threads for batch processing

## RoPE (Rotary Position Embedding) Configuration

### Basic RoPE Settings
- `ropeScalingType` (`LlamaRopeScalingType`, default: unspecified): RoPE scaling type
- `ropeFreqBase` (`double`, default: 0.0): Base frequency (0 = from model)
- `ropeFreqScale` (`double`, default: 0.0): Frequency scaling factor (0 = from model)

### YaRN Parameters
- `yarnExtFactor` (`double`, default: -1.0): Extrapolation mix factor
- `yarnAttnFactor` (`double`, default: 1.0): Magnitude scaling factor
- `yarnBetaFast` (`double`, default: 32.0): Low correction dimension
- `yarnBetaSlow` (`double`, default: 1.0): High correction dimension
- `yarnOrigCtx` (`int`, default: 0): Original context size

## Embedding Configuration
- `poolingType` (`LlamaPoolingType`, default: unspecified): Pooling type for embeddings
- `attentionType` (`LlamaAttentionType`, default: unspecified): Attention type for embeddings
- `embeddings` (`bool`, default: false): Extract embeddings with logits

## Performance Settings
- `defragThold` (`double`, default: -1.0): KV cache defragmentation threshold
- `logitsAll` (`bool`, default: false): Compute all logits in llama_decode
- `offloadKqv` (`bool`, default: true): Offload KQV operations to GPU
- `flashAttn` (`bool`, default: false): Use flash attention (experimental)
- `noPerfTimings` (`bool`, default: false): Disable performance timing measurements

## Enums

### LlamaRopeScalingType
```dart
enum LlamaRopeScalingType {
  unspecified(-1),
  none(0),
  linear(1),
  yarn(2),
  maxValue(2)
}
```

### LlamaPoolingType
```dart
enum LlamaPoolingType {
  unspecified(-1),
  none(0),
  mean(1),
  cls(2),
  last(3),
  rank(4)
}
```

### LlamaAttentionType
```dart
enum LlamaAttentionType {
  unspecified(-1),
  causal(0),
  nonCausal(1)
}
```

## Methods

### `ContextParams()`
Creates a new instance with default values.

### `llama_context_params get()`
Returns a native `llama_context_params` object with current settings.

### `ContextParams.fromJson(Map<String, dynamic> json)`
Creates an instance from a JSON map.

### `Map<String, dynamic> toJson()`
Converts the instance to a JSON map.

### `String toString()`
Returns a JSON string representation of the instance.

## Example Usage

```dart
final params = ContextParams()
  ..nCtx = 1024
  ..nThreads = 4
  ..ropeScalingType = LlamaRopeScalingType.linear
  ..offloadKqv = true;

// Convert to JSON
final json = params.toJson();

// Create from JSON
final loadedParams = ContextParams.fromJson(json);
```