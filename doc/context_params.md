# ContextParams Class Documentation

A class that manages configuration settings for the Llama model context. It controls memory allocation, performance optimizations (Flash Attention, Quantization), threading, and batch processing.

## Core Configuration

### Context and Batch Settings
- `nPredict` (`int`, default: -1): Maximum tokens to predict/generate (-1 = infinite/until EOS).
- `nCtx` (`int`, default: 512): Text context size (0 = use model default).
- `nBatch` (`int`, default: 512): Logical maximum batch size for `llama_decode`.
- `nUbatch` (`int`, default: 512): Physical maximum batch size.
- `nSeqMax` (`int`, default: 1): Maximum number of sequences (for parallel decoding).

### Threading
- `nThreads` (`int`, default: 8): Number of threads for generation.
- `nThreadsBatch` (`int`, default: 8): Number of threads for batch processing.

## Memory & Performance Optimization (New in v0.2)

### KV Cache Quantization
Reduces VRAM usage significantly by compressing the context memory.
- `typeK` (`LlamaKvCacheType`, default: `f16`): Data type for Key cache.
  *   *Recommendation:* Use `q8_0` for ~50% VRAM saving with negligible quality loss.
- `typeV` (`LlamaKvCacheType`, default: `f16`): Data type for Value cache.

### Flash Attention
- `flashAttention` (`LlamaFlashAttnType`, default: `disabled`): Enables Flash Attention.
  *   *Recommendation:* Enable for Apple Silicon (Metal) and modern NVIDIA GPUs for faster processing.

### Advanced Optimizations
- `offloadKqv` (`bool`, default: true): Offload KQV operations to GPU.
- `opOffload` (`bool`, default: true): Offload host tensor operations to device.
- `swaFull` (`bool`, default: false): Use full-size SWA (Sliding Window Attention) cache.
- `kvUnified` (`bool`, default: false): Use unified KV buffer (disable if `nSeqMax > 1`).
- `defragThold` (`double`, default: -1.0): Threshold for KV cache defragmentation.

## RoPE (Rotary Position Embedding) Configuration

### Basic RoPE Settings
- `ropeScalingType` (`LlamaRopeScalingType`, default: unspecified): RoPE scaling type.
- `ropeFreqBase` (`double`, default: 0.0): Base frequency.
- `ropeFreqScale` (`double`, default: 0.0): Frequency scaling factor.

### YaRN Parameters
- `yarnExtFactor` (`double`, default: -1.0): Extrapolation mix factor.
- `yarnAttnFactor` (`double`, default: 1.0): Magnitude scaling factor.
- `yarnBetaFast` (`double`, default: 32.0): Low correction dimension.
- `yarnBetaSlow` (`double`, default: 1.0): High correction dimension.
- `yarnOrigCtx` (`int`, default: 0): Original context size.

## General Flags
- `embeddings` (`bool`, default: false): Extract embeddings with logits.
- `logitsAll` (`bool`, default: false): Compute all logits in `llama_decode`.
- `noPerfTimings` (`bool`, default: false): Disable performance timing measurements.

## Enums

### LlamaKvCacheType
Controls the precision of the context memory.
```dart
enum LlamaKvCacheType {
  f32(0),
  f16(1),  // Default (High Precision)
  q4_0(2), // High Compression
  q4_1(3),
  q5_0(6),
  q5_1(7),
  q8_0(8)  // Balanced (Recommended)
}
```

### LlamaFlashAttnType
```dart
enum LlamaFlashAttnType {
  disabled(0),
  enabled(1)
}
```

### LlamaRopeScalingType
```dart
enum LlamaRopeScalingType {
  unspecified(-1),
  none(0),
  linear(1),
  yarn(2),
  longrope(3),
  maxValue(3)
}
```

## Methods

### `ContextParams()`
Creates a new instance with default values.

### `llama_context_params get()`
Returns a native `llama_context_params` object (FFI) populated with current settings.

### `ContextParams.fromJson(Map<String, dynamic> json)`
Creates an instance from a JSON map.

### `Map<String, dynamic> toJson()`
Converts the instance to a JSON map.

### `String toString()`
Returns a JSON string representation of the instance.

## Example Usage

### High Performance Configuration
```dart
final params = ContextParams()
  ..nCtx = 8192
  ..nBatch = 512
  // Save VRAM by quantizing context
  ..typeK = LlamaKvCacheType.q8_0
  ..typeV = LlamaKvCacheType.q8_0
  // Speed up on Mac/NVIDIA
  ..flashAttention = LlamaFlashAttnType.enabled;
```

### Standard Configuration
```dart
final params = ContextParams()
  ..nCtx = 2048
  ..nThreads = 4
  ..offloadKqv = true;
```