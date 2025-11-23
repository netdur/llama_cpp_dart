# ModelParams Class Documentation

A class that manages parameters for loading and configuring LLaMA models. It controls GPU offloading, multi-GPU splitting strategies, memory mapping, and tensor validation.

## Properties

### GPU & Hardware Configuration
- `nGpuLayers` (`int`, default: 99): Number of layers to offload to VRAM. Set to -1 or a high number to offload all layers.
- `splitMode` (`LlamaSplitMode`, default: `none`): Strategy for splitting the model across multiple GPUs.
- `mainGpu` (`int`, default: 0): The index of the GPU used for the entire model when split mode is `none`.
- `tensorSplit` (`List<double>`): Proportion of model layers/rows to offload to each GPU (for multi-GPU setups).

### Memory & Loading
- `useMemorymap` (`bool`, default: true): Enable memory mapping (mmap). Faster loading, lower system RAM usage.
- `useMemoryLock` (`bool`, default: false): Force the system to keep the model in RAM (prevent swapping).
- `vocabOnly` (`bool`, default: false): Load only the vocabulary, excluding weights.
- `checkTensors` (`bool`, default: false): Validate model tensor data on load (slow).

### Advanced Backend Settings (New in v0.2)
- `useExtraBufts` (`bool`, default: false): Use extra buffer types (used for weight repacking).
- `noHost` (`bool`, default: false): Bypass host buffer, allowing extra buffers to be used directly.

### Metadata
- `kvOverrides` (`Map<String, dynamic>`): Override key-value pairs of model metadata.
- `formatter` (`PromptFormat?`): Optional format handler associated with this model configuration.

## Enums

### LlamaSplitMode
Defines how the model is distributed across devices.
- `none` (0): Single GPU usage.
- `layer` (1): Split layers and KV cache across GPUs.
- `row` (2): Split layers and KV across GPUs with tensor parallelism.

## Methods

### `ModelParams()`
Default constructor that initializes a new instance with default values.

### `ModelParams.fromJson(Map<String, dynamic> json)`
Creates a ModelParams instance from a JSON map.

### `Map<String, dynamic> toJson()`
Converts the instance to a JSON map.

### `llama_model_params get()`
Constructs and returns a native `llama_model_params` object (FFI) populated with current settings.

### `void dispose()`
Frees native memory allocated for `tensorSplit`. **Must be called if tensorSplit is used.**

### `String toString()`
Returns a JSON string representation of the instance.

## Example Usage

### Standard GPU Load
```dart
final params = ModelParams()
  ..nGpuLayers = 99  // Offload everything
  ..useMemorymap = true;
```

### Multi-GPU Load
```dart
final params = ModelParams()
  ..nGpuLayers = 99
  ..splitMode = LlamaSplitMode.row // Tensor parallelism
  ..tensorSplit = [0.5, 0.5]; // Split evenly between 2 GPUs

// ... use params ...

// Cleanup unmanaged pointers
params.dispose(); 
```

## Notes
- **RPC Removed:** The `rpcServers` field was removed in v0.2 as it is no longer supported by the core library struct.
- **Memory Safety:** If you populate `tensorSplit`, unmanaged memory is allocated. You must call `dispose()` when you are done with the parameters (usually immediately after loading the model).