# ModelParams

A class that manages parameters for loading and configuring LLaMA models. It provides functionality to control GPU utilization, memory management, and model splitting across multiple GPUs.

## Properties

### Basic Configuration
- `formatter` (`PromptFormat?`): Format handler for model prompts
- `nGpuLayers` (`int`): Number of layers to store in VRAM (default: 99)
- `splitMode` (`LlamaSplitMode`): Determines how to split the model across multiple GPUs
- `mainGpu` (`int`): The GPU used for the entire model when split mode is none (default: 0)
- `tensorSplit` (`List<double>`): Proportion of model layers/rows to offload to each GPU
- `rpcServers` (`String`): Comma-separated list of RPC servers for offloading

### Memory Management
- `useMemorymap` (`bool`): Enable memory mapping when possible (default: true)
- `useMemoryLock` (`bool`): Force system to keep model in RAM (default: false)
- `vocabOnly` (`bool`): Load only the vocabulary, excluding weights (default: false)
- `checkTensors` (`bool`): Enable validation of model tensor data (default: false)

### Additional Settings
- `kvOverrides` (`Map<String, dynamic>`): Override key-value pairs of model metadata

## Methods

### `ModelParams()`
Default constructor that initializes a new instance with default values.

### `ModelParams.fromJson(Map<String, dynamic> json)`
Creates a ModelParams instance from a JSON map.

### `Map<String, dynamic> toJson()`
Converts the instance to a JSON map.

### `llama_model_params get()`
Constructs and returns a native `llama_model_params` object with current settings.

### `void dispose()`
Frees allocated memory for tensor split and RPC server pointers.

### `String toString()`
Returns a JSON string representation of the instance.

## Related Types

### LlamaSplitMode
Enum defining model splitting strategies:
- `none`: Single GPU usage
- `layer`: Split layers and KV across GPUs
- `row`: Split layers and KV across GPUs with tensor parallelism support

## Example Usage

```dart
final params = ModelParams()
  ..nGpuLayers = 50
  ..splitMode = LlamaSplitMode.layer
  ..mainGpu = 0
  ..useMemorymap = true;

// Convert to JSON
final json = params.toJson();

// Create from JSON
final loadedParams = ModelParams.fromJson(json);

// Don't forget to dispose when done
params.dispose();
```

## Notes
- Remember to call `dispose()` when done to prevent memory leaks
- The class handles automatic memory management for native pointers
- Custom GPU configurations can be set through tensorSplit and rpcServers properties