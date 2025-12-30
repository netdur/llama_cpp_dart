## 0.2.1
* **Android**: Added OpenCL support for GPU acceleration (#91).
* **Vision**: 
    * Fixed crash in `mtmd` context disposal.
    * Stable Qwen3-VL support.
* **Chat**: Added experimental support for Qwen3-VL chat format (`_exportQwen3Jinja`).
* **Fixes**:
    * Improved logging initialization (#88).
    * Fixed stream processing crash in chat.
* **Core**: Updated `llama.cpp` submodule.

## 0.2.0
* llama.cpp 4ffc47cb2001e7d523f9ff525335bbe34b1a2858
* Memory Safety: No more pointer being freed was not allocated crashes.
* UTF-8 Safety: Emojis and foreign languages won't break generation.
* Context Management: You can Append, Clear, or Save the brain.
* Multi-Tenancy: You can handle multiple users (Slots) if you need to.
* breakign changes:
    Old: Llama(path, modelParams, contextParams, samplerParams, verbose)
    New: Llama(path, modelParams: ..., contextParams: ..., samplerParams: ..., verbose: ...)
* Parameter Serialization: ContextParams and ModelParams JSON serialization has changed. Enums (like LlamaRopeScalingType) now store their specific C-API integer values instead of Dart list indices. Old JSON configs may need migration.
* Sampler Standardization: SamplerParams has been refactored to strictly match llama.cpp. Non-standard fields (e.g., xtcTemperature, topPKeep) have been renamed or removed.
* RPC Removed: Removed rpcServers from ModelParams as it is no longer supported in the core struct.


## 0.1.2+1
* forgot to update version

## 0.1.2
* removed flash_attn llama_context_default_params
* removed softmax
* updated llama.cpp to b8595b16e

## 0.1.1
* State load / save
* llama.cpp 25ff6f7659f6a5c47d6a73eada5813f0495331f0
* harmony prompting syntax
* isolate has vision and verbose support 
* mcp server / agent
* scope generation stopping

## 0.1.0
* Multimodal support - vision

## 0.0.9
* Major internal refactoring to improve code organization and maintainability
* Fixed critical bug where subsequent prompts would fail due to batch seq_id memory management
* Improved position tracking for continuous conversation support
* Enhanced error handling and debugging capabilities
* Added foundation for future chat optimization features
* Breaking change: Internal API restructuring (public API remains stable)

## 0.0.8
* disabled llava
* compatible with llama.cpp 42ae10bb
* add typed_isolate
* removed llama processor

## 0.0.7
* updated binding
* performance imporvement and bugs fix

## 0.0.6

* added initial support to load lora
* dart cli example
* fixed #3 by @danemadsen

## 0.0.5

* removed assets defination
* added static property `Llama.libraryPath` to set library path, in order to support linux and other platforms

## 0.0.4

* `ModelParams` disabled options `splitsMode`, `tensorSplit` and `metadataOverride`

## 0.0.3

* LlamaProcessor now take context and model parameters

## 0.0.2

* refactored code to follow dart package structure

## 0.0.1

* TODO: Describe initial release.
