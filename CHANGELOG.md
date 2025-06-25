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
