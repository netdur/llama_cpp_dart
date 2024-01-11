# Dart Binding for llama.cpp

## Integration Guide for the llama.cpp Library

This guide outlines the steps to integrate the llama.cpp library into your Dart projects, with a specific focus on macOS/iOS environments using Xcode.

### Prerequisites
- Xcode installed on your macOS/iOS device.
- CMake installed on your system.

### Step 1: Integrating the Library into Xcode
1. **Download Pre-Built Binaries:**
   - Acquire the pre-built binaries from [llama_common_c](https://github.com/netdur/llama_common_c/releases).

2. **Code Signing (if necessary):**
   - You may need to sign `libllm.dylib`. Use `security find-identity -p codesigning -v` to find your certificate.
   - Then execute: `codesign --force --verify --verbose --sign "Apple Development: ... (...)" libllm.dylib`.

3. **Add `libllm.dylib` to Your Xcode Project:**
   - Drag `libllm.dylib` into your Xcode project.
   - Ensure you select "Copy items if needed" and opt to "Embed & Sign" in the dialog.

4. **Add Metal Shader File:**
   - Drag the `ggml-metal.metal` file into your Xcode project.
   - In the file inspector on the right, set its type to "Data".
   - Navigate to the "Build Phases" tab in your project settings.
   - Under "Copy Bundle Resources," include the [`ggml-metal.metal`](https://github.com/ggerganov/llama.cpp/blob/master/ggml-metal.metal) file.

### Understanding the Sandbox in macOS
- **Sandbox Constraints:**
  - macOS enforces sandboxing, limiting apps from accessing files outside their container directories.
  - Adjust sandbox settings in your macOS system to permit file access as required by your application.

- **Alternative Approach:**
  - Alternatively, ship models as assets and extract them within the application.

### Library Overview
- **`llama_cpp`:** This module offers direct binding for `llama.h`.
- **`LLM`:** high-level class for simplified model interaction.
- **`LlamaProcessor`:** Designed for use with Flutter widgets, this higher-level class provides an easy-to-use interface for interacting with models.

### Usage Example
For a practical implementation example, please see example.dart This example demonstrates how to use the llama.cpp library effectively within a Dart project.
