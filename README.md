## Integration Guide for llama.cpp Library

### Prerequisites
- Xcode installed on your macOS.
- CMake installed on your system.

### Step 1: Cloning the Repository
First, clone the `llama.cpp` repository to your local machine:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### Step 2: Building the Library
Navigate to the cloned repository and set up the build environment:
```bash
mkdir build
cd build
```

Configure the build with CMake:
```bash
cmake -DBUILD_SHARED_LIBS=ON -DLLAMA_STATIC=OFF -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_C_FLAGS=-fPIC ..
```

Compile the library:
```bash
make
```

Install the library to a local directory:
```bash
make DESTDIR=./install install
```

### Step 3: Integrating the Library into Xcode
1. **Add `libllama.dylib` to Your Xcode Project:**
   - Drag `llama.cpp/build/install/usr/local/lib/libllama.dylib` into your Xcode project.
   - Ensure you select "Copy items if needed" and choose to "Embed & Sign" in the options.

2. **Add Metal Shader File:**
   - Drag the `ggml-metal.metal` file into your Xcode project.
   - In the file inspector on the right, change its type to "Data".
   - Go to the "Build Phases" tab of your project settings.
   - Under "Copy Bundle Resources," add the `ggml-metal.metal` file.

### Understanding the Sandbox in macOS
Before you begin, note that macOS enforces sandboxing, which restricts apps from accessing files outside their container directories. Make sure to configure sandbox settings appropriately in your macOS system to allow file access as needed for your application.

### Documentation for ModelManager.dart in llama_cpp Library

### Overview of llama_cpp.dart
The `llama_cpp.dart` file in the llama_cpp library is a direct 1:1 port of the `llama.h` header from the C++ library. This Dart file replicates the functionalities and interfaces of the original C++ header, providing a seamless bridge between Dart and the underlying C++ implementations.

### Purpose
The primary purpose of `llama_cpp.dart` is to offer Dart developers direct access to the low-level functions of the llama C++ library. It ensures that all the capabilities of the C++ library are available in Dart, maintaining the same function signatures and operational logic.

## Introduction to ModelManager.dart
The `ModelManager.dart` file in the llama_cpp library provides a high-level API to interact with the llama model. It offers an easier and more intuitive way to manage the model's lifecycle, from loading to executing and unloading the model. 

### Class Overview: ModelManager

`ModelManager` is a class that encapsulates the functionality needed to manage and run the llama model. Here's an overview of its properties and methods:

- **Properties:**
  - `bool disableLog`: Controls logging. Set `true` to disable logging.
  - `int nLen`: The expected size of the output.
  - `int threads`: Number of threads to use.
  - `int seed`: Seed value for token generation.
  - `int contextSize`: Size of the context.

- **Methods:**
  - `void loadModel(String path)`: Loads the model from the specified path.
  - `void unloadModel()`: Unloads the model.
  - `void setup(String prompt)`: Initializes the model with a given prompt.
  - `String getNext()`: Retrieves the next token string from the model.
  - `void createContext()`: Creates a new context for the model.
  - `void clearContext()`: Clears the existing model context.
  - `String modelInfo()`: Returns information about the model.

### Example Usage of ModelManager

```dart
ModelManager modelManager = ModelManager();

// Load the model
modelManager.loadModel("/path/to/model");

// Create and set up the context
modelManager.createContext();
modelManager.setup("hello, ");

// Generate and print tokens
while (modelManager.nCur < modelManager.nLen) {
  String token = modelManager.getNext();
  debugPrint(token);
}

// Clear the context and unload the model
modelManager.clearContext();
modelManager.unloadModel();
```

### Notes
- Replace `"/path/to/model"` with the actual path to your model file.
- Ensure that `nLen` and other properties are set according to your requirements before calling `loadModel`.
- The example demonstrates a basic use case. Customize it to fit the specific needs of your application.

### Overview of LlamaProcessor
`LlamaProcessor` is a higher-level Dart class that facilitates the use of the llama model in Flutter applications. It employs Dart isolates to ensure UI responsiveness by running model operations in a separate thread.

### Key Components
- **Isolates**: Used for executing model operations in parallel, preventing UI blocking.
- **Stream**: Facilitates asynchronous data handling, allowing the Flutter UI to remain responsive and update in real-time.

### Methods
- **Constructor (`LlamaProcessor(this.path)`)**: Initializes the processor with a model path.
- **`setup(String prompt)`**: Prepares the model with a given prompt.
- **`getNext()`**: Requests the next token/string from the model.
- **`clear()`**: Clears the current model context.
- **`unloadModel()`**: Unloads the model and cleans up resources.

### Example Usage in Flutter
The following Flutter UI code demonstrates how to use `LlamaProcessor`:

```dart
Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton(
                  onPressed: () {
                    llamaProcessor = LlamaProcessor(_modelPathController.text);
                  },
                  child: const Text('Load Model'),
                ),
                ElevatedButton(
                  onPressed: () {
                    llamaProcessor?.clear();
                    llamaProcessor?.unloadModel();
                  },
                  child: const Text('Unload Model'),
                ),
                ElevatedButton(
                  onPressed: () {
                    _resultController.text = "";
                    llamaProcessor?.stream.listen((data) {
                      _resultController.text += data;
                    }, onError: (error) {
                      _resultController.text = "Error: $error";
                    }, onDone: () {
                      llamaProcessor?.clear();
                    });
                    llamaProcessor?.setup(_promptController.text);
                    for (int i = 0; i < 600; i++) {
                      if (flagForStop == true) {
                        flagForStop = false;
                        break;
                      }
                      llamaProcessor?.getNext();
                    }
                  },
                  child: const Text('Run Prompt'),
                ),
                ElevatedButton(
                  onPressed: () {
                    flagForStop = true;
                  },
                  child: const Text('Stop Prompt'),
                ),
              ],
            )
```

see example.dart for complete example
