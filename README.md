# LLAMA.CPP DART

## Overview

This Dart library, powered by llama.cpp, enables advanced text generation for both Dart console and Flutter mobile apps. It offers a simple interface for high-performance text generation, making it ideal for a wide range of applications seeking dynamic content.

## Features

- Enables high-performance, asynchronous text generation using Dart isolates.
- Offers flexible configuration through customizable model and context parameters.
- Supports real-time text generation in Flutter apps with stream-based output.

## Getting Started

To begin using the llama.cpp Dart Library, ensure you meet the necessary prerequisites and follow the outlined steps. This package is designed for Dart, not as a Flutter plugin.

### Building `llama.cpp` Library

1. **Obtain the Library**: Download or clone the `llama.cpp` library from its [GitHub repository](https://github.com/ggerganov/llama.cpp).
2. **Platform-Specific Build**: Compile `llama.cpp` into a shared library using your system's C++ compiler. The output will be a .dll, .so, or .dylib file, depending on your operating system.
3. **Integrate with Your Dart Application**: Move the compiled shared library to a directory accessible by your Dart application.

### Prerequisites

- Dart SDK (for console application)
- Flutter SDK (for Flutter application)
- Additional dependencies as per your project requirements

### Installation

See sample code

## Usage (Sample Code)

### Dart Console Application

```dart
import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  ContextParams contextParams = ContextParams();
  int size = 32768;
  size = 8192 * 4;
  contextParams.batch = 8192 ~/ 4;
  contextParams.context = size;
  contextParams.ropeFreqBase = 57200 * 4;
  contextParams.ropeFreqScale = 0.75 / 4;

  Llama llama = Llama(
      "mistral-7b-openorca.Q5_K_M.gguf", // Change this to the path of your model
      ModelParams(),
      contextParams);

  llama.setPrompt("Your prompt here"); // Change this to your prompt

  // Asynchronous generation
  await for (String token in llama.prompt(prompt)) {
    stdout.write(token);
  }

  // Synchronous generation
  while (true) {
    var (token, done) = llama.getNext();
    stdout.write(token);
    if (done) {
      break;
    }
  }

  llama.dispose(); // Clean up
}
```

### Flutter Application

```dart
import 'dart:async';

import 'package:flutter/material.dart';

import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  runApp(const App());
}

class App extends StatelessWidget {
  const App({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter llama.cpp Demo',
      themeMode: ThemeMode.dark,
      darkTheme: ThemeData.dark(
        useMaterial3: true,
      ),
      home: const LandingPage(),
    );
  }
}

class LandingPage extends StatefulWidget {
  const LandingPage({super.key});

  @override
  State<LandingPage> createState() => _LandingPageState();
}

class _LandingPageState extends State<LandingPage> {
  final TextEditingController _modelPathController = TextEditingController();
  final TextEditingController _promptController = TextEditingController();
  final TextEditingController _resultController = TextEditingController();

  LlamaProcessor? llamaProcessor;
  StreamSubscription<String>? _streamSubscription;
  bool isModelLoaded = false;

  @override
  void initState() {
    super.initState();
    _modelPathController.text = "";
    _promptController.text = "### Human: divide by zero please\n### Assistant:";
    // _extractModel();
  }

  /*static */ _extractModel() async {
    String model = "phi-2-dpo.Q5_K_S.gguf";

    final directory = await getApplicationDocumentsDirectory();
    final filePath = '${directory.path}/$model';

    final fileExists = await File(filePath).exists();
    if (!fileExists) {
      final byteData = await rootBundle.load('assets/models/$model');
      final file = File(filePath);
      await file.writeAsBytes(byteData.buffer
          .asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
    }

    _modelPathController.text = filePath;
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Model Interaction'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            TextField(
              controller: _modelPathController,
              decoration: const InputDecoration(
                labelText: 'Model Path',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: _promptController,
              decoration: const InputDecoration(
                labelText: 'Prompt',
                border: OutlineInputBorder(),
              ),
              minLines: 5,
              maxLines: null,
            ),
            const SizedBox(height: 10),
            Expanded(
              child: TextField(
                  controller: _resultController,
                  decoration: const InputDecoration(
                    labelText: 'Result',
                    border: OutlineInputBorder(),
                  ),
                  maxLines: null,
                  expands: true,
                  textAlignVertical: TextAlignVertical.top),
            ),
            const SizedBox(height: 10),
            Text(isModelLoaded ? 'Model Loaded' : 'Model Not Loaded'),
            const SizedBox(height: 10),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton(
                  onPressed: () {
                    llamaProcessor = LlamaProcessor(_modelPathController.text);
                    setState(() {
                      isModelLoaded = true;
                    });
                  },
                  child: const Text('Load Model'),
                ),
                ElevatedButton(
                  onPressed: isModelLoaded
                      ? () {
                          llamaProcessor?.unloadModel();
                          setState(() {
                            isModelLoaded = false;
                          });
                        }
                      : null,
                  child: const Text('Unload Model'),
                ),
                ElevatedButton(
                  onPressed: isModelLoaded
                      ? () {
                          _streamSubscription?.cancel();
                          _resultController.text = "";
                          _streamSubscription =
                              llamaProcessor?.stream.listen((data) {
                            _resultController.text += data;
                          }, onError: (error) {
                            _resultController.text = "Error: $error";
                          }, onDone: () {});
                          llamaProcessor?.prompt(_promptController.text);
                        }
                      : null,
                  child: const Text('Generate Answer'),
                ),
                ElevatedButton(
                  onPressed: isModelLoaded
                      ? () {
                          llamaProcessor?.stop();
                        }
                      : null,
                  child: const Text('Stop Generation'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _streamSubscription?.cancel();
    _modelPathController.dispose();
    _promptController.dispose();
    _resultController.dispose();
    llamaProcessor?.unloadModel();
    super.dispose();
  }
}
```

## Documentation

For more detailed information about the classes and their functionalities, please refer to the following documentation:

- [ContextParams](doc/context_params.md) - Configuration settings for the Llama model.
- [Llama](doc/llama.md) - Interface for interacting with the Llama model.
- [LlamaProcessor](doc/llama_processor.md) - Handles asynchronous operation of a Llama model in a separate isolate.
- [LlamaSplitMode](doc/llama_split_mode.md) - Enumerates modes for splitting the Llama model across multiple GPUs.
- [ModelParams](doc/model_params.md) - Configuration settings for how the model is split and operated across multiple GPUs.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.
