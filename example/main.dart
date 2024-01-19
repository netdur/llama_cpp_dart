import 'dart:async';

import 'package:flutter/material.dart';

import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:path_provider/path_provider.dart';

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

  // ignore: unused_element
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
                  child: const Text('Run Prompt'),
                ),
                ElevatedButton(
                  onPressed: isModelLoaded
                      ? () {
                          llamaProcessor?.stop();
                        }
                      : null,
                  child: const Text('Stop Prompt'),
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
