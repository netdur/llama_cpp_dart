import 'package:flutter/material.dart';
import 'llama_processor.dart';

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
      darkTheme: ThemeData(
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

  bool flagForStop = false;

  @override
  void initState() {
    _modelPathController.text = "mistral-7b-openorca.Q5_K_M.gguf";
    _promptController.text = "The current USA president is...";
    super.initState();
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
              ),
            ),
            const SizedBox(height: 10),
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
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _modelPathController.dispose();
    _promptController.dispose();
    _resultController.dispose();
    llamaProcessor?.unloadModel();
    super.dispose();
  }
}
