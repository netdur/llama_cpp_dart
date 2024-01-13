import 'dart:async';
import 'dart:isolate';

import 'package:llama_cpp_dart/gpt_params.dart';
import 'package:llama_cpp_dart/llm.dart';

class LlamaProcessor {
  final String path;
  late Isolate _modelIsolate;
  late SendPort _modelSendPort;
  final ReceivePort _receivePort = ReceivePort();
  final StreamController<String> _controller =
      StreamController<String>.broadcast();

  LlamaProcessor(this.path) {
    _loadModelIsolate();
  }

  Stream<String> get stream => _controller.stream;

  Future<void> _loadModelIsolate() async {
    _modelIsolate = await Isolate.spawn(
      _modelIsolateEntryPoint,
      _receivePort.sendPort,
    );

    _receivePort.listen((message) {
      if (message is SendPort) {
        _modelSendPort = message;
        _modelSendPort.send({'command': 'load', 'path': path});
      } else if (message is String) {
        _controller.add(message);
      }
    });
  }

  static void _modelIsolateEntryPoint(SendPort mainSendPort) {
    ReceivePort isolateReceivePort = ReceivePort();
    mainSendPort.send(isolateReceivePort.sendPort);

    LLM llm = LLM();
    bool flagForStop = false;

    isolateReceivePort.listen((message) async {
      if (message is Map) {
        switch (message['command']) {
          case 'load':
            GptParams gptParams = GptParams();
            gptParams.model = message['path'];
            llm.loadModel(gptParams);
            break;
          case 'prompt':
            llm.setTextIter(message['prompt']);
            bool isRunning = true;
            while (isRunning) {
              if (flagForStop) {
                flagForStop = false;
                break;
              }
              var (text, hasNext, _) = llm.getNext();
              if (!hasNext) break;
              mainSendPort.send(text);
              await Future.delayed(Duration.zero);
            }
            break;
          case 'stop':
            flagForStop = true;
            break;
          case 'unload':
            llm.unloadModel();
            break;
        }
      }
    });
  }

  void prompt(String prompt) {
    _modelSendPort.send({'command': 'prompt', 'prompt': prompt});
  }

  void stop() {
    _modelSendPort.send({'command': 'stop'});
  }

  void unloadModel() {
    _modelSendPort.send({'command': 'unload'});
    _modelIsolate.kill(priority: Isolate.immediate);
    _receivePort.close();
    _controller.close();
  }
}
