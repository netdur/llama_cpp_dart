import 'dart:async';
import 'dart:isolate';

import 'model_manager.dart';

class LlamaProcessor {
  final String path;
  late Isolate _modelIsolate;
  late SendPort _modelSendPort;
  final ReceivePort _receivePort = ReceivePort();
  final StreamController<String> _controller = StreamController<String>();

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

    var modelManager = ModelManager();

    isolateReceivePort.listen((message) {
      if (message is Map) {
        switch (message['command']) {
          case 'load':
            modelManager.loadModel(message['path']);
            break;
          case 'setup':
            modelManager.createContext();
            modelManager.setup(message['prompt']);
            break;
          case 'getNext':
            String result = modelManager.getNext();
            mainSendPort.send(result);
            break;
          case 'clear':
            modelManager.clearContext();
            break;
          case 'unload':
            modelManager.unloadModel();
            break;
        }
      }
    });
  }

  void setup(String prompt) {
    _modelSendPort.send({'command': 'setup', 'prompt': prompt});
  }

  void getNext() {
    _modelSendPort.send({'command': 'getNext'});
  }

  void clear() {
    _modelSendPort.send({'command': 'clear'});
  }

  void unloadModel() {
    _modelSendPort.send({'command': 'unload'});
    _modelIsolate.kill(priority: Isolate.immediate);
    _receivePort.close();
    _controller.close();
  }
}
