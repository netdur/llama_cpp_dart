import 'dart:async';
import 'dart:isolate';

import 'alpaca_format.dart';
import 'chatml_format.dart';
import 'prompt_format.dart';
import 'model_params.dart';
import 'context_params.dart';
import 'sampling_params.dart';
import 'llama.dart';

/// The `LlamaProcessor` class handles the asynchronous operation of a Llama model in a separate isolate.
///
/// It manages the communication between the main thread and the isolate, providing a stream of generated text.
class LlamaProcessor {
  /// The path to the model file.
  final String path;

  /// context parameters
  final ContextParams contextParams;

  /// model parameters
  final ModelParams modelParams;

  /// sampling parameters
  final SamplingParams samplingParams;

  /// The isolate where the Llama model is loaded and run.
  late Isolate _modelIsolate;

  /// SendPort for sending messages to the model isolate.
  late SendPort _modelSendPort;

  /// Completer for the initialization of _modelSendPort.
  Completer _uninitialized = Completer();

  /// ReceivePort for receiving messages from the model isolate.
  final ReceivePort _receivePort = ReceivePort();

  /// StreamController for broadcasting the generated text.
  final StreamController<String> _controller =
      StreamController<String>.broadcast();

  /// Conversation messages.
  List<Map<String, dynamic>> messages = [];

  /// Constructor for LlamaProcessor.
  ///
  /// Initializes the processor and starts the model isolate.
  LlamaProcessor(this.path, this.modelParams, this.contextParams, this.samplingParams) {
    _loadModelIsolate();
  }

  /// Stream of generated text.
  ///
  /// Consumers can listen to this stream to receive text generated by the Llama model.
  Stream<String> get stream => _controller.stream;

  /// Asynchronously loads the Llama model in a separate isolate.
  ///
  /// Sets up communication channels between the main thread and the isolate.
  Future<void> _loadModelIsolate() async {
    _modelIsolate = await Isolate.spawn(
      _modelIsolateEntryPoint,
      {'port': _receivePort.sendPort, 'libraryPath': Llama.libraryPath},
    );

    _receivePort.listen((message) {
      if (message is SendPort) {
        _modelSendPort = message;
        _modelSendPort.send({
          'command': 'load',
          'path': path,
          'modelParams': modelParams.toJson(),
          'contextParams': contextParams.toJson(),
          'samplingParams': samplingParams.toJson(),
        });
        _uninitialized.complete();
      } else if (message is String) {
        _parseResponse(message);
      }
    });
  }

  /// Entry point for the model isolate.
  ///
  /// Handles commands sent to the isolate, such as loading the model, generating text, and stopping the operation.
  static void _modelIsolateEntryPoint(Map<String, dynamic> args) {
    SendPort mainSendPort = args['port'] as SendPort;
    ReceivePort isolateReceivePort = ReceivePort();
    mainSendPort.send(isolateReceivePort.sendPort);

    Llama.libraryPath = args['libraryPath'] as String?;

    Llama? llama;
    Completer stopCompleter = Completer();

    isolateReceivePort.listen((message) async {
      if (message is Map) {
        switch (message['command']) {
          case 'load':
            ContextParams contextParams =
                ContextParams.fromJson(message['contextParams']);
            ModelParams modelParams =
                ModelParams.fromJson(message['modelParams']);
            SamplingParams samplingParams =
                SamplingParams.fromJson(message['samplingParams']);
            llama = Llama(message['path'], modelParams, contextParams, samplingParams);
            break;
          case 'prompt':
            llama?.setPrompt(message['prompt']);
            while (true) {
              if (stopCompleter.isCompleted) break;

              var (text, done) = llama!.getNext();
              mainSendPort.send(text);

              if (done) stopCompleter.complete();
            }
            break;
          case 'stop':
            if (!stopCompleter.isCompleted) stopCompleter.complete();
            llama?.clear();
            break;
        }
      }
    });
  }

  void _parseResponse(String response) {
    switch (modelParams.format) {
      case PromptFormatType.raw:
        _controller.add(response);
        break;
      case PromptFormatType.alpaca:
        String? chunk = AlpacaFormat().filterResponse(response);
        if (chunk != null) _controller.add(chunk);
        break;
      case PromptFormatType.chatml:
        String? chunk = ChatMLFormat().filterResponse(response);
        if (chunk != null) _controller.add(chunk);
        break;
      default:
        _controller.add(response);
        break;
    }
  }

  /// Sends a prompt to the model isolate for text generation.
  ///
  /// The generated text will be sent back to the main thread and emitted through the stream.
  void prompt(String prompt) {
    _uninitialized.future.then((_) {
      var formattedPrompt = prompt;

      switch (modelParams.format) {
        case PromptFormatType.raw:
          break;
        case PromptFormatType.alpaca:
          formattedPrompt = AlpacaFormat().formatPrompt(prompt);
          break;
        case PromptFormatType.chatml:
          formattedPrompt = ChatMLFormat().formatPrompt(prompt);
          break;
        default:
          break;
      }

      if (messages.isNotEmpty) {
        switch (modelParams.format) {
          case PromptFormatType.raw:
            break;
          case PromptFormatType.alpaca:
            final formattedMessages = AlpacaFormat().formatMessages(messages);
            formattedPrompt = '$formattedMessages$formattedPrompt';
            break;
          case PromptFormatType.chatml:
            final formattedMessages = ChatMLFormat().formatMessages(messages);
            formattedPrompt = '$formattedMessages$formattedPrompt';
            break;
          default:
            break;
        }
      }

      _modelSendPort.send({'command': 'prompt', 'prompt': formattedPrompt});
    });
  }

  /// Sends a stop command to the model isolate.
  ///
  /// This command interrupts the current text generation process.
  void stop() {
    _uninitialized.future.then((_) {
      _modelSendPort.send({'command': 'stop'});
    });
  }

  /// Unloads the model and terminates the isolate.
  ///
  /// Closes the communication ports and stream controller.
  void unloadModel() {
    _uninitialized.future.then((_) {
      _modelSendPort.send({'command': 'clear'});
      _modelIsolate.kill(priority: Isolate.immediate);
      _receivePort.close();
      _controller.close();
      _uninitialized = Completer();
    });
  }
}
