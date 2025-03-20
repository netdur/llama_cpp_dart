import 'package:typed_isolate/typed_isolate.dart';
import "llama.dart";
import "isolate_types.dart";

class LlamaChild extends IsolateChild<LlamaResponse, LlamaCommand> {
  LlamaChild() : super(id: 1);

  bool shouldStop = false;
  Llama? llama;

  @override
  void onData(LlamaCommand data) {
    switch (data) {
      case LlamaStop():
        shouldStop = true;

      case LlamaClear():
        shouldStop = true;
        llama?.clear();

      case LlamaLoad(
          :final path,
          :final modelParams,
          :final contextParams,
          :final samplingParams
        ):
        llama = Llama(path, modelParams, contextParams, samplingParams);

      case LlamaPrompt(:final prompt):
        shouldStop = false; // Reset stop flag before new prompt
        _sendPrompt(prompt);

      case LlamaInit(:final libraryPath):
        Llama.libraryPath = libraryPath;
    }
  }

  void _sendPrompt(String prompt) {
    llama?.setPrompt(prompt);

    while (true) {
      if (shouldStop) {
        break;
      }

      final (text, isDone) = llama!.getNext();

      sendToParent(LlamaResponse(text: text, isDone: isDone));

      if (isDone) {
        shouldStop = true;
      }
    }
  }
}
