import 'package:typed_isolate/typed_isolate.dart';

import "new_llama.dart";
import "isolate_types.dart";

class LlamaChild extends IsolateChild<LlamaResponse, LlamaCommand> {
  LlamaChild() : super(id: 1);

  bool shouldStop = false;
  NewLlama? llama;

  @override
  void onData(LlamaCommand data) {
    switch (data) {
      case LlamaStop() || LlamaClear():
        shouldStop = true;
        // TODO: .clear() was removed?
        // llama?.clear();
      case LlamaLoad(:final path):
        llama = NewLlama(path);
        // TODO: All these params were removed?
        // llama = NewLlama(path, modelParams, contextParams, samplingParams);
      case LlamaPrompt(:final prompt):
        _sendPrompt(prompt);
      case LlamaInit(:final libraryPath):
        NewLlama.libraryPath = libraryPath;
    }
  }

  void _sendPrompt(String prompt) {
    llama?.setPrompt(prompt);
    while (true) {
      if (shouldStop) break;
      final (text, isDone) = llama!.getNext();
      sendToParent(LlamaResponse(text: text, isDone: isDone));
      if (isDone) shouldStop = true;
    }
  }
}
