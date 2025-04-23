import 'dart:async';
import 'package:typed_isolate/typed_isolate.dart';
import '../llama_cpp_dart.dart';

/// Child isolate that handles Llama model operations
class LlamaChild extends IsolateChild<LlamaResponse, LlamaCommand> {
  LlamaChild() : super(id: 1);

  bool shouldStop = false;
  Llama? llama;

  @override
  void onData(LlamaCommand data) {
    switch (data) {
      case LlamaStop():
        _handleStop();

      case LlamaClear():
        _handleClear();

      case LlamaLoad(
          :final path,
          :final modelParams,
          :final contextParams,
          :final samplingParams
        ):
        _handleLoad(path, modelParams, contextParams, samplingParams);

      case LlamaPrompt(:final prompt, :final promptId):
        _handlePrompt(prompt, promptId);

      case LlamaInit(:final libraryPath):
        _handleInit(libraryPath);
    }
  }

  /// Handle stop command
  void _handleStop() {
    shouldStop = true;
    sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
  }

  /// Handle clear command
  void _handleClear() {
    shouldStop = true;
    if (llama != null) {
      try {
        llama!.clear();
        sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
      } catch (e) {
        sendToParent(LlamaResponse.error("Error clearing context: $e"));
      }
    } else {
      sendToParent(LlamaResponse.error("Cannot clear: model not initialized"));
    }
  }

  /// Handle load command
  void _handleLoad(String path, ModelParams modelParams,
      ContextParams contextParams, SamplerParams samplingParams) {
    try {
      llama = Llama(path, modelParams, contextParams, samplingParams);
      sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
    } catch (e) {
      sendToParent(LlamaResponse.error("Error loading model: $e"));
    }
  }

  /// Handle prompt command
  void _handlePrompt(String prompt, String promptId) {
    shouldStop = false;
    _sendPrompt(prompt, promptId);
  }

  /// Handle init command
  void _handleInit(String? libraryPath) {
    Llama.libraryPath = libraryPath;
    sendToParent(LlamaResponse.confirmation(LlamaStatus.uninitialized));
  }

  /// Process a prompt and send responses
  void _sendPrompt(String prompt, String promptId) async {
    if (llama == null) {
      sendToParent(LlamaResponse.error(
          "Cannot generate: model not initialized", promptId));
      return;
    }

    try {
      llama!.setPrompt(prompt);

      // Send confirmation that generation has started
      sendToParent(LlamaResponse(
          text: "",
          isDone: false,
          status: LlamaStatus.generating,
          promptId: promptId));

      bool generationDone = false;

      while (!generationDone && !shouldStop) {
        final (text, isDone) = llama!.getNext();

        sendToParent(LlamaResponse(
            text: text,
            isDone: isDone,
            status: isDone ? LlamaStatus.ready : LlamaStatus.generating,
            promptId: promptId));

        generationDone = isDone;

        await Future.delayed(Duration(milliseconds: isDone ? 0 : 5));
      }

      // If stopped by external request, send completion confirmation
      if (shouldStop && !generationDone) {
        sendToParent(LlamaResponse(
            text: "",
            isDone: true,
            status: LlamaStatus.ready,
            promptId: promptId));
      }
    } catch (e) {
      sendToParent(
          LlamaResponse.error("Generation error: ${e.toString()}", promptId));
    }
  }
}
