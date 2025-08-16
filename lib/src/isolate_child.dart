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
          :final samplingParams,
          :final mmprojPath
        ):
        _handleLoad(
            path, modelParams, contextParams, samplingParams, mmprojPath);

      case LlamaPrompt(:final prompt, :final promptId, :final images):
        _handlePrompt(prompt, promptId, images);

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
  void _handleLoad(
      String path,
      ModelParams modelParams,
      ContextParams contextParams,
      SamplerParams samplingParams,
      String? mmprojPath) {
    try {
      // Create Llama instance with optional mmproj path for VLM support
      if (mmprojPath != null) {
        llama = Llama(path, modelParams, contextParams, samplingParams, false,
            mmprojPath);
      } else {
        llama = Llama(path, modelParams, contextParams, samplingParams);
      }
      sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
    } catch (e) {
      sendToParent(LlamaResponse.error("Error loading model: $e"));
    }
  }

  /// Handle prompt command
  void _handlePrompt(String prompt, String promptId, List<LlamaImage>? images) {
    shouldStop = false;
    _sendPrompt(prompt, promptId, images);
  }

  /// Handle init command
  void _handleInit(String? libraryPath) {
    Llama.libraryPath = libraryPath;
    sendToParent(LlamaResponse.confirmation(LlamaStatus.uninitialized));
  }

  /// Process a prompt and send responses
  void _sendPrompt(
      String prompt, String promptId, List<LlamaImage>? images) async {
    if (llama == null) {
      sendToParent(LlamaResponse.error(
          "Cannot generate: model not initialized", promptId));
      return;
    }

    try {
      // Send confirmation that generation has started
      sendToParent(LlamaResponse(
          text: "",
          isDone: false,
          status: LlamaStatus.generating,
          promptId: promptId));

      // Use different generation method based on whether images are provided
      if (images != null && images.isNotEmpty) {
        final stream = llama!.generateWithMeda(prompt, inputs: images);

        await for (final token in stream) {
          if (shouldStop) break;

          sendToParent(LlamaResponse(
              text: token,
              isDone: false,
              status: LlamaStatus.generating,
              promptId: promptId));
        }

        // Send completion
        sendToParent(LlamaResponse(
            text: "",
            isDone: true,
            status: LlamaStatus.ready,
            promptId: promptId));
      } else {
        // Use regular text generation
        llama!.setPrompt(prompt);

        bool generationDone = false;
        while (!generationDone && !shouldStop) {
          final (text, isDone) = llama!.getNext();

          sendToParent(LlamaResponse(
              text: text,
              isDone: isDone,
              status: isDone ? LlamaStatus.ready : LlamaStatus.generating,
              promptId: promptId));

          generationDone = isDone;

          if (!isDone) {
            await Future.delayed(Duration(milliseconds: 5));
          }
        }
      }

      // If stopped by external request, send completion confirmation
      if (shouldStop) {
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