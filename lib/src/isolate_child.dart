import 'dart:async';
import 'package:typed_isolate/typed_isolate.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

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
          :final verbose,
          :final mmprojPath
        ):
        _handleLoad(path, modelParams, contextParams, samplingParams, verbose, mmprojPath);
      case LlamaPrompt(:final prompt, :final promptId, :final images):
        _handlePrompt(prompt, promptId, images);
      case LlamaInit(:final libraryPath):
        _handleInit(libraryPath);
      case LlamaEmbedd(:final prompt):
        _handleEmbedding(prompt);
      case LlamaDispose():
        _handleDispose();
    }
  }

  void _handleStop() {
    shouldStop = true;
    // We don't send confirmation here immediately; 
    // the generation loop will detect 'shouldStop', break, and send the final status.
  }

  void _handleClear() {
    shouldStop = true; // Stop any ongoing generation first
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

  void _handleLoad(
      String path,
      ModelParams modelParams,
      ContextParams contextParams,
      SamplerParams samplingParams,
      bool verbose,
      String? mmprojPath) {
    try {
      // FIX: Use Named Parameters matching the new Llama class
      llama = Llama(
        path,
        modelParams: modelParams,
        contextParams: contextParams,
        samplerParams: samplingParams,
        verbose: verbose,
        mmprojPath: mmprojPath,
      );
      sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
    } catch (e) {
      sendToParent(LlamaResponse.error("Error loading model: $e"));
    }
  }

  void _handlePrompt(String prompt, String promptId, List<LlamaImage>? images) {
    // Reset stop flag for new prompt
    shouldStop = false; 
    _sendPrompt(prompt, promptId, images);
  }

  void _handleEmbedding(String prompt) {
    shouldStop = false;
    if (llama == null) {
      sendToParent(LlamaResponse.error("Model not initialized"));
      return;
    }
    try {
      final embeddings = llama!.getEmbeddings(prompt);
      sendToParent(LlamaResponse(
        text: "",
        isDone: true,
        embeddings: embeddings,
        status: LlamaStatus.ready,
      ));
    } catch (e) {
      sendToParent(LlamaResponse.error("Embedding error: $e"));
    }
  }

  void _handleInit(String? libraryPath) {
    try {
      Llama.libraryPath = libraryPath;
      final _ = Llama.lib; 
      sendToParent(LlamaResponse.confirmation(LlamaStatus.uninitialized));
    } catch (e) {
      sendToParent(LlamaResponse.error("Failed to open library at $libraryPath: $e"));
    }
  }

  void _handleDispose() {
    shouldStop = true;
    if (llama != null) {
      try {
        llama!.dispose();
        llama = null;
        sendToParent(LlamaResponse.confirmation(LlamaStatus.disposed));
      } catch (e) {
        sendToParent(LlamaResponse.error("Error disposing model: $e"));
      }
    } else {
      sendToParent(LlamaResponse.confirmation(LlamaStatus.disposed));
    }
  }

  Future<void> _sendPrompt(String prompt, String promptId, List<LlamaImage>? images) async {
    if (llama == null) {
      sendToParent(LlamaResponse.error("Cannot generate: model not initialized", promptId));
      return;
    }

    try {
      // 1. Send "Generating" status
      sendToParent(LlamaResponse(
          text: "",
          isDone: false,
          status: LlamaStatus.generating,
          promptId: promptId));

      Stream<String> tokenStream;

      // 2. Select Stream source
      if (images != null && images.isNotEmpty) {
        tokenStream = llama!.generateWithMedia(prompt, inputs: images);
      } else {
        // For text-only, we must set the prompt first
        llama!.setPrompt(prompt);
        tokenStream = llama!.generateText();
      }

      // 3. Consume Stream
      await for (final token in tokenStream) {
        if (shouldStop) {
          break; // Break the loop, finally block will handle cleanup
        }
        
        sendToParent(LlamaResponse(
            text: token,
            isDone: false, // Still going
            status: LlamaStatus.generating,
            promptId: promptId));
      }

      // 4. Send Completion
      sendToParent(LlamaResponse(
          text: "", // No text in final message
          isDone: true,
          status: LlamaStatus.ready, // Back to ready
          promptId: promptId));

    } catch (e) {
      sendToParent(LlamaResponse.error("Generation error: ${e.toString()}", promptId));
    }
  }
}
