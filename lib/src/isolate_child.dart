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
          :final verbose,
          :final mmprojPath
        ):
        _handleLoad(path, modelParams, contextParams, samplingParams, verbose,
            mmprojPath);

      case LlamaPrompt(:final prompt, :final promptId, :final images, :final slotId):
        _handlePrompt(prompt, promptId, images, slotId);

      case LlamaInit(:final libraryPath):
        _handleInit(libraryPath);

      case LlamaEmbedd(:final prompt):
        _handleEmbedding(prompt);

      // FIX: Handle the Dispose command to satisfy exhaustiveness
      case LlamaDispose():
        _handleDispose();
    }
  }

  void _handleDispose() {
    shouldStop = true;
    if (llama != null) {
      llama!.dispose();
      llama = null;
    }
    // The isolate usually dies shortly after this, but explicit disposal 
    // ensures C pointers are freed immediately.
  }

  /// Handle stop command
  void _handleStop() {
    shouldStop = true;
    // We don't send confirmation here immediately;
    // the generation loop will detect 'shouldStop', break, and send the final status.
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
      bool verbose,
      String? mmprojPath) {
    try {
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

  /// Handle prompt command
  void _handlePrompt(String prompt, String promptId, List<LlamaImage>? images, String? slotId) {
    shouldStop = false;
    _sendPrompt(prompt, promptId, images, slotId);
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

  /// Handle init command
  void _handleInit(String? libraryPath) {
    try {
      Llama.libraryPath = libraryPath;
      // Force load to verify path
      final _ = Llama.lib;
      sendToParent(LlamaResponse.confirmation(LlamaStatus.uninitialized));
    } catch (e) {
      sendToParent(LlamaResponse.error("Failed to open library: $e"));
    }
  }

  /// Process a prompt and send responses
  Future<void> _sendPrompt(String prompt, String promptId, List<LlamaImage>? images, String? slotId) async {
    if (llama == null) {
      sendToParent(LlamaResponse.error("Model not initialized", promptId));
      return;
    }

    try {
      // --- Slot Management ---
      if (slotId != null) {
        try {
          // Ensure slot exists and switch to it
          llama!.createSlot(slotId);
          llama!.setSlot(slotId);
        } catch (e) {
          sendToParent(LlamaResponse.error("Slot allocation failed: $e", promptId));
          return;
        }
      } else {
        // Fallback to default
        llama!.setSlot("default");
      }
      // -----------------------

      sendToParent(LlamaResponse(
          text: "",
          isDone: false,
          status: LlamaStatus.generating,
          promptId: promptId));

      Stream<String> tokenStream;

      if (images != null && images.isNotEmpty) {
        tokenStream = llama!.generateWithMedia(prompt, inputs: images);
      } else {
        llama!.setPrompt(prompt);
        tokenStream = llama!.generateText();
      }

      await for (final token in tokenStream) {
        if (shouldStop) break;
        sendToParent(LlamaResponse(
            text: token,
            isDone: false,
            status: LlamaStatus.generating,
            promptId: promptId));
      }

      sendToParent(LlamaResponse(
          text: "",
          isDone: true,
          status: LlamaStatus.ready,
          promptId: promptId));
    } catch (e) {
      sendToParent(
          LlamaResponse.error("Generation error: ${e.toString()}", promptId));
    }
  }
}