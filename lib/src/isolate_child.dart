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

      case LlamaPrompt(
          :final prompt,
          :final promptId,
          :final images,
          :final slotId
        ):
        _handlePrompt(prompt, promptId, images, slotId);

      case LlamaInit(:final libraryPath):
        _handleInit(libraryPath);

      case LlamaEmbedd(:final prompt):
        _handleEmbedding(prompt);

      case LlamaDispose():
        _handleDispose();

      // --- TIER 2 & 3 HANDLERS ---
      case LlamaSaveState(:final slotId):
        _handleSaveState(slotId);

      case LlamaLoadState(:final slotId, :final data):
        _handleLoadState(slotId, data);

      case LlamaLoadSession(:final slotId, :final path):
        _handleLoadSession(slotId, path);

      case LlamaFreeSlot(:final slotId):
        _handleFreeSlot(slotId);
    }
  }

  void _handleDispose() {
    shouldStop = true;
    if (llama != null) {
      llama!.dispose();
      llama = null;
    }
  }

  /// Handle stop command
  void _handleStop() {
    shouldStop = true;
    // Send confirmation so the Parent stops waiting!
    sendToParent(
        LlamaResponse.confirmation(llama?.status ?? LlamaStatus.ready));
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

  /// Handle embedding command
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

  /// Handle prompt command
  void _handlePrompt(
      String prompt, String promptId, List<LlamaImage>? images, String? slotId) {
    shouldStop = false;
    _sendPrompt(prompt, promptId, images, slotId);
  }

  // --- NEW TIER 2 & 3 IMPLEMENTATIONS ---

  void _handleSaveState(String slotId) {
    if (llama == null) return;
    try {
      // 1. Switch to the slot we want to save
      llama!.setSlot(slotId);

      // 2. Extract Data (Uint8List)
      final data = llama!.saveState();

      // 3. Send back to Parent
      sendToParent(LlamaResponse.stateData(data));
    } catch (e) {
      sendToParent(
          LlamaResponse.error("Failed to save state for $slotId: $e"));
    }
  }

  void _handleLoadState(String slotId, dynamic data) {
    if (llama == null) return;
    try {
      // 1. Create/Set slot
      try {
        llama!.createSlot(slotId);
      } catch (_) {
        // Ignore if already exists, logic continues
      }
      llama!.setSlot(slotId);

      // 2. Load
      llama!.loadState(data);

      sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
    } catch (e) {
      sendToParent(
          LlamaResponse.error("Failed to load state for $slotId: $e"));
    }
  }

  void _handleLoadSession(String slotId, String path) {
    if (llama == null) return;
    try {
      try {
        llama!.createSlot(slotId);
      } catch (_) {
        // Ignore if already exists
      }
      llama!.setSlot(slotId);

      final success = llama!.loadSession(path);
      if (success) {
        sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
      } else {
        sendToParent(LlamaResponse.error("Session file not found or invalid"));
      }
    } catch (e) {
      sendToParent(LlamaResponse.error("Failed to load session $path: $e"));
    }
  }

  void _handleFreeSlot(String slotId) {
    if (llama == null) return;
    try {
      llama!.freeSlot(slotId);
      sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
    } catch (e) {
      // Log but don't crash, maybe it was already freed
      // ignore: avoid_print
      print("Warning freeing slot $slotId: $e");
      sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
    }
  }

  // --- GENERATION LOOP ---

  Future<void> _sendPrompt(String prompt, String promptId,
      List<LlamaImage>? images, String? slotId) async {
    if (llama == null) {
      sendToParent(LlamaResponse.error("Model not initialized", promptId));
      return;
    }

    try {
      // --- Slot Management ---
      if (slotId != null) {
        try {
          llama!.createSlot(slotId);
          llama!.setSlot(slotId);
        } catch (e) {
          sendToParent(
              LlamaResponse.error("Slot allocation failed: $e", promptId));
          return;
        }
      } else {
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