import 'dart:typed_data'; // Needed for Uint8List
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// Base class for commands sent to the LlamaChild isolate
sealed class LlamaCommand {}

class LlamaStop extends LlamaCommand {}

class LlamaClear extends LlamaCommand {}

class LlamaDispose extends LlamaCommand {}

// --- NEW COMMANDS FOR TIER 2 & 3 ---

class LlamaSaveState extends LlamaCommand {
  final String slotId;
  LlamaSaveState(this.slotId);
}

class LlamaLoadState extends LlamaCommand {
  final String slotId;
  final Uint8List data;
  LlamaLoadState(this.slotId, this.data);
}

class LlamaLoadSession extends LlamaCommand {
  final String slotId;
  final String path;
  LlamaLoadSession(this.slotId, this.path);
}

class LlamaFreeSlot extends LlamaCommand {
  final String slotId;
  LlamaFreeSlot(this.slotId);
}

// -----------------------------------

class LlamaEmbedd extends LlamaCommand {
  final String prompt;
  LlamaEmbedd(this.prompt);
}

class LlamaInit extends LlamaCommand {
  final String? libraryPath;
  LlamaInit(this.libraryPath);
}

class LlamaPrompt extends LlamaCommand {
  final String prompt;
  final String promptId;
  final List<LlamaImage>? images;
  final String? slotId;

  LlamaPrompt(this.prompt, this.promptId, {this.images, this.slotId});
}

class LlamaLoad extends LlamaCommand {
  final String path;
  final ModelParams modelParams;
  final ContextParams contextParams;
  final SamplerParams samplingParams;
  final bool verbose;
  final String? mmprojPath;

  LlamaLoad({
    required this.path,
    required this.modelParams,
    required this.contextParams,
    required this.samplingParams,
    this.verbose = false,
    this.mmprojPath,
  });
}

class LlamaResponse {
  final String text;
  final bool isDone;
  final LlamaStatus? status;
  final String? promptId;
  final String? errorDetails;
  final bool isConfirmation;
  final List<double>? embeddings;
  
  // NEW: Carries the binary state data from Child -> Parent
  final Uint8List? stateData; 

  LlamaResponse({
    required this.text,
    required this.isDone,
    this.status,
    this.promptId,
    this.errorDetails,
    this.isConfirmation = false,
    this.embeddings,
    this.stateData,
  });

  factory LlamaResponse.confirmation(LlamaStatus status, [String? promptId]) {
    return LlamaResponse(
      text: "",
      isDone: false,
      status: status,
      promptId: promptId,
      isConfirmation: true,
    );
  }

  // NEW: Response specifically for saveState
  factory LlamaResponse.stateData(Uint8List data) {
    return LlamaResponse(
      text: "",
      isDone: true, // Treated as a completed operation
      stateData: data,
    );
  }

  factory LlamaResponse.error(String errorMessage, [String? promptId]) {
    return LlamaResponse(
      text: "",
      isDone: true,
      status: LlamaStatus.error,
      promptId: promptId,
      errorDetails: errorMessage,
    );
  }
}