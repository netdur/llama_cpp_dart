import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// Base class for commands sent to the LlamaChild isolate
sealed class LlamaCommand {}

/// Command to stop ongoing generation
class LlamaStop extends LlamaCommand {}

/// Command to clear the model context
class LlamaClear extends LlamaCommand {}

/// Command to initialize the Llama library
class LlamaInit extends LlamaCommand {
  final String? libraryPath;
  final ModelParams modelParams;
  final ContextParams contextParams;
  final SamplerParams samplingParams;

  LlamaInit(this.libraryPath, this.modelParams, this.contextParams,
      this.samplingParams);
}

/// Command to send a prompt for generation
class LlamaPrompt extends LlamaCommand {
  final String prompt;
  final String promptId;

  LlamaPrompt(this.prompt, this.promptId);
}

/// Response from the LlamaChild isolate
class LlamaResponse {
  final String text;
  final bool isDone;
  final LlamaStatus? status;
  final String? promptId;
  final String? errorDetails;
  final bool isConfirmation;

  LlamaResponse({
    required this.text,
    required this.isDone,
    this.status,
    this.promptId,
    this.errorDetails,
    this.isConfirmation = false,
  });

  /// Create a confirmation response
  factory LlamaResponse.confirmation(LlamaStatus status, [String? promptId]) {
    return LlamaResponse(
      text: "",
      isDone: false,
      status: status,
      promptId: promptId,
      isConfirmation: true,
    );
  }

  /// Create an error response
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

/// Command to load a model
class LlamaLoad extends LlamaCommand {
  final String path;
  final ModelParams modelParams;
  final ContextParams contextParams;
  final SamplerParams samplingParams;
  final PromptFormat? format;

  LlamaLoad({
    required this.path,
    required this.modelParams,
    required this.contextParams,
    required this.samplingParams,
    this.format,
  });
}
