import "package:llama_cpp_dart/llama_cpp_dart.dart";

sealed class LlamaCommand {}

class LlamaStop extends LlamaCommand {}

class LlamaClear extends LlamaCommand {}

class LlamaInit extends LlamaCommand {
  final String? libraryPath;
  final ModelParams modelParams;
  final ContextParams contextParams;
  final SamplerParams samplingParams;
  final PromptFormat? format;

  LlamaInit(this.libraryPath, this.modelParams, this.contextParams,
      this.samplingParams,
      [this.format]);
}

class LlamaPrompt extends LlamaCommand {
  final String prompt;
  LlamaPrompt(this.prompt);
}

class LlamaResponse {
  final String text;
  final bool isDone;
  final LlamaStatus? status;

  LlamaResponse({
    required this.text,
    required this.isDone,
    this.status,
  });
}

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
