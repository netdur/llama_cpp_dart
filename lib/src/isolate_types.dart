import "package:llama_cpp_dart/llama_cpp_dart.dart";

sealed class LlamaCommand {}

class LlamaStop extends LlamaCommand {}

class LlamaClear extends LlamaCommand {}

class LlamaInit extends LlamaCommand {
  final String? libraryPath;
  LlamaInit(this.libraryPath);
}

class LlamaPrompt extends LlamaCommand {
  final String prompt;
  LlamaPrompt(this.prompt);
}

class LlamaResponse {
  final String text;
  final bool isDone;
  LlamaResponse({
    required this.text,
    required this.isDone,
  });
}

class LlamaLoad extends LlamaCommand {
  final String path;
  final ModelParams modelParams;
  final ContextParams contextParams;
  final SamplerParams samplingParams;
  final PromptFormat format;
  LlamaLoad({
    required this.path,
    required this.modelParams,
    required this.contextParams,
    required this.samplingParams,
    required this.format,
  });
}
