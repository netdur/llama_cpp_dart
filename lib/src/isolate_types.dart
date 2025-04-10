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

class LlamaGetEmbeddings extends LlamaCommand {
  final String prompt;
  final bool addBos;
  final bool normalize;
  LlamaGetEmbeddings(this.prompt, {this.addBos = true, this.normalize = true});
}

class LlamaResponse {
  final String text;
  final bool isDone;
  final LlamaStatus? status;
  final List<double>? embeddings;
  final String? error;

  LlamaResponse(
      {this.text = "",
      required this.isDone,
      this.status,
      this.embeddings,
      this.error});
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
