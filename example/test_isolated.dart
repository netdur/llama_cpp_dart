// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";

  ContextParams contextParams = ContextParams();
  contextParams.nPredict = -1;
  contextParams.nCtx = 8192;

  final samplerParams = SamplerParams();
  samplerParams.temp = 0.7;
  samplerParams.topK = 64;
  samplerParams.topP = 0.95;
  samplerParams.penaltyRepeat = 1.1;

  String modelPath = "/Users/adel/Workspace/gguf/gemma-3-12b-it-q4_0.gguf";
  final loadCommand = LlamaLoad(
      path: modelPath,
      modelParams: ModelParams(),
      contextParams: contextParams,
      samplingParams: samplerParams);

  final llamaParent = LlamaParent(loadCommand);

  try {
    await llamaParent.init();

    int i = 0;
    List<String> prompts = [
      getPrompt("What is 2 * 4?"),
      getPrompt("What is 4 * 4?"),
      getPrompt("hey what is your name?")
    ];

    llamaParent.stream.listen((response) => stdout.write(response),
        onError: (e) => stderr.writeln("Stream error: $e"));

    llamaParent.completions.listen((event) {
      if (!event.success) {
        stderr.writeln("Completion error: ${event.errorDetails}");
        return;
      }

      i++;
      if (i >= prompts.length) {
        llamaParent.dispose();
      } else {
        print("\n----- Next prompt -----\n");
        llamaParent.sendPrompt(prompts[i]);
      }
    }, onError: (e) => stderr.writeln("Completion error: $e"));

    llamaParent.sendPrompt(prompts[0]);
  } catch (e) {
    stderr.writeln("Error: $e");
    await llamaParent.dispose();
    exit(1);
  }
}

String getPrompt(String content) {
  ChatHistory history = ChatHistory()
    ..addMessage(role: Role.user, content: content)
    ..addMessage(role: Role.assistant, content: "");
  return history.exportFormat(ChatFormat.gemini, leaveLastAssistantOpen: true);
}
