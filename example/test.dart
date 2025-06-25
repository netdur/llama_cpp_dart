// ignore_for_file: avoid_print

import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

String prompt = """what is 2 * 4?""";

void main() async {
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/gemma-3-12b-it-q4_0.gguf";

    ChatHistory history = ChatHistory()
      ..addMessage(role: Role.user, content: prompt)
      ..addMessage(role: Role.assistant, content: "");

    final modelParams = ModelParams()..nGpuLayers = 99;

    final contextParams = ContextParams()
      ..nPredict = -1
      ..nCtx = 8192
      ..nBatch = 8192;

    final samplerParams = SamplerParams()
      ..temp = 0.7
      ..topK = 64
      ..topP = 0.95
      ..penaltyRepeat = 1.1;

    Llama llama =
        Llama(modelPath, modelParams, contextParams, samplerParams, false);

    llama.setPrompt(
        history.exportFormat(ChatFormat.gemini, leaveLastAssistantOpen: true));
    while (true) {
      var (token, done) = llama.getNext();
      stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");

    llama.dispose();
  } catch (e) {
    print("\nError: ${e.toString()}");
  }
}
