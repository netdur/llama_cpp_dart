// ignore_for_file: avoid_print

import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/qwq-32b-q4_k_m.gguf";

    ChatHistory history = ChatHistory()
      ..addMessage(role: Role.user, content: "2 * 2 =");
    String prompt =
        history.exportFormat(ChatFormat.chatml, leaveLastAssistantOpen: true);
    prompt += "<|im_start|>assistant";
    print(prompt);

    final modelParams = ModelParams();

    final contextParams = ContextParams()
      ..nPredict = -1
      ..nCtx = 512;

    final samplerParams = SamplerParams()
      ..temp = 1.0
      ..topK = 64
      ..topP = 0.95
      ..penaltyRepeat = 1.1;

    Llama llama =
        Llama(modelPath, modelParams, contextParams, samplerParams, false);

    llama.setPrompt(prompt);
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
