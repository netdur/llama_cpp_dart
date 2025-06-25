// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  try {
    ContextParams contextParams = ContextParams();
    contextParams.nPredict = 512 * 12;
    contextParams.nCtx = 512 * 12;
    contextParams.nBatch = 512 * 12;

    final samplerParams = SamplerParams();
    samplerParams.temp = 0.6;
    samplerParams.minP = 0;
    samplerParams.topK = 20;
    samplerParams.topP = 0.95;
    // samplerParams.penaltyRepeat = 1.1;

    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath =
        "/Users/adel/Workspace/gguf/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf";
    Llama llama =
        Llama(modelPath, ModelParams(), contextParams, samplerParams, true);

    ChatHistory history = ChatHistory();
    history.addMessage(role: Role.user, content: "what is 2 * 2?");
    history.addMessage(role: Role.assistant, content: "");

    llama.setPrompt(history.exportFormat(ChatFormat.chatml));
    while (true) {
      var (token, done) = llama.getNext();
      stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");

    llama.dispose();
  } catch (e) {
    print("Error: ${e.toString()}");
  }
}
