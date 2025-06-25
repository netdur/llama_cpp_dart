// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  try {
    ContextParams contextParams = ContextParams();
    contextParams.nPredict = 8192;
    contextParams.nCtx = 8192;
    contextParams.nBatch = 512;

    final samplerParams = SamplerParams();
    samplerParams.temp = 1.0;
    samplerParams.topK = 64;
    samplerParams.topP = 0.95;
    samplerParams.penaltyRepeat = 1.1;

    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/gemma-3-12b-it-q4_0.gguf";
    Llama llama = Llama(modelPath, ModelParams(), contextParams, samplerParams);

    llama.setPrompt(
        "<start_of_turn>user\napple pie recipe?<end_of_turn>\n<start_of_turn>model\n");
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
