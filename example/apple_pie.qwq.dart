// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  try {
    ContextParams contextParams = ContextParams();
    contextParams.nPredict = 512;
    contextParams.nCtx = 512;
    contextParams.nBatch = 512;

    final samplerParams = SamplerParams();
    samplerParams.temp = 0.6;
    samplerParams.minP = 0;
    samplerParams.topK = 20;
    samplerParams.topP = 0.95;
    // samplerParams.penaltyRepeat = 1.1;

    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/qwq-32b-q4_k_m.gguf";
    Llama llama =
        Llama(modelPath, ModelParams(), contextParams, samplerParams, true);

    llama.setPrompt("apple pie recipe?");
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
