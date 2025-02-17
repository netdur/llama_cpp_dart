// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath =
        "/Users/adel/Downloads/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf";

    ModelParams modelParams = ModelParams();
    ContextParams contextParams = ContextParams();
    contextParams.embeddings = true;
    Llama llama = Llama(modelPath, modelParams, contextParams);

    List<double> embeddings = llama.getEmbeddings("2 * 2 = ?");
    stdout.write(embeddings);
    stdout.write("\n");

    llama.dispose();
  } catch (e) {
    print("Error: ${e.toString()}");
  }
}
