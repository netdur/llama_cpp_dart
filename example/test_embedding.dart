// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Downloads/gemma-3-12b-it-Q4_K_M.gguf";

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
