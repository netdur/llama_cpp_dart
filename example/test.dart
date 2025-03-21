// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Downloads/gemma-3-12b-it-Q4_K_M.gguf";
    Llama llama = Llama(modelPath);

    llama.setPrompt(
        "<start_of_turn>What is 2 * 4?<end_of_turn>\n<start_of_turn>model\n");
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
