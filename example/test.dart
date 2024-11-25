// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
// import 'package:flutter/foundation.dart';
// import 'package:system_info2/system_info2.dart';

void main() async {
  try {
    // final cores = SysInfo.cores;
    // int memory = SysInfo.getTotalVirtualMemory() ~/ megaByte;

    // SamplingParams samplingParams = SamplingParams();

    Llama.libraryPath = "./libllama.dylib";
    Llama llama = Llama("/Users/adel/Downloads/gemma-7b-it-Q4_K_M.gguf");

    llama.setPrompt("2 * 2 = ?");
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

const int megaByte = 1024 * 1024;
