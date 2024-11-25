// ignore_for_file: avoid_print

import 'dart:io';

// import 'package:flutter/foundation.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/chat.dart';
import 'package:llama_cpp_dart/src/chatml_format.dart';
// import 'package:system_info2/system_info2.dart';

void main() async {
  try {
    // final cores = SysInfo.cores;
    // int memory = SysInfo.getTotalVirtualMemory() ~/ megaByte;

    NewLlama.libraryPath = "./libllama.dylib";

    // SamplerParams samplingParams = SamplerParams();
    // samplingParams.penaltyRepeat = 1.1;
    // samplingParams.temp = 0.7;

    // ModelParams modelParams = ModelParams();

    // ContextParams contextParams = ContextParams();
    // contextParams.threads = cores.length;
    // contextParams.threadsBatch = cores.length;
    // contextParams.context = 512 * 4;

    // /Users/adel/Workspace/llama.cpp/models/tinyllama-2-1b-miniguanaco.Q3_K_L.gguf
    // /Users/adel/Workspace/llama.cpp/models/openhermes-2.5-neural-chat-v3-3-slerp.Q5_K_M.gguf
    NewLlama llama = NewLlama("/Users/adel/Downloads/gemma-7b-it-Q4_K_M.gguf");
        // modelParams, contextParams, samplingParams);

    ChatMLFormat chatMLFormat = ChatMLFormat();
    // AlpacaFormat alpacaFormat = AlpacaFormat();

    String system = chatMLFormat.preparePrompt(
        """Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.
Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.
""", Role.system.value, false);

    String prompt =
        chatMLFormat.preparePrompt("What was OKT3 originally sourced from?");

    llama.setPrompt(system + prompt);
    while (true) {
      var (token, done) = llama.getNext();
      String? chunk = chatMLFormat.filterResponse(token);
      if (chunk != null) stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");
    stdout.write("\n");

    prompt = chatMLFormat.preparePrompt("What was the company called?");
    llama.setPrompt(system + prompt);
    while (true) {
      var (token, done) = llama.getNext();
      String? chunk = chatMLFormat.filterResponse(token);
      if (chunk != null) stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");
    //*/

    llama.dispose();
  } catch (e) {
    print("Error: ${e.toString()}");
  }
}

const int megaByte = 1024 * 1024;
