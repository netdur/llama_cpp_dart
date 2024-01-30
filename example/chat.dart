import 'dart:io';

// import 'package:flutter/foundation.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/chat.dart';
import 'package:llama_cpp_dart/src/chatml_format.dart';
import 'package:system_info2/system_info2.dart';

void main() async {
  try {
    final cores = SysInfo.cores;
    // int memory = SysInfo.getTotalVirtualMemory() ~/ megaByte;

    Llama.libraryPath = "/Users/adel/Workspace/llama.cpp/build/libllama.dylib";

    ModelParams modelParams = ModelParams();

    ContextParams contextParams = ContextParams();
    contextParams.threads = cores.length;
    contextParams.threadsBatch = cores.length;
    contextParams.context = 512 * 4;

    Llama llama = Llama(
        "/Users/adel/Workspace/llama.cpp/models/mistral-7b-openorca.Q5_K_M.gguf",
        modelParams,
        contextParams);

    ChatMLFormat chatMLFormat = ChatMLFormat();

    String system = chatMLFormat.preparePrompt(
        "You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!",
        Role.system.value,
        false);

    String prompt = chatMLFormat.preparePrompt("how are you?");

    llama.setPrompt(system + prompt);
    while (true) {
      var (token, done) = llama.getNext();
      String? chunk = chatMLFormat.filterResponse(token);
      if (chunk != null) stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");

    // llama.clear();

    /*
    prompt = chatMLFormat.preparePrompt("Can you divide by zero?");
    await for (String token in llama.prompt(prompt)) {
      String? chunk = chatMLFormat.filterResponse(token);
      if (chunk != null) stdout.write(token);
    }
    stdout.write("\n");*/

    llama.dispose();
  } catch (e) {
    print("Error: ${e.toString()}");
  }
}

const int megaByte = 1024 * 1024;
