import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/chat.dart';
import 'package:llama_cpp_dart/src/chat_ml_filter.dart';
import 'package:llama_cpp_dart/src/prompt_format.dart';
import 'package:system_info2/system_info2.dart';

void main() async {
  final cores = SysInfo.cores;
  int memory = SysInfo.getTotalVirtualMemory() ~/ megaByte;

  Llama.libraryPath = "/Users/adel/Workspace/llama.cpp/build/libllama.dylib";

  ModelParams modelParams = ModelParams();

  ContextParams contextParams = ContextParams();
  contextParams.threads = cores.length;
  contextParams.threadsBatch = cores.length;

  Llama llama = Llama(
      "/Users/adel/Workspace/llama.cpp/models/mistral-7b-openorca.Q5_K_M.gguf",
      modelParams,
      contextParams);

  String system = PromptFormatter.formatPrompt(
      PromptFormat.chatml,
      "You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!",
      Role.system.value,
      false);
  String prompt =
      PromptFormatter.formatPrompt(PromptFormat.chatml, "How are you?");

  ChatMLFilter chatmlFilter = ChatMLFilter();
  llama.setPrompt(system + prompt);
  while (true) {
    var (token, done) = llama.getNext();
    String? chunk = chatmlFilter.processChunk(token);
    if (chunk != null) stdout.write(token);
    if (done) break;
  }
  stdout.write("\n");

  llama.dispose();
}

const int megaByte = 1024 * 1024;
