import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:system_info2/system_info2.dart';

void main() async {
  final cores = SysInfo.cores;
  int memory = SysInfo.getTotalVirtualMemory() ~/ megaByte;

  Llama.libraryPath = "libllama.so";

  ModelParams modelParams = ModelParams();

  ContextParams contextParams = ContextParams();
  contextParams.threads = cores.length;
  contextParams.threadsBatch = cores.length;

  Llama llama =
      Llama("mistral-7b-openorca.Q5_K_M.gguf", modelParams, contextParams);

  String system =
      "You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!";
  String prompt = "How are you?";
  String input = """<|im_start|>system\n
$system<|im_end|>\n
<|im_start|>user\n
$prompt<|im_end|>\n
<|im_start|>assistant\n
""";

  llama.setPrompt(input);
  while (true) {
    var (token, done) = llama.getNext();
    stdout.write(token);
    if (done) break;
  }
  stdout.write("\n");

  llama.dispose();
}

const int megaByte = 1024 * 1024;
