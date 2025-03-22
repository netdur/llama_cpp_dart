// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";

  ContextParams contextParams = ContextParams();
  contextParams.nPredict = 500;
  contextParams.nCtx = 8192;
  contextParams.nBatch = 8192;

  final samplerParams = SamplerParams();
  samplerParams.temp = 1.0;
  samplerParams.topK = 64;
  samplerParams.topP = 0.95;
  samplerParams.penaltyRepeat = 1.1;

  String modelPath = "/Users/adel/Workspace/gguf/gemma-3-12b-it-Q4_K_M.gguf";
  final loadCommand = LlamaLoad(
    path: modelPath,
    modelParams: ModelParams(),
    contextParams: contextParams,
    samplingParams: samplerParams,
    // format: ChatMLFormat(),
  );

  final llamaParent = LlamaParent(loadCommand);
  await llamaParent.init();

  int i = 0;
  List<String> prompts = [
    "<start_of_turn>What is 2 * 4?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>What is 4 * 4?<end_of_turn>\n<start_of_turn>model\n"
  ];

  llamaParent.stream.listen((response) {
    stdout.write(response);
  });

  llamaParent.completions.listen((event) {
    i++;
    if (i >= prompts.length) {
      llamaParent.dispose();
    } else {
      llamaParent.sendPrompt(prompts[i]);
    }
  });

  llamaParent.sendPrompt(prompts[0]);
}
