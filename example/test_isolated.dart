// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/chatml_format.dart';

void main() async {
  Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";

  ContextParams contextParams = ContextParams();
  contextParams.nPredit = 100;

  final loadCommand = LlamaLoad(
    path: "/Users/adel/Downloads/Qwen2-7B-Multilingual-RP-Q4_K_M.gguf",
    modelParams: ModelParams(),
    contextParams: contextParams,
    samplingParams: SamplerParams(),
    format: ChatMLFormat(),
  );

  final llamaParent = LlamaParent(loadCommand);
  await llamaParent.init();

  llamaParent.stream.listen((response) {
    stdout.write(response);
  });

  llamaParent.sendPrompt("2 * 2 = ?");

  await Future.delayed(Duration(seconds: 12));
  await llamaParent.dispose();
}
