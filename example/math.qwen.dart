// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  try {
    ContextParams contextParams = ContextParams();
    contextParams.nPredict = -1;
    contextParams.nCtx = 512;
    contextParams.nBatch = 512;

    final samplerParams = SamplerParams();
    samplerParams.temp = 0.6;
    samplerParams.minP = 0;
    samplerParams.topK = 20;
    samplerParams.topP = 0.95;
    samplerParams.penaltyRepeat = 1.1;

    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath =
        "/Users/adel/.hugind/Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q8_0.gguf";
    Llama llama = Llama(
      modelPath,
      modelParams: ModelParams(),
      contextParams: contextParams,
      samplerParams: samplerParams,
      verbose: true,
    );

    llama.setPrompt("""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
what is 2 * 2?<|im_end|>
<|im_start|>assistant""");
    await for (final token in llama.generateText()) {
      stdout.write(token);
    }
    stdout.write("\n");

    llama.dispose();
  } catch (e) {
    print("Error: ${e.toString()}");
  }
}
