// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";

  ContextParams contextParams = ContextParams();
  contextParams.nPredict = -1;
  contextParams.nCtx = 8192;
  contextParams.nBatch = 8192;

  final samplerParams = SamplerParams();
  samplerParams.temp = 0.7;
  samplerParams.topK = 64;
  samplerParams.topP = 0.95;
  samplerParams.penaltyRepeat = 1.1;

  String modelPath = "/Users/adel/Workspace/gguf/gemma-3-12b-it-q4_0.gguf";
  final loadCommand = LlamaLoad(
    path: modelPath,
    modelParams: ModelParams(),
    contextParams: contextParams,
    samplingParams: samplerParams,
    // format: GeminiFormat(), <- unused
  );

  final llamaParent = LlamaParent(loadCommand);
  await llamaParent.init();

  int i = 0;
  List<String> prompts = [
    "<start_of_turn>What is 2 * 4?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>What is 4 * 4?<end_of_turn>\n<start_of_turn>model\n",
    """<start_of_turn>user\n
    I’ve been testing out LLM

    hey how are you?\n<end_of_turn>\n<start_of_turn>model\n""",
    "<start_of_turn>hey what is your name?<end_of_turn>\n<start_of_turn>model\n",
  ];

  llamaParent.stream.listen((response) {
    stdout.write(response);
  });

  llamaParent.completions.listen((event) {
    i++;
    if (i >= prompts.length) {
      llamaParent.dispose();
    } else {
      print("");
      llamaParent.sendPrompt(prompts[i]);
    }
  });

  llamaParent.sendPrompt(prompts[0]);
}

/*
"""<start_of_turn>user\nHey everyone,

I’ve been testing out Gemini 2.5 Pro Preview inside Google AI Studio and I’m a bit confused about the pricing.

I’m not using the API or Vertex AI, just playing around in AI Studio (the browser interface). According to the docs, it should be free for now—but I’ve seen a few posts from people saying they got charged \$500+ after just a few hours of use. 😳

So my question is:
Is Gemini 2.5 Pro Preview actually free in AI Studio right now?
Has anyone here received a bill even though they only used it inside AI Studio?

Just trying to make sure I’m not missing something that could cost me big later on.

Thanks in advance!\n<end_of_turn>\n<start_of_turn>model\n
""",
*/
