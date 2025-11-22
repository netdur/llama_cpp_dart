// ignore_for_file: avoid_print

import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

Future<void> main() async {
  Llama.libraryPath = "bin/MAC_ARM64/libmtmd.dylib";

  final modelParams = ModelParams()..nGpuLayers = -1;

  final contextParams = ContextParams()
    ..nPredict = -1
    ..nCtx = 4096
    ..nBatch = 1024;

  final samplerParams = SamplerParams()
    ..temp = 0.25
    ..topP = 0.90;

  final llama = Llama(
    "/Users/adel/Workspace/gguf/SmolVLM-500M-Instruct-Q8_0.gguf",
    modelParams: modelParams,
    contextParams: contextParams,
    samplerParams: samplerParams,
    verbose: false,
    mmprojPath: "/Users/adel/Workspace/gguf/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
  );

  final image = LlamaImage.fromFile(File("/Users/adel/Downloads/test.jpg"));
  var prompt = """
Generate a detailed product listing for this item,
including a title, key features, and a description.
output shouly be only, title, key features and description, DO NOT ADD ANYTHING ELSE OR ASK QUESTION OR MAKE SUGGESTIONS
example of output:

**Title:** {title}"

**Key Features:**
  - feature

**Description:** {description}"
""";
  prompt = """
<|im_start|>System: You are a helpful vision assistant.<end_of_utterance>
<|im_start|>User: <image> $prompt<end_of_utterance>
Assistant:
""";

  try {
    print("First generation:");
    final stream = llama.generateWithMedia(prompt, inputs: [image]);

    await for (final token in stream) {
      stdout.write(token);
    }
    await stdout.flush();
    stdout.writeln();
  } on LlamaException catch (e) {
    stderr.writeln("An error occurred: $e");
  }

  llama.dispose();
}
