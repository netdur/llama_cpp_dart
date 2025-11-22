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

  // Create load command with mmproj path
  final loadCommand = LlamaLoad(
    path: "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf",
    modelParams: modelParams,
    contextParams: contextParams,
    samplingParams: samplerParams,
    mmprojPath: "/Users/adel/Workspace/gguf/mmproj-model-f16-4B.gguf",
  );

  // Create parent and initialize
  final parent = LlamaParent(loadCommand);
  await parent.init();

  // Load image
  final image = LlamaImage.fromFile(File("/Users/adel/Downloads/test-1.jpeg"));

  // Format prompt
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

  // Send prompt with image
  final promptId = await parent.sendPromptWithImages(prompt, [image]);

  // Listen to stream
  parent.stream.listen((token) {
    stdout.write(token);
  });

  // Wait for completion
  await parent.waitForCompletion(promptId);

  // Clean up
  await parent.dispose();
}
