import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

Future<void> main() async {
  Llama.libraryPath = "bin/MAC_ARM64/libmtmd.dylib";

  final modelParams = ModelParams()..nGpuLayers = 0;

  final contextParams = ContextParams()
    ..nPredict = -1
    ..nCtx = 8192
    ..nBatch = 8192;

  final samplerParams = SamplerParams()
    ..temp = 0.7
    ..topK = 64
    ..topP = 0.95
    ..penaltyRepeat = 1.1;

  final llama = Llama(
      "/Users/adel/Workspace/gguf/model_tq2_0.gguf",
      modelParams,
      contextParams,
      samplerParams,
      false,
      "/Users/adel/Workspace/gguf/mmproj_tq2_0.gguf");

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
<start_of_turn>user\n
<image>\n $prompt
<start_of_turn>model\n""";

  try {
    final stream = llama.generateWithMeda(prompt, inputs: [image]);

    await for (final token in stream) {
      stdout.write(token);
    }
    await stdout.flush();
    stdout.writeln();
  } on LlamaException catch (e) {
    stderr.writeln("An error occurred: $e");
  } finally {
    llama.dispose();
  }
}
