import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

Future<void> main() async {
  Llama.libraryPath = "bin/MAC_ARM64/libmtmd.dylib";

  final modelParams = ModelParams()..nGpuLayers = 99;

  final contextParams = ContextParams()
    ..nPredict = -1
    ..nCtx = 2048
    ..nUbatch = 2048
    ..nBatch = 2048;

  final samplerParams = SamplerParams()
    ..temp = 0.7
    ..topK = 64
    ..topP = 0.95
    ..penaltyRepeat = 1.1;

  final llama = Llama(
      // "/Users/adel/Downloads/gemma-3n-E2B-it-plantvillage-model.gguf",
      "/Users/adel/Downloads/gemma-3-4B-it-plantvillage-model-Q4_K_M.gguf",
      modelParams,
      contextParams,
      samplerParams,
      true,
      "/Users/adel/Downloads/gemma-3-4B-it-plantvillage-mmproj.gguf");

  final image =
      LlamaImage.fromFile(File("/Users/adel/Downloads/plantvillage-test.png"));
  final prompt = """
<start_of_turn>user\n
<image>\n
You are an expert plant pathologist.
Your task is to identify the plant species and its health status from the image.
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
