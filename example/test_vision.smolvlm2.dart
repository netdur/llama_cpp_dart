import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/llama_input.dart';

Future<void> main() async {
  Llama.libraryPath = "bin/MAC_ARM64/libmtmd.dylib";

  final modelParams = ModelParams()..nGpuLayers = -1;

  final contextParams = ContextParams()..nPredict = -1;

  final samplerParams = SamplerParams()
    ..temp = 0.2
    ..topP = 0.90;

  final llama = Llama(
      "/Users/adel/Workspace/gguf/SmolVLM-256M-Instruct-Q8_0.gguf",
      modelParams,
      contextParams,
      samplerParams,
      false,
      "/Users/adel/Workspace/gguf/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf");

  final image =
      LlamaImage.fromFile(File("/Users/adel/Downloads/image (8).png"));

  const prompt = '''
[
  {
    "role": "system",
    "content": [
      { "type": "text", "text": "You are a helpful vision assistant." }
    ]
  },
  {
    "role": "user",
    "content": [
      { "type": "<image>" },
      { "type": "text", "text": "What's in this image?" }
    ]
  }
]
''';

  final sw = Stopwatch()..start();
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
    sw.stop();
    stdout.writeln('⏱️  Inference time: ${sw.elapsed}');
    llama.dispose();
  }
}
