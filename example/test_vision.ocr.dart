import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/llama_input.dart';

Future<void> main() async {
  Llama.libraryPath = "bin/MAC_ARM64/libmtmd.dylib";

  final modelParams = ModelParams()..nGpuLayers = 99;

  final contextParams = ContextParams()
    ..nPredict = -1
    ..nCtx = 8192
    ..nBatch = 8192;

  final samplerParams = SamplerParams()
    ..temp = 0.0
    ..topK = 64
    ..topP = 0.95
    ..penaltyRepeat = 1.1;

  final llama = Llama(
      "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf",
      modelParams,
      contextParams,
      samplerParams,
      false,
      "/Users/adel/Workspace/gguf/mmproj-model-f16-4B.gguf");

  final image = LlamaImage.fromFile(
      File("/Users/adel/Desktop/Screenshot 2025-06-30 at 23.35.12.png"));

  var prompt = """<image> extract information in structured JSON
  {
  "country": "",
  "card_type": "",
  "date_of_expiry": "31/12/2000",
  "card_number": "",
  "holder_name": "",
  "holder_surname": "",
  "holder_date_of_birth": "31/12/2000"
}
  """;

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
