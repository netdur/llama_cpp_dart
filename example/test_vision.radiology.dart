import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

Future<void> main() async {
  Llama.libraryPath = "bin/MAC_ARM64/libmtmd.dylib";

  final modelParams = ModelParams()..nGpuLayers = -1;

  final contextParams = ContextParams()
    ..nPredict = 512
    ..nCtx = 8192
    ..nBatch = 8192;

  final samplerParams = SamplerParams()
    ..temp = 0.0
    ..topK = 64
    ..topP = 0.95
    ..penaltyRepeat = 1.1;

  final llama = Llama(
    "/Users/adel/Workspace/gguf/model-radiology-Q4_K_M.gguf",
    modelParams: modelParams,
    contextParams: contextParams,
    samplerParams: samplerParams,
    verbose: false,
    mmprojPath: "/Users/adel/Workspace/gguf/mmproj-radiology.gguf",
  );

  final image =
      LlamaImage.fromFile(File("/Users/adel/Downloads/radiology.png"));

  final chat = ChatHistory();
  chat.addMessage(role: Role.user, content: """<image>
      You are an expert radiographer. Describe accurately what you see in this image.""");
  final prompt =
      chat.exportFormat(ChatFormat.gemma, leaveLastAssistantOpen: true);

  final sw = Stopwatch()..start();
  try {
    final stream = llama.generateWithMedia(prompt, inputs: [image]);

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
