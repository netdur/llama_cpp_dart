import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

Future<void> main() async {
  // Ensure the library path is correct for your system
  Llama.libraryPath = "bin/MAC_ARM64/libmtmd.dylib";

  final modelParams = ModelParams()..nGpuLayers = 99;

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
    "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf",
    modelParams: modelParams,
    contextParams: contextParams,
    samplerParams: samplerParams,
    verbose: false,
    mmprojPath: "/Users/adel/Workspace/gguf/mmproj-model-f16-4B.gguf",
  );

  // Initialize ChatHistory
  final chat = ChatHistory();

  // Add a message with an image attached
  // The ChatHistory will automatically handle the <image> tag insertion
  chat.addMessage(
    role: Role.user,
    content: "Describe this image in detail",
    images: ["/Users/adel/Downloads/test-1.jpeg"],
  );

  // Export the prompt and inputs using the new convenience method
  final (prompt, inputs) =
      chat.exportWithMedia(ChatFormat.gemma, leaveLastAssistantOpen: true);

  print("Generated Prompt:\n$prompt\n");

  try {
    final stream = llama.generateWithMedia(prompt, inputs: inputs);

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
