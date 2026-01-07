import 'dart:async';
import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    print('Usage: dart example/simple_concurrent.dart <path_to_model>');
    exit(1);
  }

  // 1. Setup library path (Critical for Mac)
  if (Platform.isMacOS) {
    Llama.libraryPath =
        "/Users/adel/Workspace/llama_cpp_dart/bin/MAC_ARM64/libllama.dylib";
  }

  // 2. Setup config for concurrency
  final contextParams = ContextParams();
  contextParams.nSeqMax =
      2; // Critical: Allocate space for 2 concurrent sequences
  contextParams.nCtx = 2048; // Shared context window

  // 2. Initialize Service
  final service = LlamaService(args[0], contextParams: contextParams, verbose: false);

  print("Service initialized. Creating sessions...");

  // 3. Create logical sessions
  service.createSession("user_1");
  service.createSession("user_2");

  // 4. Define prompts (using Gemma chat format for this example)
  final prompt1 =
      "<start_of_turn>user\nCount to 5<end_of_turn>\n<start_of_turn>model\n";
  final prompt2 =
      "<start_of_turn>user\nList 3 colors<end_of_turn>\n<start_of_turn>model\n";

  print("Sending concurrent requests...");

  // 5. Run requests in parallel
  final task1 = _streamResponse(service, "user_1", prompt1);
  final task2 = _streamResponse(service, "user_2", prompt2);

  await Future.wait([task1, task2]);

  print("\nAll done!");
  await service.dispose();
}

Future<void> _streamResponse(
    LlamaService service, String sessionId, String prompt) async {
  final buffer = StringBuffer();

  // Listen to the stream *before* setting the prompt to capture everything
  final subscription = service.generateText(sessionId).listen((chunk) {
    buffer.write(chunk);
  });

  // Trigger generation
  await service.setPrompt(sessionId, prompt);

  // Poll until finished
  while (service.status(sessionId) == LlamaStatus.generating) {
    await Future.delayed(const Duration(milliseconds: 10));
  }

  await subscription.cancel();

  print("\n--- Result for $sessionId ---");
  print(buffer.toString().trim());
}
