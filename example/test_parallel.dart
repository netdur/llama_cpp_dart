import 'dart:async';
import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/context_params.dart'; // Ensure this is accessible or use main export

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    print('Usage: dart example/test_parallel.dart <path_to_model_gguf>');
    exit(1);
  }

  final modelPath = args[0];

  // Configure for parallel execution
  final contextParams = ContextParams();
  contextParams.nSeqMax = 2; // We want 2 parallel sequences
  contextParams.nBatch = 512;
  contextParams.nCtx = 2048;

  print("Loading model from $modelPath with nSeqMax=2...");

  final service =
      LlamaService(modelPath, contextParams: contextParams, verbose: false);

  print("Service initialized. Creating sessions...");

  service.createSession("user1");
  service.createSession("user2");

  print("Sessions created. Starting parallel generation...");

  final prompt1 = "Write a short poem about the sun.";
  final prompt2 = "Write a short poem about the moon.";

  // Start both
  await service.setPrompt("user1", prompt1);
  await service.setPrompt("user2", prompt2);

  final stream1 = service.generateText("user1");
  final stream2 = service.generateText("user2");

  // Interleave printing
  final done1 = Completer<void>();
  final done2 = Completer<void>();

  final out1 = StringBuffer();
  final out2 = StringBuffer();

  // Listen to 1
  stream1.listen((s) {
    stdout.write("[User1]: $s\n");
    out1.write(s);
  }, onDone: () => done1.complete());

  // Listen to 2
  stream2.listen((s) {
    stdout.write("[User2]: $s\n");
    out2.write(s);
  }, onDone: () => done2.complete());

  await Future.wait([done1.future, done2.future]);

  print("\n--- DONE ---\n");
  print("User 1 Result:\n${out1.toString()}\n");
  print("User 2 Result:\n${out2.toString()}\n");

  service.dispose();
}
