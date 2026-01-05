import 'dart:async';
import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/context_params.dart';

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    print('Usage: dart example/test_persistence.dart <path_to_model>');
    exit(1);
  }

  final modelPath = args[0];
  final contextParams = ContextParams();
  contextParams.nSeqMax = 1;
  contextParams.nCtx = 2048;

  final service = LlamaService(modelPath, contextParams: contextParams);

  service.createSession("persist_user");

  print("Step 1: Initial Prompt");
  await service.setPrompt(
      "persist_user", "I have a cat named Luna. Remember this.");
  await for (final chunk in service.generateText("persist_user")) {
    stdout.write(chunk);
  }
  print("\n--- Done Step 1 ---\n");

  print("Step 2: Follow-up (clearHistory: false)");
  // This should recall the cat's name
  await service.setPrompt("persist_user", "What is my cat's name?",
      clearHistory: false);

  await for (final chunk in service.generateText("persist_user")) {
    stdout.write(chunk);
  }
  print("\n--- Done Step 2 ---\n");

  await service.dispose();
}
