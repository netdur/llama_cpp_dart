// ignore_for_file: avoid_print

import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";

  ContextParams contextParams = ContextParams();
  contextParams.nPredict = -1;
  contextParams.nCtx = 8192;

  final samplerParams = SamplerParams();
  samplerParams.temp = 0.7;
  samplerParams.topK = 64;
  samplerParams.topP = 0.95;
  samplerParams.penaltyRepeat = 1.1;

  String modelPath = "/Users/adel/Workspace/gguf/gemma-3-12b-it-q4_0.gguf";
  final loadCommand = LlamaLoad(
      path: modelPath,
      modelParams: ModelParams(),
      contextParams: contextParams,
      samplingParams: samplerParams);

  final llamaParent = LlamaParent(loadCommand);

  try {
    print("Initializing model...");
    await llamaParent.init();
    print("Model initialized successfully");

    // Create two separate scopes
    LlamaScope scope1 = llamaParent.getScope();
    LlamaScope scope2 = llamaParent.getScope();

    // Set up listeners for scope1
    scope1.stream.listen((text) {
      print("SCOPE 1: $text");
    });
    scope1.completions.listen((event) {
      print("SCOPE 1 DONE: ${event.success}");

      // After scope1 completes, we'll send a prompt to scope2
      print("\n----- Scope 1 finished, now running Scope 2 prompt -----\n");
    });

    // Set up listeners for scope2
    scope2.stream.listen((text) {
      print("SCOPE 2: $text");
    });
    scope2.completions.listen((event) {
      print("SCOPE 2 DONE: ${event.success}");

      // When scope2 is done, we'll clean up and exit
      print("\n----- All prompts completed, cleaning up -----\n");
      scope1.dispose();
      scope2.dispose();
      llamaParent.dispose();
    });

    // Queue both prompts - they'll run sequentially
    print("Sending prompt to Scope 1...");
    await scope1.sendPrompt(getPrompt("What is 2 * 4?"));
    print("Queuing prompt for Scope 2...");
    await scope2.sendPrompt(getPrompt("What is your name?"));

    print(
        "Both prompts have been queued. Processing will happen sequentially.");

    // The program will continue running until the completion listeners clean up
  } catch (e) {
    stderr.writeln("Error: $e");
    await llamaParent.dispose();
    exit(1);
  }
}

String getPrompt(String content) {
  ChatHistory history = ChatHistory()
    ..addMessage(role: Role.user, content: content)
    ..addMessage(role: Role.assistant, content: "");
  return history.exportFormat(ChatFormat.gemini, leaveLastAssistantOpen: true);
}
