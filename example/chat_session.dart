// ignore_for_file: avoid_print

import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main(List<String> args) async {
  try {
    if (args.isEmpty) {
      print("Usage: dart chat_session.dart <session_name>");
      print("Example: dart chat_session.dart one_session");
      exit(1);
    }

    String sessionName = args[0];
    String sessionPath = "$sessionName.bin";

    print("You:");
    String? userInput = stdin.readLineSync();
    if (userInput == null || userInput.trim().isEmpty) {
      print("No input provided. Exiting.");
      exit(1);
    }

    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf";

    ChatHistory history = ChatHistory()
      ..addMessage(role: Role.user, content: userInput.trim())
      ..addMessage(role: Role.assistant, content: "");

    final modelParams = ModelParams()..nGpuLayers = 99;

    final contextParams = ContextParams()
      ..nPredict = 100
      ..nCtx = 1024 * 4
      ..nBatch = 512;

    final samplerParams = SamplerParams()
      ..temp = 0.7
      ..topK = 64
      ..topP = 0.95
      ..penaltyRepeat = 1.1;

    Llama llama =
        Llama(modelPath, modelParams, contextParams, samplerParams, false);

    if (File(sessionPath).existsSync()) {
      try {
        llama.loadSession(sessionPath);
        print("+");
      } catch (e) {
        print("!");
      }
    }

    llama.setPrompt(
        history.exportFormat(ChatFormat.gemini, leaveLastAssistantOpen: true));

    print("AI:");
    while (true) {
      var (token, done) = llama.getNext();
      stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");

    llama.saveSession(sessionPath);

    llama.dispose();
  } catch (e) {
    print("\nError: ${e.toString()}");
  }
}
