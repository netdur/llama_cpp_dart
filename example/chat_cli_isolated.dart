// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:async';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  print("Starting LLM CLI Chat App with Isolates...");

  // Library path setup
  Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";

  // Setup parameters
  ContextParams contextParams = ContextParams();
  contextParams.nPredict = 8192;
  contextParams.nCtx = 8192;
  contextParams.nBatch = 512;

  final samplerParams = SamplerParams();
  samplerParams.temp = 0.7;
  samplerParams.topK = 64;
  samplerParams.topP = 0.95;
  samplerParams.penaltyRepeat = 1.1;

  // Initialize load command for the isolate
  final loadCommand = LlamaLoad(
    path: "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf",
    modelParams: ModelParams(),
    contextParams: contextParams,
    samplingParams: samplerParams,
  );

  print("Loading model, please wait...");

  // Create the LLM parent that will spawn an isolate
  final llamaParent = LlamaParent(loadCommand);

  try {
    await llamaParent.init();

    // Add a timeout to prevent infinite waiting
    int attempts = 0;
    const maxAttempts = 60;

    print("Waiting for model to be ready...");
    while (llamaParent.status != LlamaStatus.ready && attempts < maxAttempts) {
      await Future.delayed(Duration(milliseconds: 500));
      attempts++;

      if (attempts % 10 == 0) {
        print("Still waiting... Status: ${llamaParent.status}");
      }

      if (llamaParent.status == LlamaStatus.error) {
        print("Error loading model. Exiting.");
        exit(1);
      }
    }

    if (attempts >= maxAttempts && llamaParent.status != LlamaStatus.ready) {
      print(
          "Timeout waiting for model to be ready. Current status: ${llamaParent.status}");
      print(
          "Continuing anyway as the model might be ready despite status not being updated...");
    }

    print(
        "Model loaded successfully in isolate! Status: ${llamaParent.status}");
  } catch (e) {
    print("Error initializing model: $e");
    exit(1);
  }

  // Initialize chat history with system prompt
  ChatHistory chatHistory = ChatHistory();
  chatHistory.addMessage(
      role: Role.system,
      content:
          "You are a helpful, concise assistant. Keep your answers informative but brief.");
  print("Chat history initialized with system prompt");

  print("\n=== Chat started (type 'exit' to quit) ===\n");

  // Set up a completer to help manage when completions are finished
  Completer<void> completionDone = Completer<void>();
  StringBuffer currentResponse = StringBuffer();
  // bool processingMessage = false;

  llamaParent.stream.listen((token) {
    stdout
      ..write(token)
      ..flush();

    currentResponse.write(token);
  }, onError: (e) {
    print("\nSTREAM ERROR: $e");
  });

  // Listen for completion events
  llamaParent.completions.listen((event) {
    if (event.success) {
      if (chatHistory.messages.isNotEmpty &&
          chatHistory.messages.last.role == Role.assistant) {
        chatHistory.messages.last =
            Message(role: Role.assistant, content: currentResponse.toString());
      }
      currentResponse.clear();
      if (!completionDone.isCompleted) {
        completionDone.complete();
      }
    } else {
      print("Completion failed for prompt: ${event.promptId}");
    }
  });

  // Chat loop
  bool chatActive = true;
  while (chatActive) {
    // Get user input
    stdout.write("\nYou: ");
    String? userInput = stdin.readLineSync();

    // Check for exit command
    if (userInput == null || userInput.toLowerCase() == 'exit') {
      chatActive = false;
      print("\nExiting chat. bye!");
      print(chatHistory.exportFormat(ChatFormat.gemini));
      break;
    }

    // Add user message to history
    chatHistory.addMessage(role: Role.user, content: userInput);

    // Add empty assistant message
    chatHistory.addMessage(role: Role.assistant, content: "");

    // Create a new completer for this message
    completionDone = Completer<void>();

    // Prepare prompt for the model
    String prompt = chatHistory.exportFormat(ChatFormat.gemini,
        leaveLastAssistantOpen: true);

    await llamaParent.sendPrompt(prompt);

    // Indicate that we're about to process a new message
    stdout.write("\nAssistant: ");

    // processingMessage = true;

    // Wait for completion before continuing to next message
    try {
      await completionDone.future.timeout(Duration(seconds: 60), onTimeout: () {
        print("\nTimeout waiting for response. Continuing anyway...");
      });
    } catch (e) {
      print("\nError waiting for completion: $e");
    }
    print(""); // Add a newline after the response
  }

  // Clean up
  llamaParent.dispose();
}
