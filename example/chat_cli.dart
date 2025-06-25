// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  try {
    print("Starting LLM CLI Chat App...");

    // Initialize model parameters
    ContextParams contextParams = ContextParams();
    contextParams.nPredict = 8192;
    contextParams.nCtx = 8192;
    contextParams.nBatch = 512;

    final samplerParams = SamplerParams();
    samplerParams.temp = 0.7;
    samplerParams.topK = 64;
    samplerParams.topP = 0.95;
    samplerParams.penaltyRepeat = 1.1;

    // Load the LLM model
    print("Loading model, please wait...");
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf";
    Llama llama =
        Llama(modelPath, ModelParams(), contextParams, samplerParams, false);
    print("Model loaded successfully! ${llama.status}");

    // Initialize chat history with system prompt
    ChatHistory chatHistory = ChatHistory();
    chatHistory.addMessage(role: Role.system, content: """
You are a helpful, concise assistant. Keep your answers informative but brief.""");

    print("\n=== Chat started (type 'exit' to quit) ===\n");

    // Chat loop
    bool chatActive = true;
    while (chatActive) {
      // Get user input
      stdout.write("\nYou: ");
      String? userInput = stdin.readLineSync();

      // Check for exit command
      if (userInput == null || userInput.toLowerCase() == 'exit') {
        chatActive = false;
        print("\nExiting chat. Goodbye!");
        break;
      }

      // Add user message to history
      chatHistory.addMessage(role: Role.user, content: userInput);

      // Add empty assistant message that will be filled by the model
      chatHistory.addMessage(role: Role.assistant, content: "");

      // Prepare prompt for the model
      String prompt = chatHistory.exportFormat(ChatFormat.gemini,
          leaveLastAssistantOpen: true);

      // Send to model
      llama.setPrompt(prompt);

      // Collect the response
      stdout.write("\nAssistant: ");
      StringBuffer responseBuffer = StringBuffer();
      bool endOfTurnFound = false;

      while (!endOfTurnFound) {
        var (token, done) = llama.getNext();

        // Check if we've found the end marker
        if (token.contains("<end_of_turn>")) {
          endOfTurnFound = true;
          // Only print up to the end marker
          String cleanToken =
              token.substring(0, token.indexOf("<end_of_turn>"));
          if (cleanToken.isNotEmpty) {
            stdout.write(cleanToken);
            responseBuffer.write(cleanToken);
          }
          break;
        }

        // Print and collect the token
        stdout.write(token);
        responseBuffer.write(token);

        // Break if the model is done
        if (done) break;
      }

      // Update the last assistant message with the generated content
      String assistantResponse = responseBuffer.toString();
      chatHistory.messages.last =
          Message(role: Role.assistant, content: assistantResponse);

      print(""); // Add a newline after the response
    }

    // Clean up
    llama.dispose();
  } catch (e) {
    print("\nError: ${e.toString()}");
  }
}
