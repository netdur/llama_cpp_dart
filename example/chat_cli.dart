// ignore_for_file: avoid_print

import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  try {
    print("Starting LLM CLI Chat App with Auto Context Management...");

    // Initialize model parameters
    ContextParams contextParams = ContextParams();
    contextParams.nPredict = -1;
    contextParams.nCtx = 256;
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

    // Initialize chat history with auto-trim capability
    ChatHistory chatHistory = ChatHistory(keepRecentPairs: 2);
    chatHistory.addMessage(role: Role.system, content: """
You are a helpful, concise assistant. Keep your answers informative but brief.""");

    print("\n=== Chat started (type 'exit' to quit) ===\n");

    // Chat loop
    bool chatActive = true;
    while (chatActive) {
      // Show remaining context space
      int remaining = llama.getRemainingContextSpace();
      if (remaining < 50) {
        print("⚠️ Low context space: $remaining tokens remaining");
      }

      // Get user input
      stdout.write("\nYou: ");
      String? userInput = stdin.readLineSync();

      // Check for exit command
      if (userInput == null || userInput.toLowerCase() == 'exit') {
        chatActive = false;
        print("\nExiting chat. Goodbye!");
        break;
      }

      // Check if we need to trim BEFORE adding the message
      if (chatHistory.shouldTrimBeforePrompt(llama, userInput)) {
        print("📝 Auto-trimming old messages to make space...");
        chatHistory.autoTrimForSpace(llama);
        
        // Clear llama context to match our trimmed history
        llama.clear();
        
        // Re-set the context with trimmed history
        String trimmedContext = chatHistory.exportFormat(ChatFormat.gemini);
        try {
          llama.setPrompt(trimmedContext);
        } catch (e) {
          print("Error resetting context: $e");
          continue;
        }
      }

      // Add user message to history
      chatHistory.addMessage(role: Role.user, content: userInput);

      // Add empty assistant message
      chatHistory.addMessage(role: Role.assistant, content: "");

      // Prepare prompt for the model
      String prompt = chatHistory.exportFormat(ChatFormat.gemini,
          leaveLastAssistantOpen: true);

      try {
        // Send to model
        llama.setPrompt(prompt);
      } catch (e) {
        if (e.toString().contains("Context") || e.toString().contains("context")) {
          // Auto-trim and retry
          print("\n📝 Context full! Auto-trimming conversation history...");
          
          // Remove the messages we just added
          chatHistory.messages.removeLast(); // Remove empty assistant
          chatHistory.messages.removeLast(); // Remove user message
          
          // Trim the history
          chatHistory.autoTrimForSpace(llama, reserveTokens: 150);
          
          // Clear and reset llama
          llama.clear();
          
          // Re-add the user message after trimming
          chatHistory.addMessage(role: Role.user, content: userInput);
          chatHistory.addMessage(role: Role.assistant, content: "");
          
          // Try again with trimmed context
          prompt = chatHistory.exportFormat(ChatFormat.gemini,
              leaveLastAssistantOpen: true);
          
          try {
            llama.setPrompt(prompt);
          } catch (e2) {
            print("Still failed after trimming: $e2");
            chatHistory.messages.removeLast();
            chatHistory.messages.removeLast();
            continue;
          }
        } else {
          rethrow;
        }
      }

      // Collect the response
      stdout.write("\nAssistant: ");
      StringBuffer responseBuffer = StringBuffer();
      bool endOfTurnFound = false;

      while (!endOfTurnFound) {
        var (token, done, contextLimit) = llama.getNextWithStatus();
        
        if (contextLimit) {
          print("\n\n⚠️ Hit context limit during generation!");
          break;
        }

        if (token.contains("<end_of_turn>")) {
          endOfTurnFound = true;
          String cleanToken =
              token.substring(0, token.indexOf("<end_of_turn>"));
          if (cleanToken.isNotEmpty) {
            stdout.write(cleanToken);
            responseBuffer.write(cleanToken);
          }
          break;
        }

        stdout.write(token);
        responseBuffer.write(token);

        if (done) break;
      }

      // Update the last assistant message
      String assistantResponse = responseBuffer.toString();
      if (assistantResponse.isNotEmpty) {
        chatHistory.messages.last =
            Message(role: Role.assistant, content: assistantResponse);
        chatHistory.fullHistory.last =
            Message(role: Role.assistant, content: assistantResponse);
      }

      print(""); // Newline after response
      
      // Show history status
      print(
        "\n[History: ${chatHistory.messages.length} active / "
        "${chatHistory.fullHistory.length} total messages]"
      );
    }

    // Clean up
    llama.dispose();
  } catch (e) {
    print("\nError: ${e.toString()}");
  }
}