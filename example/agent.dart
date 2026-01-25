// ignore_for_file: avoid_print

import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  Llama? llama;
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath =
        "/Users/adel/.hugind/openbmb/AgentCPM-Explore-GGUF/AgentCPM-Explore.Q4_K_M.gguf";

    final contextParams = ContextParams()
      ..nCtx = 8192 // Increased to handle thought blocks and agentic history
      ..nBatch = 512
      ..nPredict = 1024; // Ensure it doesn't run forever, but has room to think

    ModelParams modelParams = ModelParams();

    // AgentCPM-Explore prefers lower temperature for precision in agentic tasks
    SamplerParams samplerParams = SamplerParams()
      ..temp = 0.2
      ..topP = 0.95
      ..penaltyRepeat =
          1.05; // Slightly lowered to prevent breaking formatting tags

    llama = Llama(
      modelPath,
      modelParams: modelParams,
      contextParams: contextParams,
      samplerParams: samplerParams,
      verbose: false, // Set to false for cleaner output
    );

    // --- Template Implementation ---
    // Initialize History
    ChatHistory history = ChatHistory();

    // Add messages to history
    history.addMessage(
        role: Role.system,
        content: 'You are a helpful assistant respond in valid json.');
    history.addMessage(
        role: Role.user,
        content:
            "what is command to list files by size, please provide only command");

    // Use ChatFormat.chatmlThinking to get the formatted prompt
    // We use exportFormat instead of getLatestTurn because we want the whole initial prompt (including system)
    // and we want to trigger the assistant with <|thought|> (leaveLastAssistantOpen: true)
    String formattedPrompt = history.exportFormat(ChatFormat.chatmlThinking,
        leaveLastAssistantOpen: true);

    // print("Sending:\n$formattedPrompt");
    llama.setPrompt(formattedPrompt);

    print("--- Model Reasoning & Response ---");
    await _printResponse(llama);

    llama.clear();
    llama.dispose();
  } catch (e) {
    print("\nError: ${e.toString()}");
    llama?.dispose();
  }
}

/// Helper to print generation output
Future<void> _printResponse(Llama llama) async {
  await for (final token in llama.generateText()) {
    // AgentCPM-Explore uses <|im_end|> as the stop token
    if (token.contains("<|im_end|>")) break;
    stdout.write(token);
  }
  stdout.write("\n");
}
