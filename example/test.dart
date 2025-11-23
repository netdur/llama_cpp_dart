// ignore_for_file: avoid_print

import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  Llama? llama;
  try {
    // 1. Setup
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf";

    final contextParams = ContextParams()
      ..nCtx = 2048
      ..nPredict = -1;

    // 2. Initialize Llama
    llama = Llama(
      modelPath,
      modelParams: ModelParams(),
      contextParams: contextParams,
      verbose: false,
    );

    // Initialize History
    ChatHistory history = ChatHistory();

    // --- STEP 1: INTRO (State starts at 0) ---
    print("\n=== STEP 1: Introduction ===");
    
    // Add to history object
    history.addMessage(role: Role.user, content: "Hello, I am Adel.");
    
    // Use the new helper to get just this message formatted
    String prompt1 = history.getLatestTurn(ChatFormat.gemma);
    
    print("Sending:\n$prompt1");
    llama.setPrompt(prompt1);
    
    await _printResponse(llama);


    // --- STEP 2: MEMORY CHECK (State preserved) ---
    print("\n=== STEP 2: Memory Check (No Clear) ===");
    
    history.addMessage(role: Role.user, content: "What is my name?");
    
    // getLatestTurn gets ONLY the message we just added
    String prompt2 = history.getLatestTurn(ChatFormat.gemma);
    
    print("Sending:\n$prompt2");
    llama.setPrompt(prompt2); // Appends to existing VRAM context
    
    await _printResponse(llama);


    // --- STEP 3: AMNESIA (Clear State) ---
    print("\n=== STEP 3: Amnesia (Clear called) ===");
    
    llama.clear(); // Wipes VRAM
    
    print("State cleared.");
    print("Sending:\n$prompt2");
    
    // NOTE: Here we send ONLY prompt2 ("What is my name?") into an empty brain.
    // We intentionally DO NOT send the full history, to prove it forgot.
    llama.setPrompt(prompt2); 
    
    await _printResponse(llama);

    llama.dispose();
  } catch (e) {
    print("\nError: ${e.toString()}");
    llama?.dispose();
  }
}

/// Helper to print generation output
Future<void> _printResponse(Llama llama) async {
  stdout.write("AI: ");
  await for (final token in llama.generateText()) {
    // Stop if we see the end tag (logic is now inside Llama class to strip it, 
    // but breaking the loop is still good practice)
    if (token.contains("<end_of_turn>")) break; 
    stdout.write(token);
  }
  stdout.write("\n");
}