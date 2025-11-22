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
      ..nPredict = -1; // Unlimited generation

    // 2. Initialize Llama
    llama = Llama(
      modelPath,
      modelParams: ModelParams(),
      contextParams: contextParams,
      verbose: false,
    );

    print("\n=== STEP 1: Introduction ===");
    String prompt1 = _formatForGemma("Hello, I am Adel.");
    
    print("Sending: $prompt1");
    llama.setPrompt(prompt1);
    
    // Generate Response
    await _printResponse(llama);


    // --- STEP 2: MEMORY CHECK (State preserved) ---
    print("\n=== STEP 2: Memory Check (No Clear) ===");
    String prompt2 = _formatForGemma("What is my name?");
    
    print("Sending: $prompt2");
    llama.setPrompt(prompt2);
    
    // Generate Response
    await _printResponse(llama);


    // --- STEP 3: AMNESIA (Clear State) ---
    print("\n=== STEP 3: Amnesia (Clear called) ===");
    
    llama.clear();
    
    print("State cleared.");
    print("Sending: $prompt2");
    
    llama.setPrompt(prompt2);
    
    // Generate Response
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
    if (token.contains("<end_of_turn>")) break; // Stop at EOS
    stdout.write(token);
  }
  stdout.write("\n");
}

/// Helper to format prompt for Gemma
String _formatForGemma(String userText) {
  return "<start_of_turn>user\n$userText<end_of_turn><start_of_turn>model\n";
}