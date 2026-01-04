// ignore_for_file: avoid_print

import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// Demonstrates the opt-in auto-trim behavior that frees KV cache space
/// by removing old tokens and shifting the cache so generation can continue
/// past the nominal context length.
void main() async {
  // Update these paths for your environment.
  Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
  const modelPath = "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf";

  final contextParams = ContextParams()
    ..nCtx = 256 // intentionally small to force trimming
    ..nBatch = 64
    ..nPredict = 400
    ..autoTrimContext = true
    ..trimKeepTokens = 128;

  final samplerParams = SamplerParams()
    ..temp = 0.7
    ..topK = 64
    ..topP = 0.9;

  final modelParams = ModelParams()..nGpuLayers = 99;

  const prompt =
      "You are a storyteller that keeps generating short numbered fragments. "
      "Keep going without stopping. Keep each fragment short (a sentence), "
      "but do not stop at 256 tokens; keep rolling.\n\n";

  Llama? llama;
  try {
    llama = Llama(
      modelPath,
      modelParams: modelParams,
      contextParams: contextParams,
      samplerParams: samplerParams,
      verbose: true, // emits auto-trim logs
    );

    llama.setPrompt(prompt);

    stdout.writeln("Generating with auto-trim enabled (n_ctx=${contextParams.nCtx})...");
    await for (final token in llama.generateText()) {
      stdout.write(token);
    }
    stdout.writeln("\n\nDone.");
  } catch (e) {
    stderr.writeln("Error: $e");
  } finally {
    llama?.dispose();
  }
}
