// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

String longText = """
Okay, let's break down those parameters and why they might be causing a crash with long text, especially focusing on the memory implications.

The most likely culprit here is the combination of a large nCtx and an equally large nBatch, leading to excessive memory allocation, particularly during the initial prompt processing phase.

Here's a breakdown:

nCtx = 8192 (Context Size):

This defines the maximum number of tokens (roughly words/subwords) the model can "remember" at any given time (prompt + generation).

An 8192 context is quite large and requires significant memory (RAM, or VRAM if GPU offloaded). The exact amount depends on the model's architecture (number of layers, embedding size, etc.).

Potential Issue: If your system doesn't have enough free RAM/VRAM to allocate the Key-Value (KV) cache for an 8192 context, llama_init_from_file or the first llama_decode might fail or crash. Ensure the model you are using actually supports an 8192 context effectively. Some models might allow setting it but perform poorly or have underlying limits.

nBatch = 8192 (Batch Size):

This parameter primarily controls how many tokens are processed in parallel during prompt ingestion (when you feed the initial long text to the model). It does not typically limit the generation length.

When you feed a long prompt, llama.cpp tries to process it in batches of this size.

HIGHLY LIKELY CULPRIT: Setting nBatch equal to nCtx (especially when both are large like 8192) means llama.cpp will attempt to allocate a temporary buffer large enough to hold data for processing up to 8192 tokens simultaneously during the initial prompt feed. This allocation can be huge and is often much larger than the KV cache itself for a short period. This is a very common reason for crashes when processing long initial prompts.

nPredict = -1:

This tells llama.cpp to generate tokens indefinitely until either the context window (nCtx) is full or an End-of-Sequence (EOS) token is produced.

This setting itself is usually fine and shouldn't directly cause a crash unless the context gets full due to a large nCtx and memory runs out during generation (less likely than the nBatch issue during prompt processing).

SamplerParams:

These parameters (temp, topK, topP, penaltyRepeat) influence which token is chosen at each step of generation.

They generally affect the quality and style of the output, not the core memory allocation or processing stability. It's very unlikely these settings are causing a crash related to text length.

Recommendations:

Drastically Reduce nBatch: This is the first thing you should try. Set nBatch to a much smaller, more standard value. A common default is 512.

ContextParams contextParams =
    ContextParams()
      ..nPredict = -1
      ..nCtx = 8192 // Keep this for now, if your model/system supports it
      ..nBatch = 512; // SIGNIFICANTLY REDUCE THIS


Why? This reduces the peak memory required during the processing of your initial long prompt. The prompt will be processed in smaller chunks (e.g., 512 tokens at a time), which is much less memory-intensive. It might take slightly longer to ingest the prompt, but it's far less likely to crash.

Verify nCtx Support and Resources:

Confirm that the specific model file you're using is designed for or fine-tuned for an 8192 context. Using a model trained for 2048/4096 with nCtx=8192 might work but can lead to degraded performance or unexpected behavior.

Monitor your system's RAM (and VRAM if using GPU offload) consumption when you run your Dart code with the long text. Does it spike massively and then crash? This confirms a memory issue. If 8192 is too much for your system even with a smaller nBatch, you'll need to reduce nCtx as well (e.g., to 4096 or 2048).

Check FFI Bindings: Ensure your Dart struct definitions (ContextParams, etc.) perfectly match the C struct definitions in the llama.h file of the exact version of llama.cpp you are linking against. Mismatches in field order, types, or padding can cause crashes, especially when dealing with larger data.

Error Checking: Make sure your Dart code is checking the return values of the llama.cpp functions (like llama_decode). They often return non-zero values on error, which might give you more clues than a hard crash.

Isolate the Crash: Add logging before and after key llama.cpp calls (e.g., llama_new_context_with_model, llama_tokenize, llama_decode) to pinpoint exactly which native function call is triggering the crash. The crash likely happens during the first llama_decode call that processes the prompt tokens if nBatch is the issue.

Start by reducing nBatch significantly. That's the most probable fix for crashes specifically occurring when inputting long text.
""";
void main() async {
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/gemma-3-12b-it-q4_0.gguf";

    final modelParams = ModelParams();

    final contextParams = ContextParams()
      ..nPredict = 20
      ..nCtx = 1024 * 4
      ..nBatch = 1024 * 4;

    final samplerParams = SamplerParams()
      ..temp = 1.0
      ..topK = 64
      ..topP = 0.95
      ..penaltyRepeat = 1.1;

    Llama llama =
        Llama(modelPath, modelParams, contextParams, samplerParams, true);

    llama.setPrompt(
        "<start_of_turn>please shorten the following text\n $longText<end_of_turn>\n<start_of_turn>model\n");
    while (true) {
      var (token, done) = llama.getNext();
      stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");

    llama.clear();

    llama.setPrompt(
        "<start_of_turn>please shorten the following text\n $longText<end_of_turn>\n<start_of_turn>model\n");
    while (true) {
      var (token, done) = llama.getNext();
      stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");

    llama.dispose();
  } catch (e) {
    print("Error: ${e.toString()}");
  }
}
