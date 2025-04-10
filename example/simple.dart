import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:llama_cpp_dart/src/llama_cpp.dart';

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

void main() {
  // Initialize settings
  String modelPath = "/Users/adel/Workspace/gguf/gemma-3-12b-it-q4_0.gguf";
  String prompt =
      "<start_of_turn>please shorten the following text\n $longText<end_of_turn>\n<start_of_turn>model\n";
  int ngl = 99;
  int nPredict = 32;

  // Load library
  final lib = llama_cpp(DynamicLibrary.open(
      "/Users/adel/Workspace/llama_cpp_dart/bin/MAC_ARM64/libllama.dylib"));
  lib.llama_backend_init();

  // Initialize model
  var modelParams = lib.llama_model_default_params();
  modelParams.n_gpu_layers = ngl;

  final modelPathPtr = modelPath.toNativeUtf8().cast<Char>();
  final model = lib.llama_load_model_from_file(modelPathPtr, modelParams);
  malloc.free(modelPathPtr);

  if (model.address == 0) {
    stderr.writeln("error: unable to load model");
    return;
  }

  // Tokenize prompt
  final vocab = lib.llama_model_get_vocab(model);
  final promptPtr = prompt.toNativeUtf8().cast<Char>();
  final nPrompt = -lib.llama_tokenize(
      vocab, promptPtr, prompt.length, nullptr, 0, true, true);

  final tokens = malloc<llama_token>(nPrompt);
  if (lib.llama_tokenize(
          vocab, promptPtr, prompt.length, tokens, nPrompt, true, true) <
      0) {
    stderr.writeln("error: failed to tokenize the prompt");
    malloc.free(promptPtr);
    malloc.free(tokens);
    return;
  }
  malloc.free(promptPtr);

  // Initialize context
  var ctxParams = lib.llama_context_default_params();
  ctxParams.n_ctx = nPrompt + nPredict - 1;
  ctxParams.n_batch = nPrompt;
  ctxParams.no_perf = false;

  final ctx = lib.llama_new_context_with_model(model, ctxParams);
  if (ctx.address == 0) {
    stderr.writeln("error: failed to create context");
    malloc.free(tokens);
    return;
  }

  // Initialize sampler
  var sparams = lib.llama_sampler_chain_default_params();
  sparams.no_perf = false;
  final smpl = lib.llama_sampler_chain_init(sparams);
  lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_greedy());

  // Print prompt tokens
  for (int i = 0; i < nPrompt; i++) {
    final buf = malloc<Char>(128);
    int n = lib.llama_token_to_piece(vocab, tokens[i], buf, 128, 0, true);
    if (n < 0) {
      stderr.writeln("error: failed to convert token to piece");
      malloc.free(buf);
      malloc.free(tokens);
      return;
    }
    String piece = String.fromCharCodes(buf.cast<Uint8>().asTypedList(n));
    stdout.write(piece);
    malloc.free(buf);
  }

  // Track timing
  final tMainStart = DateTime.now().microsecondsSinceEpoch;

  // Prepare initial batch
  var batch = lib.llama_batch_get_one(tokens, nPrompt);

  // Main generation loop
  int nDecode = 0;
  int newTokenId;
  final tokenPtr = malloc<llama_token>();

  for (int nPos = 0; nPos + batch.n_tokens < nPrompt + nPredict;) {
    if (lib.llama_decode(ctx, batch) != 0) {
      stderr.writeln("failed to eval");
      malloc.free(tokenPtr);
      malloc.free(tokens);
      return;
    }

    nPos += batch.n_tokens;

    // Sample next token
    newTokenId = lib.llama_sampler_sample(smpl, ctx, -1);

    if (lib.llama_token_is_eog(vocab, newTokenId)) {
      break;
    }

    final buf = malloc<Char>(128);
    int n = lib.llama_token_to_piece(vocab, newTokenId, buf, 128, 0, true);
    if (n < 0) {
      stderr.writeln("error: failed to convert token to piece");
      malloc.free(buf);
      malloc.free(tokenPtr);
      malloc.free(tokens);
      return;
    }

    String piece = String.fromCharCodes(buf.cast<Uint8>().asTypedList(n));
    stdout.write(piece);
    // stdout.flush();
    malloc.free(buf);

    // Prepare next batch
    tokenPtr.value = newTokenId;
    batch = lib.llama_batch_get_one(tokenPtr, 1);

    nDecode++;
  }

  final tMainEnd = DateTime.now().microsecondsSinceEpoch;
  stdout.writeln();

  // Print performance statistics
  final decodeTime = (tMainEnd - tMainStart) / 1000000.0;
  stderr.writeln(
      "decoded $nDecode tokens in ${decodeTime.toStringAsFixed(2)} s, speed: ${(nDecode / decodeTime).toStringAsFixed(2)} t/s");
  stderr.writeln();

  // Print performance metrics
  lib.llama_perf_sampler_print(smpl);
  lib.llama_perf_context_print(ctx);
  stderr.writeln();

  // Cleanup
  malloc.free(tokenPtr);
  malloc.free(tokens);
  lib.llama_sampler_free(smpl);
  lib.llama_free(ctx);
  lib.llama_free_model(model);
  lib.llama_backend_free();
}
