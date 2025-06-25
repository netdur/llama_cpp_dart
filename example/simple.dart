import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:llama_cpp_dart/src/llama_cpp.dart';

void main() {
  // Initialize settings
  String modelPath = "/Users/adel/Workspace/gguf/gemma-3-12b-it-q4_0.gguf";
  String prompt = "<start_of_turn>2 * 4 =<end_of_turn>\n<start_of_turn>model\n";

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
