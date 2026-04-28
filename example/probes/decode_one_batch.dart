/// Tokenize a prompt, push it through `llama_decode` once, and report the
/// result. This is the smallest end-to-end exercise of model + context +
/// batch + tokenizer — generation comes in M2.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart run example/probes/decode_one_batch.dart "Hello"
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main(List<String> args) {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  if (libPath == null || modelPath == null) {
    stderr.writeln(
      'set LLAMA_CPP_DART_LIB and LLAMA_CPP_DART_MODEL before running',
    );
    exit(2);
  }

  final prompt = args.isEmpty ? 'Hello' : args.join(' ');

  LlamaLibrary.load(path: libPath);

  final model = LlamaModel.load(ModelParams(path: modelPath, gpuLayers: 99));
  final ctx = LlamaContext.create(
    model,
    const ContextParams(nCtx: 512, nBatch: 256, nUbatch: 256),
  );
  final tokenizer = Tokenizer(model.vocab);
  final batch = LlamaBatch(256);

  final tokens = tokenizer.encode(prompt);
  if (tokens.length > batch.capacity) {
    stderr.writeln(
      'prompt is ${tokens.length} tokens; batch capacity is ${batch.capacity}',
    );
    exit(2);
  }

  for (var i = 0; i < tokens.length; i++) {
    batch.add(
      tokens[i],
      i,
      const [0],
      wantLogits: i == tokens.length - 1,
    );
  }

  final swatch = Stopwatch()..start();
  final rc = LlamaLibrary.bindings.llama_decode(ctx.pointer, batch.raw);
  swatch.stop();

  stdout.writeln('prompt        : $prompt');
  stdout.writeln('tokens        : ${tokens.length}');
  stdout.writeln('llama_decode  : rc=$rc');
  stdout.writeln('decode time   : ${swatch.elapsedMilliseconds} ms');
  stdout.writeln('context n_ctx : ${ctx.nCtx}');

  batch.dispose();
  ctx.dispose();
  model.dispose();
  LlamaLibrary.dispose();
}
