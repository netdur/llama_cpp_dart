/// Repro/verify for issue #105 / PR #106: cancelling a generation should stop
/// the native decode, not run it to maxTokens.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart run tool/probe_cancel.dart
///
/// Cancels a 400-token generation after 10 tokens, then times how long until a
/// SECOND generation's first token arrives. If cancel works, the worker stops
/// almost immediately and the 2nd gen starts in ~tens of ms. If it doesn't, the
/// worker keeps decoding the remaining ~390 tokens first (seconds).
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

Future<void> main() async {
  final lib = Platform.environment['LLAMA_CPP_DART_LIB'];
  final model = Platform.environment['LLAMA_CPP_DART_MODEL'];
  if (lib == null || model == null) {
    stderr.writeln('set LLAMA_CPP_DART_LIB and LLAMA_CPP_DART_MODEL');
    exit(2);
  }

  final engine = await LlamaEngine.spawn(
    libraryPath: lib,
    modelParams: ModelParams(path: model, gpuLayers: 99),
    contextParams: const ContextParams(nCtx: 1024, nBatch: 512, nUbatch: 512),
  );
  final session = await engine.createSession();

  const cancelAt = 10;
  const firstMax = 400;

  // First generation: cancel by breaking the stream after `cancelAt` tokens.
  var count = 0;
  final sw = Stopwatch();
  await for (final ev in session.generate(
    prompt: 'Tell me a very long, detailed story about the ocean.',
    addSpecial: true,
    sampler: SamplerParams.greedyDefault,
    maxTokens: firstMax,
  )) {
    if (ev is TokenEvent) {
      count++;
      if (count >= cancelAt) {
        sw.start(); // clock starts the instant we cancel
        break;
      }
    }
  }
  stdout.writeln('cancelled 1st gen after $count tokens (asked for $firstMax)');

  // Measure how long until the engine is free to run a fresh generation.
  // While the cancelled gen1 is still "in flight" the engine rejects new
  // generates; with the bug that lasts until the worker decodes to maxTokens
  // (seconds), with the fix it clears almost immediately. Retry until accepted.
  var firstTokenMs = -1;
  while (true) {
    try {
      await for (final ev in session.generate(
        prompt: 'Say hello.',
        addSpecial: true,
        sampler: SamplerParams.greedyDefault,
        maxTokens: 5,
      )) {
        if (ev is TokenEvent) {
          firstTokenMs = sw.elapsedMilliseconds;
          break;
        }
      }
      break;
    } on LlamaLibraryException catch (e) {
      if (e.toString().contains('already in flight')) {
        await Future<void>.delayed(const Duration(milliseconds: 5));
        continue; // engine still busy with the cancelled gen1
      }
      rethrow;
    }
  }

  // stderr is unbuffered, so the verdict survives even if native teardown
  // aborts at process exit (a separate ggml/Metal finalizer quirk).
  stderr.writeln('RESULT cancel -> 2nd-gen first token: $firstTokenMs ms');
  stderr.writeln(firstTokenMs >= 0 && firstTokenMs < 500
      ? 'RESULT PASS: cancel interrupted the decode'
      : 'RESULT FAIL: worker kept decoding before the 2nd gen could start');

  await session.dispose();
  await engine.dispose();
}
