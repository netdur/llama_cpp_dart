/// Speculative decoding: a draft context proposes tokens, the target verifies
/// them in one pass. Output is identical to plain decoding on the target —
/// speculation only changes how many target forward passes it takes.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/target.gguf \
///     dart run example/probes/speculative_generate.dart "The capital of France is"
///
/// Optional:
///   LLAMA_CPP_DART_DRAFT_MODEL=/path/to/small-draft.gguf  (must share the
///       target's vocab). If unset, the target model is used as its own draft
///       — every proposal is accepted (~100%), which is handy for sanity
///       checks but gives no speedup.
///   MAX_TOKENS=64  DRAFT_LEN=4  TEMPERATURE=0.0  SEED=42
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
  final draftModelPath = Platform.environment['LLAMA_CPP_DART_DRAFT_MODEL'];

  final prompt = args.isEmpty ? 'The capital of France is' : args.join(' ');
  final maxTokens =
      int.tryParse(Platform.environment['MAX_TOKENS'] ?? '') ?? 64;
  final draftLen = int.tryParse(Platform.environment['DRAFT_LEN'] ?? '') ?? 4;
  final temperature =
      double.tryParse(Platform.environment['TEMPERATURE'] ?? '') ?? 0.0;
  final seed = int.tryParse(Platform.environment['SEED'] ?? '') ?? 42;

  LlamaLibrary.load(path: libPath);
  LlamaLog.silence();

  const ctxParams = ContextParams(nCtx: 2048, nBatch: 512, nUbatch: 512);

  // Target model + context.
  final targetModel = LlamaModel.load(
    ModelParams(path: modelPath, gpuLayers: 99),
  );
  final target = LlamaContext.create(targetModel, ctxParams);

  // Draft: a separate small model if given, otherwise the target itself.
  final LlamaModel draftModel;
  final bool sharedModel;
  if (draftModelPath != null && draftModelPath.isNotEmpty) {
    draftModel = LlamaModel.load(
      ModelParams(path: draftModelPath, gpuLayers: 99),
    );
    sharedModel = false;
  } else {
    draftModel = targetModel;
    sharedModel = true;
  }
  final draft = LlamaContext.create(draftModel, ctxParams);

  stdout.writeln('prompt      : $prompt');
  stdout.writeln('target      : $modelPath');
  stdout.writeln(
    'draft       : ${sharedModel ? "(same as target — expect ~100% accept, no speedup)" : draftModelPath}',
  );
  stdout.writeln(
    'draftLen=$draftLen  maxTokens=$maxTokens  '
    'temperature=$temperature${temperature > 0 ? "  seed=$seed" : " (greedy)"}',
  );
  stdout.writeln('---');

  final decoder = SpeculativeDecoder(target: target, draft: draft);
  final sw = Stopwatch()..start();
  final result = decoder.generate(
    prompt: prompt,
    maxTokens: maxTokens,
    draftLength: draftLen,
    temperature: temperature,
    seed: seed,
  );
  sw.stop();

  final secs = sw.elapsedMilliseconds / 1000.0;
  final tps = secs > 0 ? result.tokens.length / secs : 0;

  stdout.writeln(result.text.trim());
  stdout.writeln('---');
  stdout.writeln(
    'generated   : ${result.tokens.length} tokens  '
    '(${tps.toStringAsFixed(1)} tok/s, ${sw.elapsedMilliseconds} ms)',
  );
  stdout.writeln(
    'acceptance  : ${(result.acceptanceRate * 100).toStringAsFixed(1)}% '
    '(${result.acceptedCount}/${result.draftedCount} drafts accepted)',
  );
  stdout.writeln('stoppedOnEog: ${result.stoppedOnEog}');

  target.dispose();
  draft.dispose();
  if (!sharedModel) draftModel.dispose();
  targetModel.dispose();
  LlamaLibrary.dispose();
}
