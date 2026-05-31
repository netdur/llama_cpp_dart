/// MTP (NextN) speculative decoding probe.
///
/// Target context + a draft context built with [ContextType.mtp] off the same
/// model, driven through [MtpSpeculativeDecoder] (hidden-state passing +
/// checkpoint rollback, matching upstream llama.cpp PR #22673).
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/mtp-capable.gguf \
///     dart run tool/probe_mtp.dart "Write a short paragraph about the ocean."
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main(List<String> args) {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  if (libPath == null || modelPath == null) {
    stderr.writeln('set LLAMA_CPP_DART_LIB and LLAMA_CPP_DART_MODEL');
    exit(2);
  }

  final prompt = args.isEmpty
      ? 'Write a short paragraph about the ocean.'
      : args.join(' ');
  final maxTokens =
      int.tryParse(Platform.environment['MAX_TOKENS'] ?? '') ?? 256;
  final draftLen = int.tryParse(Platform.environment['DRAFT_LEN'] ?? '') ?? 4;
  final pMin = double.tryParse(Platform.environment['P_MIN'] ?? '') ?? 0.5;
  final quiet = Platform.environment['VERBOSE'] == null;

  LlamaLibrary.load(path: libPath);
  if (quiet) LlamaLog.silence();

  const targetParams = ContextParams(nCtx: 2048, nBatch: 512, nUbatch: 512);
  final mtpParams = targetParams.copyWith(ctxType: ContextType.mtp);

  final model = LlamaModel.load(ModelParams(path: modelPath, gpuLayers: 99));
  final target = LlamaContext.create(model, targetParams);
  final draft = LlamaContext.create(model, mtpParams);

  stdout.writeln('prompt    : $prompt');
  stdout.writeln('draft     : MTP/NextN heads (ContextType.mtp, same model)');
  stdout.writeln('draftLen=$draftLen maxTokens=$maxTokens pMin=$pMin');
  stdout.writeln('---');

  final decoder = MtpSpeculativeDecoder(target: target, draft: draft);
  final sw = Stopwatch()..start();
  final result = decoder.generate(
    prompt: prompt,
    maxTokens: maxTokens,
    draftLength: draftLen,
    pMin: pMin,
  );
  sw.stop();

  final secs = sw.elapsedMilliseconds / 1000.0;
  final tps = secs > 0 ? result.tokens.length / secs : 0;

  stdout.writeln(result.text.trim());
  stdout.writeln('---');
  stdout.writeln('generated  : ${result.tokens.length} tokens  '
      '(${tps.toStringAsFixed(1)} tok/s, ${sw.elapsedMilliseconds} ms)');
  stdout.writeln(
      'acceptance : ${(result.acceptanceRate * 100).toStringAsFixed(1)}% '
      '(${result.acceptedCount}/${result.draftedCount} drafts accepted)');
  stdout.writeln('stoppedOnEog: ${result.stoppedOnEog}');

  target.dispose();
  draft.dispose();
  model.dispose();
  LlamaLibrary.dispose();
}
