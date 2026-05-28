/// Integration test for the synchronous SpeculativeDecoder helper.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart test test/speculative_test.dart
///
/// Uses the same model as both target and draft. Identical models => the
/// draft predicts exactly what the target would, so acceptance is ~100% and
/// the output must be byte-identical to a plain greedy reference run.
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:test/test.dart';

void main() {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];

  if (libPath == null || libPath.isEmpty) {
    test('LLAMA_CPP_DART_LIB not set', () {}, skip: 'set LLAMA_CPP_DART_LIB');
    return;
  }
  if (modelPath == null || modelPath.isEmpty) {
    test('LLAMA_CPP_DART_MODEL not set', () {},
        skip: 'set LLAMA_CPP_DART_MODEL');
    return;
  }

  late LlamaModel model;

  setUpAll(() {
    LlamaLibrary.load(path: libPath);
    LlamaLog.silence();
    model = LlamaModel.load(ModelParams(path: modelPath, gpuLayers: 99));
  });

  tearDownAll(() {
    model.dispose();
    LlamaLibrary.dispose();
  });

  LlamaContext newCtx() => LlamaContext.create(
        model,
        const ContextParams(nCtx: 512, nBatch: 512, nUbatch: 512),
      );

  List<int> plainGreedy(String prompt, int maxTokens) {
    final ctx = newCtx();
    final lib = LlamaLibrary.bindings;
    final tok = Tokenizer(model.vocab);
    final promptTokens = tok.encode(prompt);
    final batch = LlamaBatch(512);
    final out = <int>[];
    try {
      batch.clear();
      for (var i = 0; i < promptTokens.length; i++) {
        batch.add(promptTokens[i], i, const [0],
            wantLogits: i == promptTokens.length - 1);
      }
      lib.llama_decode(ctx.pointer, batch.raw);
      var pos = promptTokens.length;
      var next = _argmax(ctx.logitsAt(promptTokens.length - 1)!);
      while (out.length < maxTokens) {
        out.add(next);
        if (model.vocab.isEog(next)) break;
        batch.clear();
        batch.add(next, pos, const [0], wantLogits: true);
        lib.llama_decode(ctx.pointer, batch.raw);
        next = _argmax(ctx.lastLogits()!);
        pos++;
      }
    } finally {
      batch.dispose();
      ctx.dispose();
    }
    return out;
  }

  group('SpeculativeDecoder', () {
    test('output matches plain greedy with same-model draft (full accept)',
        () {
      const prompt = 'The capital of France is';
      const maxTokens = 24;

      final reference = plainGreedy(prompt, maxTokens);

      final target = newCtx();
      final draft = newCtx();
      addTearDown(target.dispose);
      addTearDown(draft.dispose);

      final result = SpeculativeDecoder(target: target, draft: draft).generate(
        prompt: prompt,
        maxTokens: maxTokens,
        draftLength: 4,
      );

      expect(result.tokens, equals(reference));
      // Identical models agree on every draft → ~100% acceptance.
      expect(result.acceptanceRate, greaterThan(0.95));
      expect(result.draftedCount, greaterThan(0));
    });

    test('respects maxTokens', () {
      final target = newCtx();
      final draft = newCtx();
      addTearDown(target.dispose);
      addTearDown(draft.dispose);

      final result = SpeculativeDecoder(target: target, draft: draft).generate(
        prompt: 'Once upon a time',
        maxTokens: 7,
        draftLength: 4,
      );
      expect(result.tokens.length, lessThanOrEqualTo(7));
    });

    test('matches greedy across draftLength values', () {
      const prompt = 'Water boils at';
      const maxTokens = 16;
      final reference = plainGreedy(prompt, maxTokens);

      for (final k in [1, 2, 8]) {
        final target = newCtx();
        final draft = newCtx();
        final result =
            SpeculativeDecoder(target: target, draft: draft).generate(
          prompt: prompt,
          maxTokens: maxTokens,
          draftLength: k,
        );
        target.dispose();
        draft.dispose();
        expect(result.tokens, equals(reference),
            reason: 'draftLength=$k should not change the output');
      }
    });

    test('stochastic: same-model draft accepts everything (p == q)', () {
      final target = newCtx();
      final draft = newCtx();
      addTearDown(target.dispose);
      addTearDown(draft.dispose);

      final result = SpeculativeDecoder(target: target, draft: draft).generate(
        prompt: 'Once upon a time',
        maxTokens: 20,
        draftLength: 4,
        temperature: 0.8,
        seed: 123,
      );
      // Identical target/draft distributions => acceptance ratio is always 1.
      expect(result.acceptanceRate, closeTo(1.0, 1e-9));
      expect(result.tokens.length, lessThanOrEqualTo(20));
    });

    test('stochastic: reproducible with a fixed seed', () {
      List<int> run() {
        final t = newCtx();
        final d = newCtx();
        final res = SpeculativeDecoder(target: t, draft: d).generate(
          prompt: 'The weather today',
          maxTokens: 16,
          draftLength: 4,
          temperature: 0.9,
          seed: 777,
        );
        t.dispose();
        d.dispose();
        return res.tokens;
      }

      expect(run(), equals(run()));
    });
  });
}

int _argmax(List<double> v) {
  var best = 0;
  var bestVal = v[0];
  for (var i = 1; i < v.length; i++) {
    if (v[i] > bestVal) {
      bestVal = v[i];
      best = i;
    }
  }
  return best;
}
