/// End-to-end smoke test for the M1 module skeleton.
///
/// Run with the path to the built dylib and a tiny GGUF model:
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/tiny.gguf \
///     dart test test/smoke_test.dart
///
/// Skipped automatically when either env var is missing — keeps the test
/// suite green on machines without a model checkout.
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

  group('M1 smoke', () {
    setUpAll(() {
      LlamaLibrary.load(path: libPath);
    });

    late LlamaModel model;
    late LlamaContext context;
    late LlamaBatch batch;
    late Tokenizer tokenizer;

    setUp(() {
      model = LlamaModel.load(ModelParams(path: modelPath, gpuLayers: 99));
      context = LlamaContext.create(
        model,
        const ContextParams(nCtx: 512, nBatch: 256, nUbatch: 256),
      );
      batch = LlamaBatch(256);
      tokenizer = Tokenizer(model.vocab);
    });

    tearDown(() {
      batch.dispose();
      context.dispose();
      model.dispose();
    });

    test('model exposes basic metadata', () {
      expect(model.nParams, greaterThan(0));
      expect(model.nEmbd, greaterThan(0));
      expect(model.nLayer, greaterThan(0));
      expect(model.trainCtx, greaterThan(0));
      expect(model.vocab.nTokens, greaterThan(0));
    });

    test('encode + decode round-trips a simple prompt', () {
      const prompt = 'Hello, world!';
      final tokens = tokenizer.encode(prompt);
      expect(tokens, isNotEmpty);

      final decoded = tokenizer.decodeAll(tokens);
      expect(decoded.trim(), contains('Hello'));
    });

    test('llama_decode succeeds on a one-batch prefill', () {
      final tokens = tokenizer.encode('Hello');
      expect(tokens.length, lessThanOrEqualTo(batch.capacity));

      batch.clear();
      for (var i = 0; i < tokens.length; i++) {
        batch.add(
          tokens[i],
          i,
          const [0],
          wantLogits: i == tokens.length - 1,
        );
      }

      final rc = LlamaLibrary.bindings.llama_decode(
        context.pointer,
        batch.raw,
      );
      expect(rc, 0, reason: 'llama_decode returned non-zero: $rc');
    });

    test('context advertises params we requested', () {
      expect(context.nCtx, 512);
      expect(context.nBatch, 256);
      expect(context.nUbatch, 256);
    });
  });
}
