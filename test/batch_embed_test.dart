/// Integration test for the synchronous BatchEmbedder helper.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart test test/batch_embed_test.dart
///
/// The model does not need to be a dedicated embedding model — pooling is
/// forced to mean and we only assert the batched-decode plumbing and
/// per-sequence routing, which hold for any model.
library;

import 'dart:io';
import 'dart:math' as math;

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

  LlamaContext newEmbedContext({int nSeqMax = 4}) => LlamaContext.create(
        model,
        ContextParams(
          nCtx: 512,
          nBatch: 512,
          nUbatch: 512,
          nSeqMax: nSeqMax,
          embeddings: true,
          poolingType: PoolingType.mean,
        ),
      );

  double l2(List<double> v) =>
      math.sqrt(v.fold<double>(0, (s, x) => s + x * x));

  double cosine(List<double> a, List<double> b) {
    var dot = 0.0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot / (l2(a) * l2(b));
  }

  group('BatchEmbedder', () {
    test('embeds multiple texts in one pass, shape + normalization', () {
      final ctx = newEmbedContext();
      addTearDown(ctx.dispose);

      final results = BatchEmbedder(ctx).embed([
        'The cat sat on the mat.',
        'A feline rested on the rug.',
        'Quarterly revenue exceeded expectations.',
      ]);

      expect(results, hasLength(3));
      for (final r in results) {
        expect(r.nEmbd, model.nEmbd);
        expect(r.pooled, isTrue);
        expect(r.normalized, isTrue);
        expect(r.vector, hasLength(model.nEmbd));
        expect(l2(r.vector), closeTo(1.0, 1e-3));
        expect(r.nTokens, greaterThan(0));
      }
    });

    test('per-sequence routing: identical inputs give identical vectors', () {
      final ctx = newEmbedContext();
      addTearDown(ctx.dispose);

      // Same text in two slots must produce the same vector — proves the
      // per-seq embedding reads are not crossing wires.
      final results = BatchEmbedder(ctx).embed(['hello world', 'hello world']);
      expect(results, hasLength(2));
      expect(cosine(results[0].vector, results[1].vector), closeTo(1.0, 1e-4));
    });

    test('distinct inputs yield distinct vectors', () {
      final ctx = newEmbedContext();
      addTearDown(ctx.dispose);

      final results =
          BatchEmbedder(ctx).embed(['apples and oranges', 'tax law statute']);
      expect(cosine(results[0].vector, results[1].vector), lessThan(0.9999));
    });

    test('empty input returns empty list', () {
      final ctx = newEmbedContext();
      addTearDown(ctx.dispose);
      expect(BatchEmbedder(ctx).embed(const []), isEmpty);
    });

    test('throws when context was not built for embeddings', () {
      final ctx = LlamaContext.create(
        model,
        const ContextParams(nCtx: 512, nBatch: 512, nUbatch: 512),
      );
      addTearDown(ctx.dispose);
      expect(
        () => BatchEmbedder(ctx).embed(['x']),
        throwsA(isA<LlamaContextException>()),
      );
    });

    test('throws when texts exceed nSeqMax', () {
      final ctx = newEmbedContext(nSeqMax: 2);
      addTearDown(ctx.dispose);
      expect(
        () => BatchEmbedder(ctx).embed(['a', 'b', 'c']),
        throwsA(isA<LlamaContextException>()),
      );
    });
  });
}
