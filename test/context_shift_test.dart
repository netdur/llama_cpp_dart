/// Tests for context-shift API.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart test test/context_shift_test.dart
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/context/context.dart' as low;
import 'package:llama_cpp_dart/src/ffi/library_loader.dart' as low;
import 'package:llama_cpp_dart/src/model/model.dart' as low;
import 'package:llama_cpp_dart/src/session/session.dart' as low;
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

  group('LlamaContext.canShift', () {
    test('reflects llama_memory_can_shift on the loaded context', () {
      low.LlamaLibrary.load(path: libPath);
      final model =
          low.LlamaModel.load(ModelParams(path: modelPath, gpuLayers: 0));
      addTearDown(model.dispose);
      final ctx = low.LlamaContext.create(
        model,
        const ContextParams(nCtx: 256, nBatch: 64, nUbatch: 64),
      );
      addTearDown(ctx.dispose);

      // The boolean is whatever the backend reports — we just assert it's
      // observable and consistent across calls.
      final first = ctx.canShift;
      final second = ctx.canShift;
      expect(first, second);
    });
  });

  group('LlamaSession.shiftContext', () {
    late low.LlamaModel model;
    late low.LlamaContext context;

    setUpAll(() {
      low.LlamaLibrary.load(path: libPath);
      model = low.LlamaModel.load(ModelParams(path: modelPath, gpuLayers: 0));
      context = low.LlamaContext.create(
        model,
        const ContextParams(nCtx: 256, nBatch: 64, nUbatch: 64),
      );
    });

    tearDownAll(() {
      context.dispose();
      model.dispose();
    });

    test('drops the requested middle window', () async {
      if (!context.canShift) {
        markTestSkipped('this context does not support shifting');
        return;
      }

      final session = low.LlamaSession(context);
      addTearDown(session.clear);

      // Prefill ~30 tokens so we have room for both n_keep and n_discard.
      session.appendText(
        'one two three four five six seven eight nine ten '
        'eleven twelve thirteen fourteen fifteen sixteen seventeen',
        addSpecial: true,
      );
      var emitted = 0;
      await for (final ev in session.generate(
        sampler: SamplerParams.greedyDefault,
        maxTokens: 1,
      )) {
        if (ev is TokenEvent) emitted++;
      }
      expect(emitted, 1);

      final beforeKv = session.kvHead;
      final beforeTokens = session.tokenCount;
      expect(beforeKv, greaterThan(8));

      final discarded = session.shiftContext(nKeep: 4, nDiscard: 4);
      expect(discarded, 4);
      expect(session.kvHead, beforeKv - 4);
      expect(session.tokenCount, beforeTokens - 4);
    });

    test('is a no-op when nLeft is zero', () {
      if (!context.canShift) {
        markTestSkipped('this context does not support shifting');
        return;
      }
      final session = low.LlamaSession(context);
      // No tokens; n_keep > n_past → n_left becomes negative → return 0.
      expect(session.shiftContext(nKeep: 100), 0);
    });
  });
}
