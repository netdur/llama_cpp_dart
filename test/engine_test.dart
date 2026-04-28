/// Integration test for M3 LlamaEngine isolate worker.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart test test/engine_test.dart
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

  late LlamaEngine engine;

  setUpAll(() async {
    engine = await LlamaEngine.spawn(
      libraryPath: libPath,
      modelParams: ModelParams(path: modelPath, gpuLayers: 99),
      contextParams: const ContextParams(nCtx: 1024, nBatch: 512, nUbatch: 512),
    );
  });

  tearDownAll(() async {
    await engine.dispose();
  });

  group('LlamaEngine', () {
    test('greedy generate yields tokens and DoneEvent(StopMaxTokens)',
        () async {
      final session = await engine.createSession();
      addTearDown(session.dispose);

      final tokens = <TokenEvent>[];
      DoneEvent? done;

      await for (final ev in session.generate(
        prompt: 'Hello',
        addSpecial: true,
        sampler: SamplerParams.greedyDefault,
        maxTokens: 6,
      )) {
        switch (ev) {
          case TokenEvent():
            tokens.add(ev);
          case ShiftEvent():
            break;
          case DoneEvent():
            done = ev;
        }
      }

      expect(done, isNotNull);
      expect(done!.reason, isA<StopMaxTokens>());
      expect(tokens, hasLength(6));
    });

    test('cancellation stops the worker mid-generation', () async {
      final session = await engine.createSession();
      addTearDown(session.dispose);

      final stream = session.generate(
        prompt: 'Tell me a long story',
        addSpecial: true,
        sampler: SamplerParams.greedyDefault,
        maxTokens: 256,
      );

      var count = 0;
      await for (final ev in stream) {
        if (ev is TokenEvent) {
          count++;
          if (count >= 4) break; // breaking the stream cancels it
        }
      }

      expect(count, 4);
      // Wait a beat to give the cancel command time to round-trip and
      // confirm a follow-up generate works (i.e. the worker isn't stuck).
      await Future<void>.delayed(const Duration(milliseconds: 250));

      final tail = <int>[];
      await for (final ev in session.generate(
        prompt: ' continued',
        addSpecial: false,
        sampler: SamplerParams.greedyDefault,
        maxTokens: 3,
      )) {
        if (ev is TokenEvent) tail.add(ev.id);
      }
      expect(tail, hasLength(3));
    });

    test('two sessions on different seq ids run in series', () async {
      final engine2 = await LlamaEngine.spawn(
        libraryPath: libPath,
        modelParams: ModelParams(path: modelPath, gpuLayers: 99),
        contextParams: const ContextParams(
          nCtx: 1024,
          nBatch: 512,
          nUbatch: 512,
          nSeqMax: 2,
        ),
      );
      addTearDown(engine2.dispose);

      final s1 = await engine2.createSession(seqId: 0);
      final s2 = await engine2.createSession(seqId: 1);

      Future<List<int>> oneRun(EngineSession s) async {
        final ids = <int>[];
        await for (final ev in s.generate(
          prompt: 'Hello',
          addSpecial: true,
          sampler: SamplerParams.greedyDefault,
          maxTokens: 4,
        )) {
          if (ev is TokenEvent) ids.add(ev.id);
        }
        return ids;
      }

      final r1 = await oneRun(s1);
      final r2 = await oneRun(s2);
      expect(r1, equals(r2),
          reason: 'greedy on identical prompts should match across sessions');

      await s1.dispose();
      await s2.dispose();
    });
  });
}
