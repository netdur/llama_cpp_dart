/// Integration test for M2 generation pipeline.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart test test/generation_test.dart
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
  late LlamaContext context;

  setUpAll(() {
    LlamaLibrary.load(path: libPath);
    LlamaLog.silence();
    model = LlamaModel.load(ModelParams(path: modelPath, gpuLayers: 99));
    context = LlamaContext.create(
      model,
      const ContextParams(nCtx: 1024, nBatch: 512, nUbatch: 512),
    );
  });

  tearDownAll(() {
    context.dispose();
    model.dispose();
    LlamaLibrary.dispose();
  });

  group('Generator', () {
    test('emits tokens until maxTokens then DoneEvent(StopMaxTokens)',
        () async {
      final tokenizer = Tokenizer(model.vocab);
      final generator = Generator(context, tokenizer);

      final request = Request(
        promptTokens: tokenizer.encode('Hello'),
        sampler: SamplerParams.greedyDefault,
        maxTokens: 8,
      );

      final tokenEvents = <TokenEvent>[];
      DoneEvent? done;

      await for (final event in generator.run(request)) {
        switch (event) {
          case TokenEvent():
            tokenEvents.add(event);
          case ShiftEvent():
            break;
          case DoneEvent():
            done = event;
        }
      }

      expect(done, isNotNull);
      expect(done!.reason, isA<StopMaxTokens>());
      expect(tokenEvents, hasLength(8));
      expect(done.generatedCount, 8);

      generator.dispose();
      // Reset the sequence so the next test starts from a clean KV.
      LlamaLibrary.bindings.llama_memory_seq_rm(
        LlamaLibrary.bindings.llama_get_memory(context.pointer),
        0,
        -1,
        -1,
      );
    });

    test('greedy sampling is deterministic across runs', () async {
      final tokenizer = Tokenizer(model.vocab);

      Future<List<int>> runOnce() async {
        // Each run uses its own seq id to isolate KV.
        final generator = Generator(context, tokenizer);
        final ids = <int>[];
        final request = Request(
          promptTokens: tokenizer.encode('The capital of France is'),
          sampler: SamplerParams.greedyDefault,
          maxTokens: 5,
          seqId: 0,
        );
        await for (final ev in generator.run(request)) {
          if (ev is TokenEvent) ids.add(ev.id);
        }
        generator.dispose();
        LlamaLibrary.bindings.llama_memory_seq_rm(
          LlamaLibrary.bindings.llama_get_memory(context.pointer),
          0,
          -1,
          -1,
        );
        return ids;
      }

      final first = await runOnce();
      final second = await runOnce();
      expect(first, equals(second));
      expect(first, isNotEmpty);
    });
  });

  group('LlamaSession', () {
    test('appendText + generate populates token history and KV head', () async {
      // Reuse seqId 0; previous tests cleared its KV state.
      final session = LlamaSession(context);
      session.appendText('Hello', addSpecial: true);

      final preTokens = session.tokenCount;
      expect(preTokens, greaterThan(0));
      expect(session.kvHead, 0);
      expect(session.hasPendingPrompt, isTrue);

      final emitted = <int>[];
      DoneEvent? done;
      await for (final ev in session.generate(
        sampler: SamplerParams.greedyDefault,
        maxTokens: 4,
      )) {
        switch (ev) {
          case TokenEvent():
            emitted.add(ev.id);
          case ShiftEvent():
            break;
          case DoneEvent():
            done = ev;
        }
      }

      expect(done, isNotNull);
      expect(emitted, hasLength(4));
      expect(session.tokenCount, preTokens + 4);
      // After stop on maxTokens, the last sampled token is in history but
      // not in KV cache. committedPosition counts everything except that
      // last token: prefilled prompt + (generatedCount - 1).
      expect(session.kvHead, preTokens + 3);
      expect(session.hasPendingPrompt, isTrue);

      session.clear();
      expect(session.tokenCount, 0);
      expect(session.kvHead, 0);
    });
  });
}
