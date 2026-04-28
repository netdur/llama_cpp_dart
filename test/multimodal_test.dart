/// Integration test for M5 multimodal pipeline.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/llm.gguf \
///   LLAMA_CPP_DART_MMPROJ=/path/to/mmproj.gguf \
///     dart test test/multimodal_test.dart
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:test/test.dart';

void main() {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  final mmprojPath = Platform.environment['LLAMA_CPP_DART_MMPROJ'];

  if (libPath == null || libPath.isEmpty) {
    test('LLAMA_CPP_DART_LIB not set', () {}, skip: 'set LLAMA_CPP_DART_LIB');
    return;
  }
  if (modelPath == null || modelPath.isEmpty) {
    test('LLAMA_CPP_DART_MODEL not set', () {},
        skip: 'set LLAMA_CPP_DART_MODEL');
    return;
  }
  if (mmprojPath == null || mmprojPath.isEmpty) {
    test('LLAMA_CPP_DART_MMPROJ not set', () {},
        skip: 'set LLAMA_CPP_DART_MMPROJ');
    return;
  }

  // Use the test image bundled with mtmd. Skip the suite if not present.
  final fixtureImage =
      '${Directory.current.path}/src/llama.cpp/tools/mtmd/test-1.jpeg';
  if (!File(fixtureImage).existsSync()) {
    test('mtmd test fixture not present', () {},
        skip: 'expected $fixtureImage');
    return;
  }

  late LlamaEngine engine;

  setUpAll(() async {
    engine = await LlamaEngine.spawn(
      libraryPath: libPath,
      modelParams: ModelParams(path: modelPath, gpuLayers: 99),
      contextParams: const ContextParams(nCtx: 4096, nBatch: 512, nUbatch: 512),
      multimodalParams: MultimodalParams(mmprojPath: mmprojPath),
    );
  });

  tearDownAll(() async {
    await engine.dispose();
  });

  group('LlamaEngine multimodal', () {
    test('reports projector capability flags', () {
      expect(engine.multimodalLoaded, isTrue);
      expect(engine.supportsVision || engine.supportsAudio, isTrue);
    });

    test('EngineSession.generate with an image produces tokens', () async {
      if (!engine.supportsVision) {
        markTestSkipped('this projector does not support vision');
        return;
      }
      final session = await engine.createSession();
      addTearDown(session.dispose);

      final tokens = <TokenEvent>[];
      DoneEvent? done;
      await for (final event in session.generate(
        prompt: 'What does this image show?\n<__media__>',
        addSpecial: true,
        sampler: SamplerParams.greedyDefault,
        maxTokens: 24,
        media: [LlamaMedia.imageFile(fixtureImage)],
      )) {
        switch (event) {
          case TokenEvent():
            tokens.add(event);
          case ShiftEvent():
            break;
          case DoneEvent():
            done = event;
        }
      }
      expect(done, isNotNull);
      expect(tokens, isNotEmpty);
    });

    test('EngineChat.addUser auto-prepends a media marker', () async {
      if (!engine.supportsVision) {
        markTestSkipped('this projector does not support vision');
        return;
      }
      if (engine.modelChatTemplate == null) {
        markTestSkipped('model has no embedded chat template');
        return;
      }

      final chat = await engine.createChat();
      addTearDown(chat.dispose);

      chat.addUser(
        'Describe this image briefly.',
        media: [LlamaMedia.imageFile(fixtureImage)],
      );
      // Verify that the message body now contains a marker.
      expect(chat.messages.last.content, contains('<__media__>'));

      final tokens = <TokenEvent>[];
      DoneEvent? done;
      await for (final event in chat.generate(
        sampler: SamplerParams.greedyDefault,
        maxTokens: 24,
      )) {
        switch (event) {
          case TokenEvent():
            tokens.add(event);
          case ShiftEvent():
            break;
          case DoneEvent():
            done = event;
        }
      }
      expect(done, isNotNull);
      expect(tokens, isNotEmpty);
      expect(chat.messageCount, 2, reason: 'user + assistant after one turn');
      expect(chat.messages.last.role, 'assistant');
    });

    test('LlamaMedia.imageBytes path works from in-memory data', () async {
      if (!engine.supportsVision) {
        markTestSkipped('this projector does not support vision');
        return;
      }
      final bytes = File(fixtureImage).readAsBytesSync();
      final session = await engine.createSession();
      addTearDown(session.dispose);

      var emitted = 0;
      await for (final event in session.generate(
        prompt: 'Describe: <__media__>',
        addSpecial: true,
        sampler: SamplerParams.greedyDefault,
        maxTokens: 6,
        media: [LlamaMedia.imageBytes(bytes)],
      )) {
        if (event is TokenEvent) emitted++;
      }
      expect(emitted, greaterThan(0));
    });
  });
}
