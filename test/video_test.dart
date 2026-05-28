/// Integration test for video multimodal pipeline (SmolVLM2-256M-Video).
///
/// This test verifies that video input works correctly with the llama.cpp
/// official API specification.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/smolvlm2-256m-video.gguf \
///   LLAMA_CPP_DART_MMPROJ=/path/to/mmproj.gguf \
///     dart test test/video_test.dart
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

  // Use a test video from llama.cpp fixtures if available
  // For now, we check if any video file can be found
  final fixtureDir = '${Directory.current.path}/test/fixtures';
  late String fixtureVideo;
  if (Directory(fixtureDir).existsSync()) {
    final files = Directory(fixtureDir)
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.mp4') || f.path.endsWith('.mov'))
        .map((f) => f.path)
        .toList();
    if (files.isEmpty) {
      test('test video fixture not found', () {},
          skip: 'expected at least one .mp4 or .mov in $fixtureDir');
      return;
    }
    fixtureVideo = files.first;
  } else {
    test('test fixtures directory not found', () {},
        skip: 'expected $fixtureDir');
    return;
  }

  late LlamaEngine engine;

  setUpAll(() async {
    engine = await LlamaEngine.spawn(
      libraryPath: libPath,
      modelParams: ModelParams(path: modelPath, gpuLayers: 99),
      contextParams: const ContextParams(nCtx: 4096, nBatch: 512, nUbatch: 512),
      multimodalParams: MultimodalParams(
        mmprojPath: mmprojPath,
        videoMinTokens: 0,
        videoMaxTokens: 0,
      ),
    );
  });

  tearDownAll(() async {
    await engine.dispose();
  });

  group('LlamaEngine video multimodal', () {
    test('reports projector capability flags', () {
      expect(engine.multimodalLoaded, isTrue);
      expect(engine.supportsVision || engine.supportsAudio, isTrue);
    });

    test('EngineSession.generate with a video produces tokens', () async {
      final session = await engine.createSession();
      addTearDown(session.dispose);

      final tokens = <TokenEvent>[];
      DoneEvent? done;
      await for (final event in session.generate(
        prompt: 'What does this video show?\n<__media__>',
        addSpecial: true,
        sampler: SamplerParams.greedyDefault,
        maxTokens: 32,
        media: [LlamaMedia.videoFile(fixtureVideo)],
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

    test('EngineChat.addUser with video auto-prepends a media marker', () async {
      if (engine.modelChatTemplate == null) {
        markTestSkipped('model has no embedded chat template');
        return;
      }

      final chat = await engine.createChat();
      addTearDown(chat.dispose);

      chat.addUser(
        'Describe this video briefly.',
        media: [LlamaMedia.videoFile(fixtureVideo)],
      );
      // Verify that the message body now contains a marker.
      expect(chat.messages.last.content, contains('<__media__>'));

      final tokens = <TokenEvent>[];
      DoneEvent? done;
      await for (final event in chat.generate(
        sampler: SamplerParams.greedyDefault,
        maxTokens: 32,
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

    test('LlamaMedia.videoBytes from in-memory data works', () async {
      final bytes = File(fixtureVideo).readAsBytesSync();
      final session = await engine.createSession();
      addTearDown(session.dispose);

      var emitted = 0;
      await for (final event in session.generate(
        prompt: 'Describe: <__media__>',
        addSpecial: true,
        sampler: SamplerParams.greedyDefault,
        maxTokens: 12,
        media: [LlamaMedia.videoBytes(bytes, id: 'test-video')],
      )) {
        switch (event) {
          case TokenEvent():
            emitted++;
          case ShiftEvent():
            break;
          case DoneEvent():
            break;
        }
      }
      expect(emitted, greaterThan(0));
    });

    test('MultimodalParams.videoMinTokens and videoMaxTokens are respected',
        () async {
      // Verify video token parameters are passed through correctly
      final params = MultimodalParams(
        mmprojPath: mmprojPath,
        videoMinTokens: 512,
        videoMaxTokens: 2048,
      );
      expect(params.videoMinTokens, 512);
      expect(params.videoMaxTokens, 2048);

      // Verify copyWith works
      final updated = params.copyWith(videoMaxTokens: 4096);
      expect(updated.videoMaxTokens, 4096);
      expect(updated.videoMinTokens, 512);

      // Verify toJson/fromJson round-trip
      final json = params.toJson();
      expect(json['videoMinTokens'], 512);
      expect(json['videoMaxTokens'], 2048);

      final restored = MultimodalParams.fromJson(json);
      expect(restored.videoMinTokens, 512);
      expect(restored.videoMaxTokens, 2048);
    });

    test('MediaKind.video enum value exists', () {
      expect(MediaKind.video, isNotNull);
      expect(MediaKind.video.name, 'video');
    });

    test('LlamaMedia correctly identifies video kind', () {
      final bytes = File(fixtureVideo).readAsBytesSync();
      final media = LlamaMedia.videoBytes(bytes);
      expect(media.kind, MediaKind.video);
      expect(media.toString(), contains('video'));
    });
  });
}
