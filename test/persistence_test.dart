/// Integration test for M6 persistence pipeline.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart test test/persistence_test.dart
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/session/state_codec.dart';
import 'package:test/test.dart';

void main() {
  group('state codec (pure Dart)', () {
    StateMetadata mkMeta({
      int tokensCount = 4,
      int tokensChecksum = 0,
      int modelSizeBytes = 1234,
    }) =>
        StateMetadata(
          codecVersion: stateCodecVersion,
          savedAt: '2026-04-28T00:00:00Z',
          wrapperVersion: 'test',
          modelPath: '/path/to/model.gguf',
          modelSizeBytes: modelSizeBytes,
          modelNParams: 256,
          modelNEmbd: 64,
          modelNLayer: 4,
          modelTrainCtx: 4096,
          nCtx: 1024,
          nBatch: 256,
          nUbatch: 256,
          nSeqMax: 1,
          embeddings: false,
          mmprojPath: null,
          mmprojSizeBytes: null,
          mmprojSupportsVision: null,
          mmprojSupportsAudio: null,
          seqId: 0,
          kvHead: 12,
          tokensCount: tokensCount,
          tokensChecksum: tokensChecksum,
          extra: const {'note': 'hello'},
        );

    test('round-trips metadata + tokens + raw bytes', () {
      final tokens = [101, 202, 303, 404];
      final tokenBytes = Uint8List(tokens.length * 4)
        ..buffer.asInt32List().setAll(0, tokens);
      final raw = Uint8List.fromList(List.generate(64, (i) => i & 0xff));

      final metadata = mkMeta(
        tokensCount: tokens.length,
        tokensChecksum: fnv1a64(tokenBytes),
      );
      final encoded =
          encodeState(metadata: metadata, tokens: tokens, rawState: raw);
      expect(encoded, isNotEmpty);

      final decoded = decodeState(encoded);
      expect(decoded.tokens, tokens);
      expect(decoded.rawState, raw);
      expect(decoded.metadata.kvHead, 12);
      expect(decoded.metadata.extra['note'], 'hello');
    });

    test('badMagic on a non-state file', () {
      final junk = Uint8List.fromList(List.filled(32, 0));
      expect(
        () => decodeState(junk),
        throwsA(isA<LlamaStateException>().having(
          (e) => e.reason,
          'reason',
          LlamaStateError.badMagic,
        )),
      );
    });

    test('truncation triggers truncated error', () {
      final tokens = [1, 2, 3];
      final tokenBytes = Uint8List(tokens.length * 4)
        ..buffer.asInt32List().setAll(0, tokens);
      final encoded = encodeState(
        metadata: mkMeta(
          tokensCount: tokens.length,
          tokensChecksum: fnv1a64(tokenBytes),
        ),
        tokens: tokens,
        rawState: Uint8List(8),
      );
      final cut = Uint8List.fromList(encoded.sublist(0, encoded.length ~/ 2));
      expect(
        () => decodeState(cut),
        throwsA(isA<LlamaStateException>()),
      );
    });

    test('verifyCompatible flags model size mismatch', () {
      final saved = mkMeta(modelSizeBytes: 1000);
      final actual = mkMeta(modelSizeBytes: 2000);
      expect(
        () => verifyCompatible(saved, actual),
        throwsA(isA<LlamaStateException>().having(
          (e) => e.reason,
          'reason',
          LlamaStateError.modelMismatch,
        )),
      );
    });
  });

  // -------- integration tests below need a model + dylib --------
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
  late Directory tmp;

  setUpAll(() async {
    engine = await LlamaEngine.spawn(
      libraryPath: libPath,
      modelParams: ModelParams(path: modelPath, gpuLayers: 99),
      contextParams: const ContextParams(nCtx: 1024, nBatch: 256, nUbatch: 256),
    );
    tmp = Directory.systemTemp.createTempSync('llamacppdart_persist_');
  });

  tearDownAll(() async {
    await engine.dispose();
    if (tmp.existsSync()) tmp.deleteSync(recursive: true);
  });

  group('EngineSession.saveState / loadState', () {
    test('round-trips token history + KV', () async {
      final session = await engine.createSession();
      addTearDown(session.dispose);

      final path = '${tmp.path}/session_round_trip.lcdc';

      var emitted = 0;
      await for (final ev in session.generate(
        prompt: 'Hello',
        addSpecial: true,
        sampler: SamplerParams.greedyDefault,
        maxTokens: 4,
      )) {
        if (ev is TokenEvent) emitted++;
      }
      expect(emitted, 4);

      await session.saveState(path, extra: {'note': 'unit-test'});
      expect(File(path).existsSync(), isTrue);
      expect(File(path).statSync().size, greaterThan(64));

      final restored = await engine.createSession();
      addTearDown(restored.dispose);
      final extra = await restored.loadState(path);
      expect(extra['note'], 'unit-test');
    });
  });

  group('EngineChat.saveState / loadState', () {
    test('persists message history and lets us continue the chat', () async {
      if (engine.modelChatTemplate == null) {
        markTestSkipped('model has no embedded chat template');
        return;
      }

      final chat = await engine.createChat();
      addTearDown(chat.dispose);

      chat.addSystem('Be brief.');
      chat.addUser('Say "ok".');
      await for (final _ in chat.generate(
        sampler: SamplerParams.greedyDefault,
        maxTokens: 6,
      )) {/* drain */}
      expect(chat.messageCount, 3);

      final path = '${tmp.path}/chat_round_trip.lcdc';
      await chat.saveState(path);

      final resumed = await engine.createChat();
      addTearDown(resumed.dispose);
      await resumed.loadState(path);

      expect(resumed.messageCount, 3);
      expect(resumed.messages[0].role, 'system');
      expect(resumed.messages[1].role, 'user');
      expect(resumed.messages[2].role, 'assistant');

      resumed.addUser('Now say "yes".');
      await for (final _ in resumed.generate(
        sampler: SamplerParams.greedyDefault,
        maxTokens: 4,
      )) {/* drain */}
      expect(resumed.messageCount, 5);
    });

    test('rejects a state file written for a different model size', () async {
      final session = await engine.createSession();
      addTearDown(session.dispose);

      final path = '${tmp.path}/spoofed.lcdc';
      const tokens = <int>[];
      final tokenBytes = Uint8List(0);
      final metadata = StateMetadata(
        codecVersion: stateCodecVersion,
        savedAt: 'now',
        wrapperVersion: 'test',
        modelPath: '/wrong/model.gguf',
        modelSizeBytes: 99999999999,
        modelNParams: 99999,
        modelNEmbd: 1,
        modelNLayer: 1,
        modelTrainCtx: 1,
        nCtx: 1024,
        nBatch: 256,
        nUbatch: 256,
        nSeqMax: 1,
        embeddings: false,
        mmprojPath: null,
        mmprojSizeBytes: null,
        mmprojSupportsVision: null,
        mmprojSupportsAudio: null,
        seqId: 0,
        kvHead: 0,
        tokensCount: 0,
        tokensChecksum: fnv1a64(tokenBytes),
      );
      File(path).writeAsBytesSync(encodeState(
        metadata: metadata,
        tokens: tokens,
        rawState: Uint8List(0),
      ));

      var failed = false;
      try {
        await session.loadState(path);
      } catch (_) {
        failed = true;
      }
      expect(failed, isTrue);
    });
  });
}
