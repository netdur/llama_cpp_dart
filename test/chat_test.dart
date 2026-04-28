/// Integration test for M4 chat template + EngineChat.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart test test/chat_test.dart
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
      contextParams: const ContextParams(nCtx: 2048, nBatch: 512, nUbatch: 512),
    );
  });

  tearDownAll(() async {
    await engine.dispose();
  });

  group('EngineChat', () {
    test('exposes the model\'s chat template (or null)', () {
      // Just assert the call works; either embedded or null is acceptable.
      expect(engine.modelChatTemplate, anyOf(isNull, isA<String>()));
    });

    test('renders a system+user history and produces an assistant reply',
        () async {
      if (engine.modelChatTemplate == null) {
        markTestSkipped('model has no embedded chat template');
        return;
      }

      final chat = await engine.createChat();
      addTearDown(chat.dispose);

      chat.addSystem('You are a brief assistant.');
      chat.addUser('Say only the word "ok".');

      final tokens = <TokenEvent>[];
      DoneEvent? done;
      await for (final event in chat.generate(
        sampler: SamplerParams.greedyDefault,
        maxTokens: 16,
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
      expect(chat.messageCount, 3,
          reason: 'system + user + assistant after one turn');
      expect(chat.messages.last.role, 'assistant');
      expect(chat.messages.last.content, isNotEmpty);
    });

    test('multi-turn conversation appends each assistant reply', () async {
      if (engine.modelChatTemplate == null) {
        markTestSkipped('model has no embedded chat template');
        return;
      }

      final chat = await engine.createChat();
      addTearDown(chat.dispose);

      chat.addSystem('Be concise.');

      Future<void> runTurn(String userMsg) async {
        chat.addUser(userMsg);
        await for (final _ in chat.generate(
          sampler: SamplerParams.greedyDefault,
          maxTokens: 12,
        )) {/* drain */}
      }

      await runTurn('Reply with the word "one".');
      await runTurn('Now the word "two".');

      // 1 system + 2 user + 2 assistant = 5
      expect(chat.messageCount, 5);
      expect(chat.messages[0].role, 'system');
      expect(chat.messages[1].role, 'user');
      expect(chat.messages[2].role, 'assistant');
      expect(chat.messages[3].role, 'user');
      expect(chat.messages[4].role, 'assistant');
      expect(chat.messages[2].content, isNotEmpty);
      expect(chat.messages[4].content, isNotEmpty);
    });

    test('templateOverride lets callers supply a Jinja template', () async {
      // Use a simple ChatML template directly. Works regardless of what
      // the model has embedded — llama.cpp pattern-matches to ChatML.
      const chatml =
          '''{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}''';

      final chat = await engine.createChat();
      addTearDown(chat.dispose);

      chat.addSystem('Be brief.');
      chat.addUser('Hi.');

      var emitted = 0;
      await for (final event in chat.generate(
        sampler: SamplerParams.greedyDefault,
        maxTokens: 4,
        templateOverride: chatml,
      )) {
        if (event is TokenEvent) emitted++;
      }
      expect(emitted, greaterThan(0));
    });
  });

  group('ChatTemplate', () {
    test('builtinNames returns a non-empty list', () {
      // Engine setup loaded the library; ChatTemplate.builtinNames runs in
      // the main isolate and uses the same bindings cached there. But the
      // main isolate hasn't called LlamaLibrary.load directly, so we skip
      // the check unless that path is exercised. ChatTemplate.apply is
      // exercised through the worker via the chat tests above.
    }, skip: 'main isolate has no LlamaLibrary.load; covered via worker');
  });
}
