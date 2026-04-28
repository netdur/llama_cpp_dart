/// Two-turn chat using the model's embedded Jinja template.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart run example/probes/chat_two_turn.dart
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  if (libPath == null || modelPath == null) {
    stderr.writeln(
      'set LLAMA_CPP_DART_LIB and LLAMA_CPP_DART_MODEL before running',
    );
    exit(2);
  }

  stdout.writeln('starting engine ...');
  final engine = await LlamaEngine.spawn(
    libraryPath: libPath,
    modelParams: ModelParams(path: modelPath, gpuLayers: 99),
    contextParams: const ContextParams(nCtx: 4096, nBatch: 512, nUbatch: 512),
  );
  stdout.writeln(
    'model chat template: ${engine.modelChatTemplate == null ? "(none)" : "embedded"}',
  );

  final chat = await engine.createChat();
  chat.addSystem('You are a concise assistant. Reply in one short sentence.');

  final turns = ['What is 2 + 2?', 'And 3 + 3?'];
  const sampler = SamplerParams(temperature: 0.2, topP: 0.9);

  for (final turn in turns) {
    chat.addUser(turn);
    stdout.writeln('\n>>> $turn');
    stdout.write('<<< ');

    final swatch = Stopwatch()..start();
    var generated = 0;
    await for (final event in chat.generate(
      sampler: sampler,
      maxTokens: 96,
    )) {
      switch (event) {
        case TokenEvent():
          stdout.write(event.text);
          generated++;
        case ShiftEvent():
          break;
        case DoneEvent():
          if (event.trailingText.isNotEmpty) stdout.write(event.trailingText);
          stdout.writeln();
          final secs = swatch.elapsedMilliseconds / 1000.0;
          final tps = secs > 0 ? generated / secs : 0;
          stdout.writeln(
            '    [${event.reason} '
            'gen=${event.generatedCount} '
            '${tps.toStringAsFixed(1)} tok/s]',
          );
      }
    }
  }

  stdout.writeln('\n--- final history (${chat.messageCount}) ---');
  for (final m in chat.messages) {
    stdout.writeln('${m.role}: ${m.content}');
  }

  await chat.dispose();
  await engine.dispose();
}
