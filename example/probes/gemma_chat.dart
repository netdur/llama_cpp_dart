/// Chat probe for Gemma-4-Unsloth (and similar) models whose embedded
/// Jinja template is too custom for `llama_chat_apply_template`.
///
/// Builds the prompt manually using the model's actual turn markers
/// (`<|turn>{role}\n{content}<turn|>`) and feeds it via
/// `EngineSession.generate(prompt:)`, bypassing chat-template rendering.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/Users/adel/Workspace/gguf/gemma-4-E2B-it-Q8_0.gguf \
///     dart run example/probes/gemma_chat.dart
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// Render messages in Gemma-4-Unsloth's `<|turn>{role}\n...<turn|>` style.
String renderGemma4(List<({String role, String content})> messages) {
  final buf = StringBuffer();
  for (final m in messages) {
    final role = m.role == 'assistant' ? 'model' : m.role;
    buf
      ..write('<|turn>')
      ..writeln(role)
      ..write(m.content)
      ..writeln('<turn|>');
  }
  buf
    ..write('<|turn>')
    ..writeln('model');
  return buf.toString();
}

void main() async {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB']!;
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL']!;

  stdout.writeln('starting engine ...');
  final engine = await LlamaEngine.spawn(
    libraryPath: libPath,
    modelParams: ModelParams(path: modelPath, gpuLayers: 99),
    contextParams: const ContextParams(nCtx: 2048, nBatch: 512, nUbatch: 512),
  );
  stdout.writeln('canShift=${engine.canShift}');

  final session = await engine.createSession();
  final messages = [
    (role: 'system', content: 'Answer in one short sentence.'),
    (role: 'user', content: 'What is 2 + 2?'),
  ];
  final prompt = renderGemma4(messages);
  stdout.writeln('--- prompt ---');
  stdout.write(prompt);
  stdout.writeln('--- reply ---');

  final swatch = Stopwatch()..start();
  var got = 0;
  await for (final ev in session.generate(
    prompt: prompt,
    addSpecial: true,
    parseSpecial: true,
    sampler: const SamplerParams(temperature: 0.3, topP: 0.9),
    maxTokens: 64,
  )) {
    switch (ev) {
      case TokenEvent():
        stdout.write(ev.text);
        got++;
      case ShiftEvent():
        break;
      case DoneEvent():
        if (ev.trailingText.isNotEmpty) stdout.write(ev.trailingText);
        stdout.writeln();
        final secs = swatch.elapsedMilliseconds / 1000.0;
        final tps = secs > 0 ? got / secs : 0;
        stdout.writeln(
          '--- ${ev.reason} gen=${ev.generatedCount} '
          '${tps.toStringAsFixed(1)} tok/s',
        );
    }
  }

  await session.dispose();
  await engine.dispose();
}
