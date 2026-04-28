/// Stream tokens from a prompt. Demonstrates the M2 generation pipeline.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart run example/probes/generate_stream.dart "Once upon a time"
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main(List<String> args) async {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  if (libPath == null || modelPath == null) {
    stderr.writeln(
      'set LLAMA_CPP_DART_LIB and LLAMA_CPP_DART_MODEL before running',
    );
    exit(2);
  }

  final prompt = args.isEmpty ? 'Once upon a time' : args.join(' ');
  final maxTokens =
      int.tryParse(Platform.environment['MAX_TOKENS'] ?? '') ?? 128;

  LlamaLibrary.load(path: libPath);
  LlamaLog.silence();

  final model = LlamaModel.load(ModelParams(path: modelPath, gpuLayers: 99));
  final context = LlamaContext.create(
    model,
    const ContextParams(nCtx: 2048, nBatch: 512, nUbatch: 512),
  );
  final session = LlamaSession(context);

  session.appendText(prompt, addSpecial: true);

  stdout.write(prompt);

  final swatch = Stopwatch()..start();
  var generated = 0;

  await for (final event in session.generate(
    sampler: const SamplerParams(temperature: 0.7, topP: 0.9),
    maxTokens: maxTokens,
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
        stdout.writeln('---');
        stdout.writeln(
          'reason=${event.reason} '
          'generated=${event.generatedCount} '
          'committedPos=${event.committedPosition}',
        );
        final secs = swatch.elapsedMilliseconds / 1000.0;
        final tps = secs > 0 ? generated / secs : 0;
        stdout.writeln(
          'elapsed=${swatch.elapsedMilliseconds} ms  '
          '${tps.toStringAsFixed(1)} tok/s  '
          'kv_head=${session.kvHead}',
        );
    }
  }

  context.dispose();
  model.dispose();
  LlamaLibrary.dispose();
}
