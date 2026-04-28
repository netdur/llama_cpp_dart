/// Stream tokens from a worker isolate via [LlamaEngine].
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart run example/probes/engine_stream.dart "Once upon a time"
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
      int.tryParse(Platform.environment['MAX_TOKENS'] ?? '') ?? 64;

  stdout.writeln('starting engine ...');
  final engine = await LlamaEngine.spawn(
    libraryPath: libPath,
    modelParams: ModelParams(path: modelPath, gpuLayers: 99),
    contextParams: const ContextParams(nCtx: 2048, nBatch: 512, nUbatch: 512),
  );

  final session = await engine.createSession();
  stdout.write(prompt);

  final swatch = Stopwatch()..start();
  var generated = 0;

  await for (final event in session.generate(
    prompt: prompt,
    addSpecial: true,
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
        final secs = swatch.elapsedMilliseconds / 1000.0;
        final tps = secs > 0 ? generated / secs : 0;
        stdout.writeln(
          'reason=${event.reason}  '
          'generated=${event.generatedCount}  '
          'committedPos=${event.committedPosition}  '
          '${tps.toStringAsFixed(1)} tok/s',
        );
    }
  }

  await session.dispose();
  await engine.dispose();
}
