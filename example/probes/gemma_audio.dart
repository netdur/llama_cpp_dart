/// Audio probe for Gemma-4-Unsloth via manual prompt rendering.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/Users/adel/Workspace/gguf/gemma-4-E2B-it-Q8_0.gguf \
///   LLAMA_CPP_DART_MMPROJ=/Users/adel/Workspace/gguf/mmproj-F16.gguf \
///     dart run example/probes/gemma_audio.dart `<audio-path>` [prompt]
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main(List<String> args) async {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB']!;
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL']!;
  final mmprojPath = Platform.environment['LLAMA_CPP_DART_MMPROJ']!;

  if (args.isEmpty) {
    stderr.writeln('usage: gemma_audio.dart <audio-path> [prompt]');
    exit(2);
  }
  final audioPath = args[0];
  final userText = args.length > 1
      ? args.sublist(1).join(' ')
      : 'Transcribe this audio.';

  stdout.writeln('starting engine ...');
  final engine = await LlamaEngine.spawn(
    libraryPath: libPath,
    modelParams: ModelParams(path: modelPath, gpuLayers: 99),
    contextParams: const ContextParams(nCtx: 4096, nBatch: 512, nUbatch: 512),
    multimodalParams: MultimodalParams(mmprojPath: mmprojPath),
  );
  stdout.writeln(
    'audio=${engine.supportsAudio} '
    'sample_rate=${engine.audioSampleRate} '
    'canShift=${engine.canShift}',
  );

  if (!engine.supportsAudio) {
    stderr.writeln('this model does not advertise audio support');
    await engine.dispose();
    exit(1);
  }

  final session = await engine.createSession();
  final prompt = '<|turn>user\n<__media__>\n$userText<turn|>\n<|turn>model\n';

  stdout.write('<<< ');
  final swatch = Stopwatch()..start();
  var got = 0;
  await for (final ev in session.generate(
    prompt: prompt,
    addSpecial: true,
    parseSpecial: true,
    sampler: const SamplerParams(temperature: 0.3, topP: 0.9),
    maxTokens: 192,
    media: [LlamaMedia.audioFile(audioPath)],
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
