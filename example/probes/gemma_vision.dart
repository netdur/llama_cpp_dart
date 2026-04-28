/// Multimodal chat probe for Gemma-4-Unsloth.
///
/// Builds the prompt manually with Gemma-4 turn markers + a `<__media__>`
/// placeholder, then feeds it through `EngineSession.generate(media:)`.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/Users/adel/Workspace/gguf/gemma-4-E2B-it-Q8_0.gguf \
///   LLAMA_CPP_DART_MMPROJ=/Users/adel/Workspace/gguf/mmproj-F16.gguf \
///     dart run example/probes/gemma_vision.dart `<image-path>` [prompt]
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main(List<String> args) async {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB']!;
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL']!;
  final mmprojPath = Platform.environment['LLAMA_CPP_DART_MMPROJ']!;

  if (args.isEmpty) {
    stderr.writeln('usage: gemma_vision.dart <image-path> [prompt]');
    exit(2);
  }
  final imagePath = args[0];
  final userText = args.length > 1
      ? args.sublist(1).join(' ')
      : 'Describe this image in one sentence.';

  stdout.writeln('starting engine ...');
  final engine = await LlamaEngine.spawn(
    libraryPath: libPath,
    modelParams: ModelParams(path: modelPath, gpuLayers: 99),
    contextParams: const ContextParams(nCtx: 4096, nBatch: 512, nUbatch: 512),
    multimodalParams: MultimodalParams(mmprojPath: mmprojPath),
  );
  stdout.writeln(
    'multimodal: vision=${engine.supportsVision} '
    'audio=${engine.supportsAudio} '
    'sample_rate=${engine.audioSampleRate} '
    'canShift=${engine.canShift}',
  );

  final session = await engine.createSession();

  // Manual Gemma-4 turn-marker prompt with the media placeholder.
  final prompt = '<|turn>user\n<__media__>\n$userText<turn|>\n<|turn>model\n';

  stdout.write('<<< ');
  final swatch = Stopwatch()..start();
  var got = 0;
  await for (final ev in session.generate(
    prompt: prompt,
    addSpecial: true,
    parseSpecial: true,
    sampler: const SamplerParams(temperature: 0.3, topP: 0.9),
    maxTokens: 128,
    media: [LlamaMedia.imageFile(imagePath)],
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
