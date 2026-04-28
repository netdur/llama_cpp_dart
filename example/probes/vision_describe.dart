/// Stream a multimodal description of an image.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/llm.gguf \
///   LLAMA_CPP_DART_MMPROJ=/path/to/mmproj.gguf \
///     dart run example/probes/vision_describe.dart /path/to/image.jpg
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main(List<String> args) async {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  final mmprojPath = Platform.environment['LLAMA_CPP_DART_MMPROJ'];
  if (libPath == null || modelPath == null || mmprojPath == null) {
    stderr.writeln(
      'set LLAMA_CPP_DART_LIB, LLAMA_CPP_DART_MODEL, and LLAMA_CPP_DART_MMPROJ',
    );
    exit(2);
  }

  if (args.isEmpty) {
    stderr.writeln('usage: vision_describe.dart <image-path> [prompt]');
    exit(2);
  }
  final imagePath = args[0];
  final prompt = args.length > 1
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
    'sample_rate=${engine.audioSampleRate}',
  );

  final chat = await engine.createChat();
  chat.addUser(prompt, media: [LlamaMedia.imageFile(imagePath)]);

  stdout.write('<<< ');
  final swatch = Stopwatch()..start();
  var generated = 0;

  await for (final event in chat.generate(
    sampler: const SamplerParams(temperature: 0.4, topP: 0.9),
    maxTokens: 128,
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
          '--- ${event.reason} gen=${event.generatedCount} '
          '${tps.toStringAsFixed(1)} tok/s '
          'elapsed=${swatch.elapsedMilliseconds}ms',
        );
    }
  }

  await chat.dispose();
  await engine.dispose();
}
