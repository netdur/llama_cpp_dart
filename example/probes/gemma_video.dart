/// Multimodal video chat probe for SmolVLM2-256M-Video-Instruct.
///
/// Demonstrates video processing capabilities compatible with llama.cpp's
/// official API specification. Videos are automatically decoded by libmtmd
/// (no manual frame extraction required).
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/smolvlm2-256m-video.gguf \
///   LLAMA_CPP_DART_MMPROJ=/path/to/mmproj.gguf \
///     dart run example/probes/gemma_video.dart `<video-path>` [prompt]
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main(List<String> args) async {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB']!;
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL']!;
  final mmprojPath = Platform.environment['LLAMA_CPP_DART_MMPROJ']!;

  if (args.isEmpty) {
    stderr.writeln('usage: gemma_video.dart <video-path> [prompt]');
    exit(2);
  }
  final videoPath = args[0];
  final userText = args.length > 1
      ? args.sublist(1).join(' ')
      : 'Describe this video in detail.';

  // Validate video file exists
  if (!File(videoPath).existsSync()) {
    stderr.writeln('error: video file not found: $videoPath');
    exit(1);
  }

  stdout.writeln('starting engine ...');
  final engine = await LlamaEngine.spawn(
    libraryPath: libPath,
    modelParams: ModelParams(path: modelPath, gpuLayers: 99),
    contextParams: const ContextParams(nCtx: 4096, nBatch: 512, nUbatch: 512),
    multimodalParams: MultimodalParams(
      mmprojPath: mmprojPath,
      // Optional: tune video token limits for SmolVLM2
      videoMinTokens: 0,  // use model default
      videoMaxTokens: 0,  // use model default
    ),
  );
  stdout.writeln(
    'multimodal: vision=${engine.supportsVision} '
    'audio=${engine.supportsAudio} '
    'sample_rate=${engine.audioSampleRate} '
    'canShift=${engine.canShift}',
  );

  final session = await engine.createSession();

  // Manual Gemma-4 turn-marker prompt with the media placeholder.
  // Video is automatically decoded by libmtmd from the video bytes.
  final prompt = '<|turn>user\n<__media__>\n$userText<turn|>\n<|turn>model\n';

  stdout.write('<<< ');
  final swatch = Stopwatch()..start();
  var got = 0;
  
  // Load video and pass to generate()
  // Format is automatically detected (mp4/mov/webm/...)
  // Frames are automatically extracted by llama.cpp's mtmd encoder
  await for (final ev in session.generate(
    prompt: prompt,
    addSpecial: true,
    parseSpecial: true,
    sampler: const SamplerParams(temperature: 0.3, topP: 0.9),
    maxTokens: 256,
    media: [LlamaMedia.videoFile(videoPath)],
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
