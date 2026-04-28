/// Stream a description / transcription of an audio clip via the model's
/// multimodal projector.
///
/// Bypasses the chat template — useful for models whose Jinja isn't
/// recognized by `llama_chat_apply_template`. The prompt is fed verbatim,
/// with one `<__media__>` marker per audio attachment.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/llm.gguf \
///   LLAMA_CPP_DART_MMPROJ=/path/to/mmproj.gguf \
///     dart run example/probes/audio_describe.dart /path/to/clip.mp3
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main(List<String> args) async {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  final mmprojPath = Platform.environment['LLAMA_CPP_DART_MMPROJ'];
  if (libPath == null || modelPath == null || mmprojPath == null) {
    stderr.writeln(
      'set LLAMA_CPP_DART_LIB, LLAMA_CPP_DART_MODEL, LLAMA_CPP_DART_MMPROJ',
    );
    exit(2);
  }

  if (args.isEmpty) {
    stderr.writeln('usage: audio_describe.dart <audio-path> [prompt]');
    exit(2);
  }
  final audioPath = args[0];
  final userPrompt = args.length > 1
      ? args.sublist(1).join(' ')
      : 'Describe what you hear in this audio.';

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

  if (!engine.supportsAudio) {
    stderr.writeln('this model + projector does not advertise audio support');
    await engine.dispose();
    exit(1);
  }

  final session = await engine.createSession();

  // Plain prompt (no chat template). Adjust the wrapper if you know the
  // model's expected user-token style; many audio-capable models work with
  // a bare prompt + marker.
  final prompt = '$userPrompt\n<__media__>';

  stdout.write('<<< ');
  final swatch = Stopwatch()..start();
  var generated = 0;

  await for (final event in session.generate(
    prompt: prompt,
    addSpecial: true,
    sampler: const SamplerParams(temperature: 0.4, topP: 0.9),
    maxTokens: 192,
    media: [LlamaMedia.audioFile(audioPath)],
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

  await session.dispose();
  await engine.dispose();
}
