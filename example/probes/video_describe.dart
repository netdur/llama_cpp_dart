/// Describe a video with a frame-based vision model (e.g.
/// SmolVLM2-256M-Video-Instruct).
///
/// libmtmd has no video decoder — a "video" is fed to the model as a
/// **sequence of image frames**. This probe extracts frames with ffmpeg,
/// wraps them via [LlamaMedia.videoFrames], and sends them through the
/// normal multimodal chat path (one `<__media__>` marker per frame, added
/// automatically by `EngineChat.addUser`).
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/SmolVLM2-256M-Video-Instruct-Q8_0.gguf \
///   LLAMA_CPP_DART_MMPROJ=/path/to/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf \
///     dart run example/probes/video_describe.dart video.mp4 "Describe this video."
///
/// Tunables (env): FPS (frames/sec to sample, default 1), MAX_FRAMES
/// (cap, default 8), FRAME_HEIGHT (downscale, default 384).
///
/// Requires `ffmpeg` on PATH for frame extraction.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main(List<String> args) async {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  final mmprojPath = Platform.environment['LLAMA_CPP_DART_MMPROJ'];
  if (libPath == null || modelPath == null || mmprojPath == null) {
    stderr.writeln('set LLAMA_CPP_DART_LIB, LLAMA_CPP_DART_MODEL, '
        'and LLAMA_CPP_DART_MMPROJ');
    exit(2);
  }
  if (args.isEmpty) {
    stderr.writeln('usage: video_describe.dart <video-path> [prompt]');
    exit(2);
  }
  final videoPath = args[0];
  final prompt =
      args.length > 1 ? args.sublist(1).join(' ') : 'Describe this video.';

  final fps = double.tryParse(Platform.environment['FPS'] ?? '') ?? 1.0;
  final maxFrames = int.tryParse(Platform.environment['MAX_FRAMES'] ?? '') ?? 8;
  final frameHeight =
      int.tryParse(Platform.environment['FRAME_HEIGHT'] ?? '') ?? 384;

  // ---- 1. Extract frames with ffmpeg (outside libmtmd). ----
  final frames = _extractFrames(
    videoPath,
    fps: fps,
    maxFrames: maxFrames,
    height: frameHeight,
  );
  if (frames.isEmpty) {
    stderr.writeln('no frames extracted — is ffmpeg installed and the '
        'video readable?');
    exit(1);
  }
  stdout.writeln('extracted ${frames.length} frames');

  // ---- 2. Wrap frames as image media (one marker per frame). ----
  final media = LlamaMedia.videoFrames(frames, idPrefix: 'frame');

  final engine = await LlamaEngine.spawn(
    libraryPath: libPath,
    modelParams: ModelParams(path: modelPath, gpuLayers: 99),
    contextParams: const ContextParams(nCtx: 8192, nBatch: 4096, nUbatch: 4096),
    multimodalParams: MultimodalParams(mmprojPath: mmprojPath),
  );
  if (!engine.supportsVision) {
    stderr.writeln('this mmproj does not support vision');
    await engine.dispose();
    exit(1);
  }

  // ---- 3. Feed the frames through the normal multimodal chat path. ----
  final chat = await engine.createChat();
  chat.addUser(prompt, media: media);

  stdout.write('<<< ');
  final sw = Stopwatch()..start();
  var got = 0;
  await for (final ev in chat.generate(
    sampler: const SamplerParams(temperature: 0.3, topP: 0.9),
    maxTokens: 256,
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
        final secs = sw.elapsedMilliseconds / 1000.0;
        stdout.writeln('--- ${ev.reason} gen=${ev.generatedCount} '
            '${(secs > 0 ? got / secs : 0).toStringAsFixed(1)} tok/s');
    }
  }

  await engine.dispose();
}

/// Sample up to [maxFrames] frames at [fps] frames/sec, downscaled to
/// [height] px, returning the encoded JPG bytes of each frame in order.
List<Uint8List> _extractFrames(
  String videoPath, {
  required double fps,
  required int maxFrames,
  required int height,
}) {
  final outDir = Directory.systemTemp.createTempSync('llama_video_frames_');
  try {
    final result = Process.runSync('ffmpeg', [
      '-y',
      '-loglevel',
      'error',
      '-i',
      videoPath,
      '-vf',
      'fps=$fps,scale=-2:$height',
      '-frames:v',
      '$maxFrames',
      '${outDir.path}/frame_%03d.jpg',
    ]);
    if (result.exitCode != 0) {
      stderr.writeln('ffmpeg failed: ${result.stderr}');
      return const [];
    }
    final files = outDir
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.jpg'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    return [for (final f in files) f.readAsBytesSync()];
  } finally {
    outDir.deleteSync(recursive: true);
  }
}
