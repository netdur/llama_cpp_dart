# Video Support in llama_cpp_dart

This document describes how to use video input with llama_cpp_dart, conforming to llama.cpp's official multimodal API specification.

## Overview

llama_cpp_dart now supports video input for multimodal models like **SmolVLM2-256M-Video-Instruct**. Videos are automatically decoded by llama.cpp's `libmtmd` encoder — no manual frame extraction is required on the Dart side.

## Architecture

```
Video File (mp4/mov/webm/...)
    ↓
LlamaMedia.videoFile() / LlamaMedia.videoBytes()
    ↓
EngineSession.generate() / EngineChat
    ↓
libmtmd (automatic frame extraction & decoding)
    ↓
Vision Encoder (e.g., SigLIP for SmolVLM2)
    ↓
Text Embedding Space
```

## Quick Start

### 1. Basic Video Generation

```dart
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

// Initialize engine with video support
final engine = await LlamaEngine.spawn(
  libraryPath: '/path/to/libllama.dylib',
  modelParams: ModelParams(
    path: '/path/to/smolvlm2-256m-video.gguf',
    gpuLayers: 99,
  ),
  contextParams: const ContextParams(nCtx: 4096),
  multimodalParams: MultimodalParams(
    mmprojPath: '/path/to/mmproj.gguf',
  ),
);

// Create a session
final session = await engine.createSession();

// Generate from video
await for (final event in session.generate(
  prompt: 'Describe this video:\n<__media__>',
  addSpecial: true,
  maxTokens: 256,
  media: [LlamaMedia.videoFile('path/to/video.mp4')],
)) {
  if (event is TokenEvent) {
    stdout.write(event.text);
  }
}

await session.dispose();
await engine.dispose();
```

### 2. Video in Chat

```dart
final chat = await engine.createChat();

// Add user message with video
chat.addUser(
  'What happens in this video?',
  media: [LlamaMedia.videoFile('video.mp4')],
);

// The media marker is automatically prepended
// chat.messages.last.content now contains: "<__media__>\nWhat happens in this video?"

// Generate response
await for (final event in chat.generate(maxTokens: 256)) {
  if (event is TokenEvent) {
    stdout.write(event.text);
  }
}

await chat.dispose();
```

### 3. In-Memory Video Data

```dart
import 'dart:io';

final videoBytes = File('video.mp4').readAsBytesSync();
final media = LlamaMedia.videoBytes(videoBytes, id: 'my-video');

await for (final event in session.generate(
  prompt: '<__media__>\nAnalyze this video.',
  media: [media],
)) {
  // ... process events
}
```

## API Reference

### LlamaMedia

The `LlamaMedia` class carries video (and image/audio) data.

#### Constructors

```dart
// Load video from file
factory LlamaMedia.videoFile(String path)
  /// Frames are automatically extracted by llama.cpp's mtmd

// Wrap in-memory video bytes
factory LlamaMedia.videoBytes(Uint8List bytes, {String? id})
  /// Frames are automatically extracted by llama.cpp's mtmd
  /// Optional [id] is used for diagnostics and KV-cache hashing
```

#### Supported Formats

- **MP4** (H.264, H.265, VP9, AV1 codecs)
- **MOV** (QuickTime)
- **WebM** (VP9, AV1)
- **MKV** (Matroska)
- **And more** (format auto-detected via magic bytes)

The exact supported formats depend on the video decoders available in llama.cpp's build.

### MediaKind Enum

```dart
enum MediaKind {
  image,  // JPG/PNG/BMP/GIF/...
  audio,  // WAV/MP3/FLAC/...
  video,  // MP4/MOV/WebM/... (NEW)
}
```

All three kinds are handled uniformly by the `<__media__>` marker in prompts.

### MultimodalParams

Extended with video-specific parameters:

```dart
final params = MultimodalParams(
  mmprojPath: '/path/to/mmproj.gguf',

  // Image settings (existing)
  imageMinTokens: 0,   // Lower bound on image token count
  imageMaxTokens: 0,   // Upper bound (0 = use model default)

  // Video settings (NEW)
  videoMinTokens: 0,   // Lower bound on video token count
  videoMaxTokens: 0,   // Upper bound (0 = use model default)

  // Other settings
  useGpu: true,
  mediaMarker: '<__media__>',
  warmup: false,
);
```

**Note**: Token limits depend on the model's architecture. SmolVLM2 uses dynamic resolution, so these parameters control how many tokens are emitted per frame.

### EngineSession.generate()

The `media` parameter now supports video:

```dart
await for (final event in session.generate(
  prompt: 'Describe: <__media__>',
  media: [
    LlamaMedia.videoFile('video.mp4'),
    LlamaMedia.imageFile('image.png'),  // Mix with images
  ],
  maxTokens: 256,
)) {
  // Generate as normal
}
```

- `media` list items are processed in order
- Each item gets replaced by its corresponding marker in the prompt
- Videos are automatically decoded by libmtmd

## Implementation Details

### Frame Extraction

Unlike some frameworks, llama_cpp_dart **does not require** manual frame extraction:

1. Video bytes are passed directly to `libmtmd`
2. `libmtmd` automatically:
   - Detects the video format (mp4/mov/webm/...)
   - Decodes frames using platform video decoders
   - Optionally resamples or crops based on model requirements
   - Encodes frames through the vision model (e.g., SigLIP for SmolVLM2)

### Media Marker System

All media types (image, audio, video) use the same marker syntax:

```dart
// Marker is substituted in the prompt for each media item
final prompt = 'Compare these: <__media__> and <__media__>';
final media = [
  LlamaMedia.videoFile('video.mp4'),
  LlamaMedia.imageFile('image.png'),
];
// Result: "Compare these: [video_tokens] and [image_tokens]"
```

**Important**: Marker count must equal media count.

### Video Token Calculation

For SmolVLM2-256M-Video:

- **Per-frame tokens**: Varies based on video resolution (dynamic resolution)
- **Total tokens**: frames × per-frame tokens
- **Min/Max limits**: Controlled by `videoMinTokens` / `videoMaxTokens`

Example with SmolVLM2:
- 30-frame video at 256x256: ~3,000–6,000 tokens total
- Token budget affects max_tokens parameter effectiveness

## SmolVLM2-256M-Video-Instruct Example

This example targets the SmolVLM2-256M-Video-Instruct model specifically:

```dart
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'dart:io';

void main() async {
  final engine = await LlamaEngine.spawn(
    libraryPath: Platform.environment['LLAMA_CPP_DART_LIB']!,
    modelParams: ModelParams(
      path: Platform.environment['LLAMA_CPP_DART_MODEL']!,
      gpuLayers: 99,
    ),
    contextParams: const ContextParams(nCtx: 4096, nBatch: 512),
    multimodalParams: MultimodalParams(
      mmprojPath: Platform.environment['LLAMA_CPP_DART_MMPROJ']!,
      // SmolVLM2 uses dynamic resolution
      videoMinTokens: 0,
      videoMaxTokens: 0,
    ),
  );

  print('Vision support: ${engine.supportsVision}');

  final session = await engine.createSession();

  final result = <String>[];
  await for (final event in session.generate(
    prompt: '<__media__>\nWhat is happening in this video? Answer in one sentence.',
    addSpecial: true,
    sampler: const SamplerParams(temperature: 0.3),
    maxTokens: 128,
    media: [LlamaMedia.videoFile('action_video.mp4')],
  )) {
    if (event is TokenEvent) {
      result.add(event.text);
    }
  }

  print('Response: ${result.join()}');

  await session.dispose();
  await engine.dispose();
}
```

Run with:

```bash
LLAMA_CPP_DART_LIB=/path/to/libllama.dylib \
LLAMA_CPP_DART_MODEL=/path/to/smolvlm2-256m-video-instruct.gguf \
LLAMA_CPP_DART_MMPROJ=/path/to/mmproj.gguf \
  dart run example/probes/gemma_video.dart video.mp4 "Your prompt here"
```

## Performance Considerations

1. **Memory**: Video decoding happens on the GPU/NPU when `useGpu: true`
2. **Latency**: First token latency includes video frame extraction (warmup phase recommended)
3. **Token budget**: Video tokens consume significant context; adjust `max_tokens` accordingly
4. **Batch size**: Larger batch sizes recommended for video encoding efficiency

## Comparison: Image vs Video vs Audio

| Aspect | Image | Audio | Video |
|--------|-------|-------|-------|
| Formats | JPG, PNG, BMP, GIF | WAV, MP3, FLAC | MP4, MOV, WebM, MKV |
| Decoder | stb_image | miniaudio | libmtmd video codec |
| Frame handling | Single frame | Single channel | Multiple frames (auto-extracted) |
| Token count | Fixed per resolution | Variable (sample rate) | Variable per frame count |
| Marker usage | `<__media__>` | `<__media__>` | `<__media__>` |
| Dart-side processing | None required | None required | **None required** ✨ |

## Troubleshooting

### "Unsupported video format"
- Check that llama.cpp is built with video decoder support
- Verify the video file is not corrupted (try with MP4 H.264)
- Check libmtmd logs for more details

### Slow first frame
- Consider `warmup: true` in `MultimodalParams` for more consistent latency

### Out of memory
- Reduce `max_tokens` to leave more context for video encoding
- Reduce video resolution before passing to the model
- Disable GPU acceleration with `useGpu: false`

### Wrong token count
- Verify `videoMinTokens` and `videoMaxTokens` align with model expectations
- Use llama.cpp's `--verbose` mode to see actual token counts

## API Compliance

This implementation follows llama.cpp's official multimodal API:

- ✅ Uniform media marker system (`<__media__>`)
- ✅ Automatic format detection (no manual codec specification)
- ✅ libmtmd-based encoding (seamless decoding)
- ✅ GPU acceleration support
- ✅ Compatible with all projector types (CLIP, SigLIP, etc.)

See [llama.cpp multimodal documentation](https://github.com/ggerganov/llama.cpp/tree/master/examples/llava) for details.

## References

- [SmolVLM2 Model Card](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)
- [llama.cpp mtmd (multimodal)](https://github.com/ggerganov/llama.cpp/tree/master/tools/mtmd)
- [Idefics3 Architecture](https://huggingface.co/papers/2405.02246)
