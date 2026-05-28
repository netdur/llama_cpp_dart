import 'dart:io';
import 'dart:typed_data';

/// Kind of media carried by a [LlamaMedia].
enum MediaKind {
  /// An image. mtmd auto-detects the file format (jpg/png/bmp/gif/...) via
  /// magic bytes, decoded with stb_image inside libmtmd.
  image,

  /// An audio clip. mtmd decodes wav/mp3/flac with miniaudio inside libmtmd
  /// and resamples to the model's expected rate (typically 16 kHz mono).
  audio,

  /// A video clip. mtmd auto-detects the video format (mp4/mov/webm/...) via
  /// magic bytes, decoded with libmtmd's video decoder. Video frames are
  /// automatically extracted and processed (e.g., SmolVLM2-256M-Video-Instruct).
  /// Sendable across isolates like images/audio.
  video,
}

/// One image or audio clip attached to a chat turn.
///
/// Sendable across isolates (holds only [Uint8List] + primitives). The
/// worker hands the bytes to llama.cpp's mtmd helper for decoding —
/// neither stb_image nor miniaudio are required on the Dart side.
final class LlamaMedia {
  /// Raw bytes of the media file (encoded — JPG/PNG/MP3/WAV/etc.).
  final Uint8List bytes;

  /// Image vs audio. Used for documentation and validation; mtmd's helper
  /// auto-detects from the file's magic bytes regardless.
  final MediaKind kind;

  /// Optional id used for diagnostics + KV-cache hashing inside mtmd.
  final String? id;

  const LlamaMedia({
    required this.bytes,
    required this.kind,
    this.id,
  });

  /// Load image bytes from [path]. Format detected from file contents, not
  /// the extension.
  factory LlamaMedia.imageFile(String path) =>
      LlamaMedia(bytes: File(path).readAsBytesSync(), kind: MediaKind.image);

  /// Wrap already-read image bytes.
  factory LlamaMedia.imageBytes(Uint8List bytes, {String? id}) =>
      LlamaMedia(bytes: bytes, kind: MediaKind.image, id: id);

  /// Load audio bytes from [path]. Format detected from file contents.
  factory LlamaMedia.audioFile(String path) =>
      LlamaMedia(bytes: File(path).readAsBytesSync(), kind: MediaKind.audio);

  /// Wrap already-read audio bytes.
  factory LlamaMedia.audioBytes(Uint8List bytes, {String? id}) =>
      LlamaMedia(bytes: bytes, kind: MediaKind.audio, id: id);

  /// Load video bytes from [path]. Format detected from file contents (mp4/mov/webm/...).
  /// Frames are automatically extracted and processed by llama.cpp's mtmd.
  factory LlamaMedia.videoFile(String path) =>
      LlamaMedia(bytes: File(path).readAsBytesSync(), kind: MediaKind.video);

  /// Wrap already-read video bytes.
  /// Frames are automatically extracted and processed by llama.cpp's mtmd.
  factory LlamaMedia.videoBytes(Uint8List bytes, {String? id}) =>
      LlamaMedia(bytes: bytes, kind: MediaKind.video, id: id);

  @override
  String toString() =>
      'LlamaMedia(${kind.name}, ${bytes.length} bytes${id == null ? '' : ', id=$id'})';
}
