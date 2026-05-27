import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import 'multimodal_context.dart';

/// Owned wrapper around an `mtmd_bitmap`.
///
/// Carries a decoded image (RGB, `nx * ny * 3` bytes) or an audio sample
/// stream (`nx` floats). Construct via [MtmdBitmap.image] / [MtmdBitmap.audio]
/// for raw input, or [MtmdBitmap.loadImage] to let llama.cpp decode a file
/// for you. Always [dispose] when you're done — the bitmap holds native
/// memory.
final class MtmdBitmap implements Finalizable {
  final Pointer<mtmd_bitmap> _ptr;
  bool _disposed = false;

  MtmdBitmap._(this._ptr);

  /// Build an RGB image bitmap of `nx x ny` pixels. [rgb] must contain
  /// exactly `nx * ny * 3` bytes in row-major RGB order.
  factory MtmdBitmap.image({
    required int nx,
    required int ny,
    required Uint8List rgb,
  }) {
    final expected = nx * ny * 3;
    if (rgb.length != expected) {
      throw ArgumentError(
        'rgb must have $expected bytes for ${nx}x$ny (got ${rgb.length})',
      );
    }
    final buf = calloc<UnsignedChar>(rgb.length);
    try {
      buf.cast<Uint8>().asTypedList(rgb.length).setAll(0, rgb);
      final ptr = LlamaLibrary.bindings.mtmd_bitmap_init(nx, ny, buf);
      if (ptr == nullptr) {
        throw const MultimodalException('mtmd_bitmap_init returned null');
      }
      return MtmdBitmap._(ptr);
    } finally {
      calloc.free(buf);
    }
  }

  /// Build an audio bitmap of [samples] mono float PCM samples.
  factory MtmdBitmap.audio(Float32List samples) {
    final buf = calloc<Float>(samples.length);
    try {
      buf.asTypedList(samples.length).setAll(0, samples);
      final ptr = LlamaLibrary.bindings.mtmd_bitmap_init_from_audio(
        samples.length,
        buf,
      );
      if (ptr == nullptr) {
        throw const MultimodalException(
          'mtmd_bitmap_init_from_audio returned null',
        );
      }
      return MtmdBitmap._(ptr);
    } finally {
      calloc.free(buf);
    }
  }

  /// Let llama.cpp decode an image file (PNG / JPEG / etc.) into an RGB
  /// bitmap. Requires a live [MultimodalContext] because the decoder uses
  /// the loaded mmproj's preprocessing parameters.
  factory MtmdBitmap.loadImage({
    required MultimodalContext context,
    required String path,
  }) {
    final pathPtr = path.toNativeUtf8();
    try {
      final ptr = LlamaLibrary.bindings.mtmd_helper_bitmap_init_from_file(
        context.pointer,
        pathPtr.cast<Char>(),
      );
      if (ptr == nullptr) {
        throw MultimodalException('failed to decode image: $path');
      }
      return MtmdBitmap._(ptr);
    } finally {
      calloc.free(pathPtr);
    }
  }

  /// Decode an image from in-memory bytes (e.g. a `Uint8List` from
  /// `rootBundle.load` or a network fetch). Pairs with [loadImage].
  factory MtmdBitmap.loadImageBytes({
    required MultimodalContext context,
    required Uint8List bytes,
  }) {
    final buf = calloc<UnsignedChar>(bytes.length);
    try {
      buf.cast<Uint8>().asTypedList(bytes.length).setAll(0, bytes);
      final ptr = LlamaLibrary.bindings.mtmd_helper_bitmap_init_from_buf(
        context.pointer,
        buf,
        bytes.length,
      );
      if (ptr == nullptr) {
        throw const MultimodalException(
          'failed to decode image from buffer',
        );
      }
      return MtmdBitmap._(ptr);
    } finally {
      calloc.free(buf);
    }
  }

  Pointer<mtmd_bitmap> get pointer {
    _ensureAlive();
    return _ptr;
  }

  /// Width for image bitmaps; total sample count for audio bitmaps.
  int get nx => LlamaLibrary.bindings.mtmd_bitmap_get_nx(pointer);

  /// Height for image bitmaps; `1` for audio bitmaps.
  int get ny => LlamaLibrary.bindings.mtmd_bitmap_get_ny(pointer);

  /// Total number of bytes addressable through [bytes]. For image bitmaps
  /// this is `nx * ny * 3`; for audio it is `nx * sizeof(float)`.
  int get nBytes => LlamaLibrary.bindings.mtmd_bitmap_get_n_bytes(pointer);

  /// True if this bitmap carries audio samples rather than RGB pixels.
  bool get isAudio => LlamaLibrary.bindings.mtmd_bitmap_is_audio(pointer);

  /// Optional caller-assigned id. Used by some pipelines to track which
  /// chunks correspond to which source image. Empty when unset.
  String get id {
    final ptr = LlamaLibrary.bindings.mtmd_bitmap_get_id(pointer);
    if (ptr == nullptr) return '';
    return ptr.cast<Utf8>().toDartString();
  }

  /// Zero-copy view over the raw byte buffer. **Lifetime is tied to this
  /// bitmap** — don't read after [dispose].
  Uint8List bytes() {
    final ptr = LlamaLibrary.bindings.mtmd_bitmap_get_data(pointer);
    if (ptr == nullptr) return Uint8List(0);
    return ptr.cast<Uint8>().asTypedList(nBytes);
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    LlamaLibrary.bindings.mtmd_bitmap_free(_ptr);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('MtmdBitmap has been disposed.');
    }
  }
}

/// Capabilities of an mmproj file probed without fully loading it.
final class MtmdCapabilities {
  /// True if the projector supports image input.
  final bool supportsVision;

  /// True if the projector supports audio input.
  final bool supportsAudio;

  const MtmdCapabilities({
    required this.supportsVision,
    required this.supportsAudio,
  });

  /// Cheap probe — opens the file just enough to read its header. Useful
  /// for app-side feature gating before paying for a full
  /// [MultimodalContext.init].
  factory MtmdCapabilities.probeFromFile(String mmprojPath) {
    final p = mmprojPath.toNativeUtf8();
    try {
      final caps =
          LlamaLibrary.bindings.mtmd_get_cap_from_file(p.cast<Char>());
      return MtmdCapabilities(
        supportsVision: caps.inp_vision,
        supportsAudio: caps.inp_audio,
      );
    } finally {
      calloc.free(p);
    }
  }
}
