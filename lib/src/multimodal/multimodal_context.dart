import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../context/context.dart';
import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import '../model/model.dart';
import '../types/exceptions.dart';
import 'media.dart';
import 'multimodal_params.dart';

/// Thrown when multimodal init / encoding fails.
final class MultimodalException extends LlamaException {
  const MultimodalException(super.message);
}

/// Owned wrapper around an `mtmd_context`.
///
/// Bound to a single [LlamaModel] for the lifetime of the wrapper. Holds
/// the multimodal projector weights and the encoder graph; one instance
/// can serve many generation turns. Not thread-safe — use one per worker
/// isolate.
final class MultimodalContext implements Finalizable {
  final LlamaModel model;
  final MultimodalParams params;
  final Pointer<mtmd_context> _ptr;
  bool _disposed = false;

  MultimodalContext._(this.model, this.params, this._ptr);

  static MultimodalContext init({
    required LlamaModel model,
    required MultimodalParams params,
  }) {
    final b = LlamaLibrary.bindings;

    final cp = b.mtmd_context_params_default()
      ..use_gpu = params.useGpu
      ..print_timings = params.printTimings
      ..n_threads = params.nThreads
      ..warmup = params.warmup
      ..image_min_tokens = params.imageMinTokens
      ..image_max_tokens = params.imageMaxTokens;

    final markerPtr = params.mediaMarker.toNativeUtf8(allocator: calloc);
    final pathPtr = params.mmprojPath.toNativeUtf8(allocator: calloc);
    try {
      cp.media_marker = markerPtr.cast();

      final ptr = b.mtmd_init_from_file(
        pathPtr.cast(),
        model.pointer,
        cp,
      );
      if (ptr == nullptr) {
        throw MultimodalException(
          'mtmd_init_from_file returned null for ${params.mmprojPath}',
        );
      }
      return MultimodalContext._(model, params, ptr);
    } finally {
      calloc.free(markerPtr);
      calloc.free(pathPtr);
    }
  }

  Pointer<mtmd_context> get pointer {
    _ensureAlive();
    return _ptr;
  }

  bool get supportsVision => LlamaLibrary.bindings.mtmd_support_vision(pointer);

  bool get supportsAudio => LlamaLibrary.bindings.mtmd_support_audio(pointer);

  /// Sample rate the model expects audio at, or `-1` if audio isn't supported.
  int get audioSampleRate =>
      LlamaLibrary.bindings.mtmd_get_audio_sample_rate(pointer);

  /// Default media marker — read from llama.cpp at init time.
  static String defaultMarker() {
    final ptr = LlamaLibrary.bindings.mtmd_default_marker();
    if (ptr == nullptr) return '<__media__>';
    return ptr.cast<Utf8>().toDartString();
  }

  /// Render [prompt] + [media] into a list of `mtmd_input_chunk`s and feed
  /// them to [llamaContext], starting at [nPast] / sequence [seqId].
  ///
  /// Returns the new `n_past` (= total tokens / positions consumed). The
  /// caller is expected to drive sampling from there.
  ///
  /// [logitsLast] requests that the final position have logits available
  /// (so the immediate next sample picks up where we left off).
  int evalChunks({
    required LlamaContext llamaContext,
    required String prompt,
    required List<LlamaMedia> media,
    required int nPast,
    required int seqId,
    required int nBatch,
    bool addSpecial = true,
    bool parseSpecial = true,
    bool logitsLast = true,
  }) {
    _ensureAlive();
    final b = LlamaLibrary.bindings;

    // 1. Build mtmd_bitmap pointers from each media item.
    final bitmaps = <Pointer<mtmd_bitmap>>[];
    final byteArenas = <Pointer<UnsignedChar>>[];
    final idArenas = <Pointer<Utf8>>[];
    Pointer<Pointer<mtmd_bitmap>>? bitmapArray;
    Pointer<mtmd_input_chunks>? chunks;
    Pointer<mtmd_input_text>? textStruct;
    Pointer<Utf8>? promptPtr;
    Pointer<llama_pos>? newNPast;

    try {
      for (final m in media) {
        final buf = calloc<UnsignedChar>(m.bytes.length);
        buf.cast<Uint8>().asTypedList(m.bytes.length).setAll(0, m.bytes);
        byteArenas.add(buf);

        final bm = b.mtmd_helper_bitmap_init_from_buf(
          pointer,
          buf,
          m.bytes.length,
        );
        if (bm == nullptr) {
          throw MultimodalException(
            'mtmd_helper_bitmap_init_from_buf returned null '
            '(${m.kind.name}, ${m.bytes.length} bytes)',
          );
        }
        if (m.id != null) {
          final idPtr = m.id!.toNativeUtf8(allocator: calloc);
          idArenas.add(idPtr);
          b.mtmd_bitmap_set_id(bm, idPtr.cast());
        }
        bitmaps.add(bm);
      }

      bitmapArray = calloc<Pointer<mtmd_bitmap>>(bitmaps.length);
      for (var i = 0; i < bitmaps.length; i++) {
        bitmapArray[i] = bitmaps[i];
      }

      // 2. Tokenize prompt + bitmaps into mtmd_input_chunks.
      chunks = b.mtmd_input_chunks_init();
      promptPtr = prompt.toNativeUtf8(allocator: calloc);
      textStruct = calloc<mtmd_input_text>();
      textStruct.ref
        ..text = promptPtr.cast()
        ..add_special = addSpecial
        ..parse_special = parseSpecial;

      final tokenizeRc = b.mtmd_tokenize(
        pointer,
        chunks,
        textStruct,
        bitmapArray,
        bitmaps.length,
      );
      if (tokenizeRc != 0) {
        throw MultimodalException(
          'mtmd_tokenize failed: rc=$tokenizeRc '
          '(${tokenizeRc == 1 ? "marker count mismatch" : "preprocessing error"})',
        );
      }

      // 3. Encode + decode all chunks into the llama context's KV.
      newNPast = calloc<llama_pos>();
      newNPast.value = nPast;
      final evalRc = b.mtmd_helper_eval_chunks(
        pointer,
        llamaContext.pointer,
        chunks,
        nPast,
        seqId,
        nBatch,
        logitsLast,
        newNPast,
      );
      if (evalRc != 0) {
        throw MultimodalException(
          'mtmd_helper_eval_chunks failed: rc=$evalRc',
        );
      }
      return newNPast.value;
    } finally {
      if (newNPast != null) calloc.free(newNPast);
      if (textStruct != null) calloc.free(textStruct);
      if (promptPtr != null) calloc.free(promptPtr);
      if (chunks != null) b.mtmd_input_chunks_free(chunks);
      if (bitmapArray != null) calloc.free(bitmapArray);
      for (final bm in bitmaps) {
        b.mtmd_bitmap_free(bm);
      }
      for (final p in byteArenas) {
        calloc.free(p);
      }
      for (final p in idArenas) {
        calloc.free(p);
      }
    }
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    LlamaLibrary.bindings.mtmd_free(_ptr);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('MultimodalContext has been disposed.');
    }
  }
}
