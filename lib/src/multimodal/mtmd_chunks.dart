import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';

/// Kind of payload an [MtmdChunk] carries.
enum MtmdChunkKind {
  /// A run of regular text tokens (the prompt + any text between media markers).
  text,

  /// A pre-encoded image patch sequence.
  image,

  /// A pre-encoded audio frame sequence.
  audio,
}

/// Borrowed read-only view over one `mtmd_input_chunk`.
///
/// Lifetime is tied to the owning [MtmdChunks]; do not retain a chunk
/// after the chunks list is disposed.
final class MtmdChunk {
  final Pointer<mtmd_input_chunk> _ptr;
  const MtmdChunk._(this._ptr);

  /// Raw FFI pointer for handoff into low-level calls.
  Pointer<mtmd_input_chunk> get pointer => _ptr;

  /// Payload kind.
  MtmdChunkKind get kind {
    final v = LlamaLibrary.bindings.mtmd_input_chunk_get_type(_ptr);
    return switch (v) {
      mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_TEXT => MtmdChunkKind.text,
      mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_IMAGE => MtmdChunkKind.image,
      mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_AUDIO => MtmdChunkKind.audio,
    };
  }

  /// Number of llama tokens this chunk contributes to the context.
  int get nTokens => LlamaLibrary.bindings.mtmd_input_chunk_get_n_tokens(_ptr);

  /// Number of position slots this chunk advances by (may differ from
  /// [nTokens] for mrope / patchified image inputs).
  int get nPos => LlamaLibrary.bindings.mtmd_input_chunk_get_n_pos(_ptr);

  /// Caller-supplied id (typically copied from the source bitmap). Empty
  /// when unset.
  String get id {
    final ptr = LlamaLibrary.bindings.mtmd_input_chunk_get_id(_ptr);
    if (ptr == nullptr) return '';
    return ptr.cast<Utf8>().toDartString();
  }

  /// For [MtmdChunkKind.text] chunks: the llama token ids. Returns an
  /// empty list for image / audio chunks (their tokens are opaque image
  /// embeddings, not vocab ids).
  Int32List textTokens() {
    if (kind != MtmdChunkKind.text) return Int32List(0);
    final nOut = calloc<Size>();
    try {
      final ptr = LlamaLibrary.bindings.mtmd_input_chunk_get_tokens_text(
        _ptr,
        nOut,
      );
      final n = nOut.value;
      if (ptr == nullptr || n == 0) return Int32List(0);
      return Int32List.fromList(ptr.asTypedList(n));
    } finally {
      calloc.free(nOut);
    }
  }

  /// Deep-copy this chunk into a new owned handle. Useful when you need
  /// to keep one chunk alive after the parent list is freed.
  MtmdChunkOwned copy() {
    final raw = LlamaLibrary.bindings.mtmd_input_chunk_copy(_ptr);
    if (raw == nullptr) {
      throw StateError('mtmd_input_chunk_copy returned null');
    }
    return MtmdChunkOwned._(raw);
  }
}

/// Owned counterpart to [MtmdChunk] — returned by [MtmdChunk.copy] when
/// you need a chunk that outlives its parent [MtmdChunks].
final class MtmdChunkOwned implements Finalizable {
  final Pointer<mtmd_input_chunk> _ptr;
  bool _disposed = false;

  MtmdChunkOwned._(this._ptr);

  /// Borrowed view at this owned chunk.
  MtmdChunk get view {
    _ensureAlive();
    return MtmdChunk._(_ptr);
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    LlamaLibrary.bindings.mtmd_input_chunk_free(_ptr);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('MtmdChunkOwned has been disposed.');
    }
  }
}

/// Owned wrapper around an `mtmd_input_chunks` list. Produced by
/// `MultimodalContext.tokenize`. Index or iterate to inspect each chunk;
/// pass to `MultimodalContext.evalChunksList` to drive a llama context.
final class MtmdChunks implements Finalizable {
  final Pointer<mtmd_input_chunks> _ptr;
  bool _disposed = false;

  /// Construct from a raw `mtmd_input_chunks*`. Takes ownership.
  MtmdChunks.takeOwnership(this._ptr);

  /// Raw pointer for handoff. Throws if [dispose] has run.
  Pointer<mtmd_input_chunks> get pointer {
    _ensureAlive();
    return _ptr;
  }

  /// Total chunk count.
  int get length => LlamaLibrary.bindings.mtmd_input_chunks_size(pointer);

  /// Borrowed view of chunk [index].
  MtmdChunk operator [](int index) {
    final raw = LlamaLibrary.bindings.mtmd_input_chunks_get(pointer, index);
    if (raw == nullptr) {
      throw RangeError.range(index, 0, length - 1, 'index');
    }
    return MtmdChunk._(raw);
  }

  /// Iterate the chunks in order, yielding borrowed views.
  Iterable<MtmdChunk> get views sync* {
    final n = length;
    for (var i = 0; i < n; i++) {
      yield this[i];
    }
  }

  /// Sum of `n_tokens` across every chunk. Mirrors `mtmd_helper_get_n_tokens`.
  int get totalTokens =>
      LlamaLibrary.bindings.mtmd_helper_get_n_tokens(pointer);

  /// Sum of `n_pos` across every chunk. Mirrors `mtmd_helper_get_n_pos`.
  int get totalPositions =>
      LlamaLibrary.bindings.mtmd_helper_get_n_pos(pointer);

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    LlamaLibrary.bindings.mtmd_input_chunks_free(_ptr);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('MtmdChunks has been disposed.');
    }
  }
}
