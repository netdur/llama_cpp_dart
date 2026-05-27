import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import '../model/model.dart';
import '../types/exceptions.dart';

/// A LoRA adapter loaded against a [LlamaModel].
///
/// Construct with [LlamaLora.load], pass to [LlamaContext.setLoraAdapters]
/// (optionally with a scale), and call [dispose] when done. One adapter can
/// be applied to multiple contexts that share the same model.
///
/// Dispose **after** every context that references it has stopped using it —
/// freeing while applied is undefined behavior in llama.cpp.
final class LlamaLora implements Finalizable {
  /// The model this adapter was loaded against.
  final LlamaModel model;

  /// On-disk path of the LoRA file.
  final String path;

  final Pointer<llama_adapter_lora> _ptr;
  bool _disposed = false;

  LlamaLora._(this.model, this.path, this._ptr);

  /// Load a LoRA `.gguf` from disk into the given [model].
  ///
  /// Throws [LlamaLibraryException] when llama.cpp rejects the file (wrong
  /// architecture, malformed gguf, etc.).
  static LlamaLora load({required LlamaModel model, required String path}) {
    final pathPtr = path.toNativeUtf8();
    try {
      final adapter = LlamaLibrary.bindings.llama_adapter_lora_init(
        model.pointer,
        pathPtr.cast<Char>(),
      );
      if (adapter == nullptr) {
        throw LlamaLibraryException('failed to load LoRA adapter: $path');
      }
      return LlamaLora._(model, path, adapter);
    } finally {
      calloc.free(pathPtr);
    }
  }

  /// Raw pointer for handoff into FFI calls. Throws if [dispose] has run.
  Pointer<llama_adapter_lora> get pointer {
    _ensureAlive();
    return _ptr;
  }

  /// Number of metadata key/value pairs the adapter file exposes.
  int get metaCount =>
      LlamaLibrary.bindings.llama_adapter_meta_count(pointer);

  /// Read the metadata key at [index].
  String? metaKeyAt(int index) =>
      _readString(maxLen: 256, (buf, size) {
        return LlamaLibrary.bindings.llama_adapter_meta_key_by_index(
          pointer,
          index,
          buf,
          size,
        );
      });

  /// Read the metadata value at [index].
  String? metaValueAt(int index) =>
      _readString(maxLen: 1024, (buf, size) {
        return LlamaLibrary.bindings.llama_adapter_meta_val_str_by_index(
          pointer,
          index,
          buf,
          size,
        );
      });

  /// Read the value for metadata [key]. Returns `null` when missing.
  String? metaValue(String key) {
    final keyPtr = key.toNativeUtf8();
    try {
      return _readString(maxLen: 1024, (buf, size) {
        return LlamaLibrary.bindings.llama_adapter_meta_val_str(
          pointer,
          keyPtr.cast<Char>(),
          buf,
          size,
        );
      });
    } finally {
      calloc.free(keyPtr);
    }
  }

  /// All metadata entries as a key/value map.
  Map<String, String> metaEntries() {
    final n = metaCount;
    final out = <String, String>{};
    for (var i = 0; i < n; i++) {
      final k = metaKeyAt(i);
      if (k == null) continue;
      out[k] = metaValueAt(i) ?? '';
    }
    return out;
  }

  /// Trigger tokens the adapter listens for, if it is an Activated LoRA
  /// (aLoRA). Empty for plain LoRAs.
  List<int> get aloraInvocationTokens {
    final n =
        LlamaLibrary.bindings.llama_adapter_get_alora_n_invocation_tokens(
      pointer,
    );
    if (n <= 0) return const <int>[];
    final ptr =
        LlamaLibrary.bindings.llama_adapter_get_alora_invocation_tokens(
      pointer,
    );
    if (ptr == nullptr) return const <int>[];
    return List<int>.generate(n, (i) => ptr[i], growable: false);
  }

  /// Release the adapter. Detach from every context first.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    LlamaLibrary.bindings.llama_adapter_lora_free(_ptr);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('LlamaLora has been disposed.');
    }
  }

  String? _readString(
    int Function(Pointer<Char> buf, int size) call, {
    required int maxLen,
  }) {
    final buf = calloc<Char>(maxLen);
    try {
      final n = call(buf, maxLen);
      if (n < 0) return null;
      if (n == 0) return '';
      return buf.cast<Utf8>().toDartString(length: n);
    } finally {
      calloc.free(buf);
    }
  }
}

/// One entry in a context's active LoRA stack: an adapter and the scale
/// to apply it at. Pass a list of these to [LlamaContext.setLoraAdapters].
final class LoraBinding {
  final LlamaLora adapter;
  final double scale;
  const LoraBinding(this.adapter, {this.scale = 1.0});
}
