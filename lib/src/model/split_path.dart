import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../ffi/library_loader.dart';

/// Helpers for split-gguf filenames (`-NNNNN-of-MMMMM.gguf`).
///
/// llama.cpp lets you ship a large quant as multiple `.gguf` files plus an
/// auto-discovered index. These functions translate between the prefix you
/// pass to the model loader and the per-shard filenames on disk.
final class SplitPath {
  SplitPath._();

  /// Compose the per-shard filename for shard [splitNo] of [splitCount].
  ///
  /// Example:
  /// ```
  /// SplitPath.compose(prefix: '/models/llama-7b-q4_0',
  ///                   splitNo: 2, splitCount: 4)
  ///   => '/models/llama-7b-q4_0-00002-of-00004.gguf'
  /// ```
  ///
  /// [maxLen] sizes the output buffer; default fits typical paths.
  static String compose({
    required String prefix,
    required int splitNo,
    required int splitCount,
    int maxLen = 1024,
  }) {
    final prefixPtr = prefix.toNativeUtf8();
    final outPtr = calloc<Char>(maxLen);
    try {
      final n = LlamaLibrary.bindings.llama_split_path(
        outPtr,
        maxLen,
        prefixPtr.cast<Char>(),
        splitNo,
        splitCount,
      );
      if (n <= 0) return '';
      return outPtr.cast<Utf8>().toDartString(length: n);
    } finally {
      calloc.free(prefixPtr);
      calloc.free(outPtr);
    }
  }

  /// Recover the path prefix from a [splitPath] when the shard counters
  /// are known. Returns `null` if the path does not match the requested
  /// [splitNo] / [splitCount].
  static String? decomposePrefix({
    required String splitPath,
    required int splitNo,
    required int splitCount,
    int maxLen = 1024,
  }) {
    final inPtr = splitPath.toNativeUtf8();
    final outPtr = calloc<Char>(maxLen);
    try {
      final n = LlamaLibrary.bindings.llama_split_prefix(
        outPtr,
        maxLen,
        inPtr.cast<Char>(),
        splitNo,
        splitCount,
      );
      if (n <= 0) return null;
      return outPtr.cast<Utf8>().toDartString(length: n);
    } finally {
      calloc.free(inPtr);
      calloc.free(outPtr);
    }
  }
}
