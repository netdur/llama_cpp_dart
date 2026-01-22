import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'llama_cpp.dart';
import 'llama_types.dart';
import 'service/state_codec.dart';

class LlamaStateRestore {
  final int nPos;
  final int nKeep;

  LlamaStateRestore({required this.nPos, required this.nKeep});
}

class LlamaSessionLoadResult {
  final bool ok;
  final int nPos;
  final int nKeep;

  LlamaSessionLoadResult({
    required this.ok,
    required this.nPos,
    required this.nKeep,
  });
}

class LlamaStateIO {
  static const int _headerSize = 16;
  static const int _magic = 0x4C4C5346; // "LLSF"
  static const int _version = 1;

  static Uint8List saveState({
    required llama_cpp lib,
    required Pointer<llama_context> context,
    required int nPos,
    required int nKeep,
  }) {
    final int stateSize = lib.llama_get_state_size(context);
    final int totalSize = stateSize + _headerSize;

    final ptr = malloc<Uint8>(totalSize);

    try {
      final headerData = ptr.asTypedList(_headerSize).buffer.asByteData();
      headerData.setUint32(0, _magic, Endian.little);
      headerData.setUint32(4, _version, Endian.little);
      headerData.setUint32(8, nPos, Endian.little);
      headerData.setUint32(12, nKeep, Endian.little);

      final dataPtr = Pointer<Uint8>.fromAddress(ptr.address + _headerSize);
      lib.llama_copy_state_data(context, dataPtr);

      return Uint8List.fromList(ptr.asTypedList(totalSize));
    } finally {
      malloc.free(ptr);
    }
  }

  static LlamaStateRestore loadState({
    required llama_cpp lib,
    required Pointer<llama_context> context,
    required Uint8List stateData,
  }) {
    if (stateData.length < _headerSize) {
      throw LlamaException('State data too short');
    }

    final header = ByteData.sublistView(stateData, 0, _headerSize);
    final magic = header.getUint32(0, Endian.little);
    final version = header.getUint32(4, Endian.little);

    if (magic != _magic || version != _version) {
      throw LlamaException('Invalid state data header');
    }

    final nPos = header.getUint32(8, Endian.little);
    final nKeep = header.getUint32(12, Endian.little);

    final int expectedStateSize = lib.llama_get_state_size(context);
    if (stateData.length - _headerSize != expectedStateSize) {
      // ignore: avoid_print
      print(
          "Warning: State size mismatch. Expected $expectedStateSize, got ${stateData.length - _headerSize}");
    }

    final ptr = malloc<Uint8>(expectedStateSize);

    try {
      final dataView = stateData.sublist(_headerSize);
      ptr.asTypedList(expectedStateSize).setAll(0, dataView);

      lib.llama_set_state_data(context, ptr);
    } finally {
      malloc.free(ptr);
    }

    return LlamaStateRestore(nPos: nPos, nKeep: nKeep);
  }
}

class LlamaSessionIO {
  static void saveSession({
    required llama_cpp lib,
    required Pointer<llama_context> context,
    required String path,
    required int nPos,
    required int nKeep,
    required bool verbose,
  }) {
    final tempPath = "${path}.tmp.${DateTime.now().millisecondsSinceEpoch}";
    final tempPathPtr = tempPath.toNativeUtf8().cast<Char>();

    try {
      // 1. Native Save to Temp
      final result = lib.llama_state_save_file(context, tempPathPtr, nullptr, 0);
      if (!result) {
        throw LlamaException(
            'Failed to save session (llama_state_save_file returned false)');
      }

      final tempFile = File(tempPath);
      final stateData = tempFile.readAsBytesSync();

      // 2. Write Final File with Header
      final file = File(path);
      final wrapped = StateCodec.encode(
        stateData,
        nPos: nPos,
        nKeep: nKeep,
      );

      file.writeAsBytesSync(wrapped, flush: true);

      if (verbose) {
        // ignore: avoid_print
        print("Session saved (Native Wrapper). Size: ${wrapped.length}");
      }

      // Cleanup
      tempFile.deleteSync();
    } finally {
      malloc.free(tempPathPtr);
    }
  }

  static LlamaSessionLoadResult loadSession({
    required llama_cpp lib,
    required Pointer<llama_context> context,
    required String path,
    required bool verbose,
  }) {
    final file = File(path);
    if (!file.existsSync()) {
      return LlamaSessionLoadResult(ok: false, nPos: 0, nKeep: 0);
    }

    // Read entire file
    final bytes = file.readAsBytesSync();
    final decoded = StateCodec.decode(bytes);

    if (decoded.hasHeader) {
      // DART wrapper
      final savedPos = decoded.nPos ?? 0;
      final savedPrompt = decoded.nKeep ?? 0;
      if (verbose) {
        // ignore: avoid_print
        print("Custom header found. Saved nPos=$savedPos");
      }

      // Unwrap Native Blob
      final stateData = decoded.payload;

      // Write to Temp
      final tempPath =
          "${path}.tmp.load.${DateTime.now().millisecondsSinceEpoch}";
      final tempFile = File(tempPath);
      tempFile.writeAsBytesSync(stateData, flush: true);

      final tempPathPtr = tempPath.toNativeUtf8().cast<Char>();
      final countOut = calloc<Size>(1);

      try {
        final result = lib.llama_state_load_file(
            context, tempPathPtr, nullptr, 0, countOut);
        if (verbose) {
          // ignore: avoid_print
          print("Native load result: $result");
        }
        if (result) {
          final restoredPos = countOut.value;
          final resolvedPos = restoredPos != 0 ? restoredPos : savedPos;
          final resolvedKeep =
              savedPrompt != 0 ? savedPrompt : resolvedPos;
          return LlamaSessionLoadResult(
            ok: true,
            nPos: resolvedPos,
            nKeep: resolvedKeep,
          );
        }
        return LlamaSessionLoadResult(ok: false, nPos: 0, nKeep: 0);
      } finally {
        malloc.free(tempPathPtr);
        calloc.free(countOut);
        if (tempFile.existsSync()) tempFile.deleteSync();
      }
    } else {
      // Fallback: Try Raw Native (no wrapper)
      if (verbose) {
        // ignore: avoid_print
        print(
            "No custom header. Attempting raw native load (nPos resets to 0).");
      }

      final pathPtr = path.toNativeUtf8().cast<Char>();
      final countOut = calloc<Size>(1);
      try {
        final result =
            lib.llama_state_load_file(context, pathPtr, nullptr, 0, countOut);
        if (result) {
          int resolvedPos = countOut.value;
          int resolvedKeep = resolvedPos;
          final metaFile = File("$path.meta");
          if (metaFile.existsSync()) {
            try {
              final json = jsonDecode(metaFile.readAsStringSync());
              resolvedPos = resolvedPos != 0 ? resolvedPos : (json["nPos"] ?? 0);
              resolvedKeep = json["nPrompt"] ?? resolvedPos;
            } catch (_) {}
          }
          return LlamaSessionLoadResult(
            ok: true,
            nPos: resolvedPos,
            nKeep: resolvedKeep,
          );
        }
        return LlamaSessionLoadResult(ok: false, nPos: 0, nKeep: 0);
      } finally {
        malloc.free(pathPtr);
        calloc.free(countOut);
      }
    }
  }
}
