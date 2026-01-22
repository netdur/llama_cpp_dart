import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../llama_types.dart';
import '../llama_cpp.dart';
import '../llama_input.dart';
import 'pending_item.dart';

class VisionBuildResult {
  final List<PendingItem> items;
  final int totalPromptTokens;

  VisionBuildResult(this.items, this.totalPromptTokens);
}

class VisionHelper {
  static VisionBuildResult buildPendingItems({
    required llama_cpp lib,
    required Pointer<mtmd_context> mctx,
    required Pointer<llama_model> model,
    required String prompt,
    required List<LlamaInput> inputs,
  }) {
    final images = inputs.whereType<LlamaImage>().toList();
    const marker = '<image>';

    if (marker.allMatches(prompt).length != images.length) {
      throw ArgumentError(
          "Mismatch between <image> markers and provided inputs");
    }

    final bitmapRefs = <BitmapPointers>[];
    Pointer<Pointer<mtmd_bitmap>>? bmpArr;

    Pointer<mtmd_input_chunks> chunks = nullptr;
    Pointer<mtmd_input_text> txtPtr = nullptr;
    Pointer<Char> fullPtr = nullptr;

    final items = <PendingItem>[];
    int totalPromptTokens = 0;

    try {
      for (final img in images) {
        bitmapRefs.add(img.toBitmap(lib, malloc));
      }

      bmpArr = malloc<Pointer<mtmd_bitmap>>(bitmapRefs.length);
      for (var i = 0; i < bitmapRefs.length; ++i) {
        bmpArr[i] = bitmapRefs[i].bitmap;
      }

      final modelMark = lib.mtmd_default_marker().cast<Utf8>().toDartString();
      final fullPrompt = prompt.replaceAll(marker, modelMark);
      fullPtr = fullPrompt.toNativeUtf8().cast<Char>();

      txtPtr = calloc<mtmd_input_text>();
      txtPtr.ref
        ..text = fullPtr
        ..add_special = true
        ..parse_special = true;

      chunks = lib.mtmd_input_chunks_init();

      final tk = lib.mtmd_tokenize(mctx, chunks, txtPtr, bmpArr, bitmapRefs.length);
      if (tk != 0) throw LlamaException('mtmd_tokenize failed ($tk)');

      final nChunks = lib.mtmd_input_chunks_size(chunks);

      for (var i = 0; i < nChunks; ++i) {
        final chunk = lib.mtmd_input_chunks_get(chunks, i);
        final type = lib.mtmd_input_chunk_get_type(chunk);

        if (type == mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_IMAGE) {
          if (lib.mtmd_encode_chunk(mctx, chunk) != 0) {
            throw LlamaException('encode image failed');
          }
          final embdPtr = lib.mtmd_get_output_embd(mctx);
          final nTok = lib.mtmd_input_chunk_get_n_tokens(chunk);
          final nEmbd = lib.llama_n_embd(model);

          totalPromptTokens += nTok;

          final totalFloats = nTok * nEmbd;
          final nativeStore = malloc<Float>(totalFloats);

          final srcList = embdPtr.asTypedList(totalFloats);
          final dstList = nativeStore.asTypedList(totalFloats);
          dstList.setAll(0, srcList);

          items.add(PendingItem.embedding(nativeStore, nTok));
        } else {
          final nPtr = malloc<Size>();
          final tokPt = lib.mtmd_input_chunk_get_tokens_text(chunk, nPtr);
          final nTok = nPtr.value;
          malloc.free(nPtr);

          totalPromptTokens += nTok;

          for (int k = 0; k < nTok; k++) {
            items.add(PendingItem.token(tokPt[k]));
          }
        }
      }
    } catch (e) {
      for (final item in items) {
        item.dispose();
      }
      rethrow;
    } finally {
      if (chunks != nullptr) lib.mtmd_input_chunks_free(chunks);
      if (bmpArr != null) malloc.free(bmpArr);
      for (final r in bitmapRefs) {
        if (r.bitmap != nullptr) lib.mtmd_bitmap_free(r.bitmap);
      }
      if (txtPtr != nullptr) calloc.free(txtPtr);
      if (fullPtr != nullptr) malloc.free(fullPtr);
    }

    return VisionBuildResult(items, totalPromptTokens);
  }
}
