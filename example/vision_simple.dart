import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:image/image.dart' as img;
import 'package:llama_cpp_dart/src/llama_cpp.dart';

Future<void> main() async {
  String modelPath = "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf";
  String mmprojPath = "/Users/adel/Workspace/gguf/mmproj-model-f16-4B.gguf";
  String imagePath = "/Users/adel/Downloads/test-1.jpeg";

  String promptText = "\nUSER: Describe this image in detail.\nASSISTANT:";
  int ngl = 99;
  int nPredict = 128;
  int nCtx = 2048;

  final lib = llama_cpp(DynamicLibrary.open(
      "/Users/adel/Workspace/llama_cpp_dart/bin/MAC_ARM64/libmtmd.dylib"));
  lib.llama_backend_init();

  var modelParams = lib.llama_model_default_params();
  modelParams.n_gpu_layers = ngl;
  final modelPathPtr = modelPath.toNativeUtf8().cast<Char>();
  final model = lib.llama_load_model_from_file(modelPathPtr, modelParams);
  malloc.free(modelPathPtr);

  if (model.address == 0) {
    stderr.writeln("error: unable to load model");
    return;
  }
  final vocab = lib.llama_model_get_vocab(model);

  var ctxParams = lib.llama_context_default_params();
  ctxParams.n_ctx = nCtx;
  final ctx = lib.llama_new_context_with_model(model, ctxParams);
  if (ctx.address == 0) {
    stderr.writeln("error: failed to create context");
    return;
  }

  var mparam = lib.mtmd_context_params_default();
  mparam.use_gpu = true;
  final mprojPathPtr = mmprojPath.toNativeUtf8().cast<Char>();
  final mctx = lib.mtmd_init_from_file(mprojPathPtr, model, mparam);
  malloc.free(mprojPathPtr);

  if (mctx.address == 0) {
    stderr.writeln("error: failed to create multimodal projector context");
    lib.llama_free_model(model);
    lib.llama_free(ctx);
    return;
  }

  Pointer<mtmd_bitmap> imageBitmap;
  late Pointer<Uint8> imageDataPtr;
  try {
    final imageFile = File(imagePath);
    if (!imageFile.existsSync()) {
      throw Exception("Image file not found at $imagePath");
    }
    final imageBytes = imageFile.readAsBytesSync();
    final decodedImage = img.decodeImage(imageBytes);
    if (decodedImage == null) {
      throw Exception("Failed to decode image");
    }

    final Uint8List rgbBytes =
        decodedImage.getBytes(order: img.ChannelOrder.rgb);
    imageDataPtr = malloc<Uint8>(rgbBytes.length);
    imageDataPtr.asTypedList(rgbBytes.length).setAll(0, rgbBytes);

    imageBitmap = lib.mtmd_bitmap_init(decodedImage.width, decodedImage.height,
        imageDataPtr.cast<UnsignedChar>());
  } catch (e) {
    stderr.writeln("Error loading image: $e");
    lib.mtmd_free(mctx);
    lib.llama_free(ctx);
    lib.llama_free_model(model);
    return;
  }

  final markerPtr = lib.mtmd_default_marker();
  final marker = markerPtr.cast<Utf8>().toDartString();
  final fullPrompt = marker + promptText;
  final promptPtr = fullPrompt.toNativeUtf8();

  final inputText = calloc<mtmd_input_text>();
  inputText.ref.text = promptPtr.cast<Char>();
  inputText.ref.add_special = true;
  inputText.ref.parse_special = true;

  final chunks = lib.mtmd_input_chunks_init();
  final bitmapArray = malloc<Pointer<mtmd_bitmap>>(1);
  bitmapArray[0] = imageBitmap;

  int ret = lib.mtmd_tokenize(mctx, chunks, inputText, bitmapArray, 1);
  if (ret != 0) {
    stderr.writeln("mtmd_tokenize failed with code: $ret");
    return;
  }

  malloc.free(bitmapArray);
  calloc.free(inputText);
  malloc.free(promptPtr);

  int nPast = 0;
  final nChunks = lib.mtmd_input_chunks_size(chunks);

  stdout.write(fullPrompt);
  await stdout.flush();

  for (int i = 0; i < nChunks; i++) {
    final chunk = lib.mtmd_input_chunks_get(chunks, i);
    final chunkType = lib.mtmd_input_chunk_get_type(chunk);

    if (chunkType == mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      if (lib.mtmd_encode_chunk(mctx, chunk) != 0) {
        stderr.writeln("Failed to encode image chunk");
        return;
      }
      final embd = lib.mtmd_get_output_embd(mctx);
      final nImageTokens = lib.mtmd_input_chunk_get_n_tokens(chunk);

      final batch = lib.llama_batch_init(nImageTokens, 1, 1);
      batch.n_tokens = nImageTokens;
      batch.embd = embd;

      for (int k = 0; k < nImageTokens; k++) {
        batch.pos[k] = nPast + k;
        batch.n_seq_id[k] = 1;
        batch.seq_id[k][0] = 0;
        batch.logits[k] = 0;
      }
      batch.logits[nImageTokens - 1] = 1;

      if (lib.llama_decode(ctx, batch) != 0) {
        stderr.writeln("llama_decode failed on image embeddings");
        lib.llama_batch_free(batch);
        return;
      }
      lib.llama_batch_free(batch);
      nPast += nImageTokens;
    } else if (chunkType == mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_TEXT) {
      final nTokensPtr = malloc<Size>();
      final tokens = lib.mtmd_input_chunk_get_tokens_text(chunk, nTokensPtr);
      final nTokens = nTokensPtr.value;
      malloc.free(nTokensPtr);

      final batch = lib.llama_batch_init(nTokens, 0, 1);
      for (int k = 0; k < nTokens; k++) {
        batch.token[k] = tokens[k];
        batch.pos[k] = nPast + k;
        batch.n_seq_id[k] = 1;
        batch.seq_id[k][0] = 0;
        batch.logits[k] = 0;
      }
      batch.logits[nTokens - 1] = 1;
      batch.n_tokens = nTokens;

      if (lib.llama_decode(ctx, batch) != 0) {
        stderr.writeln("llama_decode failed on text tokens");
        lib.llama_batch_free(batch);
        return;
      }
      lib.llama_batch_free(batch);
      nPast += nTokens;
    }
  }

  lib.mtmd_input_chunks_free(chunks);

  var smplParams = lib.llama_sampler_chain_default_params();
  final smpl = lib.llama_sampler_chain_init(smplParams);
  lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_greedy());

  int nDecode = 0;
  final tMainStart = DateTime.now().microsecondsSinceEpoch;

  while (nPast < nCtx && nDecode < nPredict) {
    final newTokenId = lib.llama_sampler_sample(smpl, ctx, -1);

    if (lib.llama_token_is_eog(vocab, newTokenId)) {
      break;
    }

    final buf = malloc<Char>(128);
    int n = lib.llama_token_to_piece(vocab, newTokenId, buf, 128, 0, false);
    if (n < 0) {
      stderr.writeln("error: failed to convert token to piece");
      malloc.free(buf);
      return;
    }
    String piece = String.fromCharCodes(buf.cast<Uint8>().asTypedList(n));
    stdout.write(piece);
    await stdout.flush();
    malloc.free(buf);

    final batch = lib.llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = newTokenId;
    batch.pos[0] = nPast;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;

    if (lib.llama_decode(ctx, batch) != 0) {
      stderr.writeln("failed to eval next token");
      lib.llama_batch_free(batch);
      return;
    }
    lib.llama_batch_free(batch);

    nPast++;
    nDecode++;
  }

  final tMainEnd = DateTime.now().microsecondsSinceEpoch;
  stdout.writeln();

  final decodeTime = (tMainEnd - tMainStart) / 1000000.0;
  stderr.writeln(
      "decoded $nDecode tokens in ${decodeTime.toStringAsFixed(2)} s, speed: ${(nDecode / decodeTime).toStringAsFixed(2)} t/s");

  // ===================================
  // CORRECTED CLEANUP SEQUENCE
  // ===================================

  // 1. Free the sampler.
  lib.llama_sampler_free(smpl);

  // 2. Free the image-related resources first.
  //    The bitmap is no longer needed after processing.
  lib.mtmd_bitmap_free(imageBitmap);

  // 3. Free the raw image data buffer you allocated. This fixes the memory leak.
  malloc.free(imageDataPtr);

  // 4. Free the multimodal and language model contexts.
  // lib.mtmd_free(mctx);
  lib.llama_free(ctx);

  // 5. NOW it's safe to free the model itself.
  lib.llama_free_model(model);

  // 6. Finally, shut down the backend.
  lib.llama_backend_free();
}
