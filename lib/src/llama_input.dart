import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:llama_cpp_dart/src/llama_cpp.dart';

/// An abstract class representing a multimodal input for the Llama model.
/// This can be extended to support different types of media like images or audio.
abstract class LlamaInput {}

// A record to hold the FFI pointers for a bitmap, ensuring they are managed together.
typedef BitmapPointers = ({
  Pointer<mtmd_bitmap> bitmap,
  Pointer<Uint8> imageData
});

/// Represents an image input for the Llama model.
class LlamaImage extends LlamaInput {
  final Uint8List _imageBytes;

  /// Creates a LlamaImage from a file path.
  /// Throws an exception if the file doesn't exist.
  LlamaImage.fromFile(File file) : _imageBytes = file.readAsBytesSync();

  /// Creates a LlamaImage from raw image bytes (e.g., PNG, JPEG).
  LlamaImage.fromBytes(Uint8List bytes) : _imageBytes = bytes;

  /// Internal method to convert the image data into a native `mtmd_bitmap`.
  /// This is called by the Llama class and handles the FFI conversion.
  ///
  /// Throws an exception if the image data cannot be decoded.
  /// Returns a record containing pointers to the bitmap and the raw image data,
  /// both of which must be freed by the caller.
  BitmapPointers toBitmap(llama_cpp lib, Allocator allocator) {
    final decodedImage = img.decodeImage(_imageBytes);
    if (decodedImage == null) {
      throw Exception(
          "Failed to decode image. Ensure it's a valid format (PNG, JPEG, etc.).");
    }

    // Convert the image to RGB format, which is what the model expects.
    final Uint8List rgbBytes =
        decodedImage.getBytes(order: img.ChannelOrder.rgb);

    // Allocate memory for the image data and copy it.
    final imageDataPtr = allocator<Uint8>(rgbBytes.length);
    imageDataPtr.asTypedList(rgbBytes.length).setAll(0, rgbBytes);

    // Create the multimodal bitmap structure.
    final imageBitmap = lib.mtmd_bitmap_init(decodedImage.width,
        decodedImage.height, imageDataPtr.cast<UnsignedChar>());

    return (bitmap: imageBitmap, imageData: imageDataPtr);
  }
}
