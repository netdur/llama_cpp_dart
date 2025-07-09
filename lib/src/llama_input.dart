import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:llama_cpp_dart/src/llama_cpp.dart';

/// An abstract class representing a multimodal input for the Llama model.
abstract class LlamaInput {}

// A record to hold the FFI pointers for a bitmap, ensuring they are managed together.
typedef BitmapPointers = ({
  Pointer<mtmd_bitmap> bitmap,
  Pointer<Uint8> imageData
});

/// Represents an image input for the Llama model.
///
/// This class is designed to be safely sent between Dart isolates. It stores
/// either a file path or raw image bytes. The actual file reading and native
/// conversion is deferred until it's processed inside the Llama instance.
class LlamaImage extends LlamaInput {
  // The LlamaImage will hold EITHER a path OR bytes, but not both.
  final String? _path;
  final Uint8List? _bytes;

  /// Creates a LlamaImage from a file. This is the preferred method when
  /// working with isolates as it only sends the file path string, not the
  /// full image data.
  LlamaImage.fromFile(File file)
      : _path = file.path,
        _bytes = null;

  /// Creates a LlamaImage from raw image bytes (e.g., from a network request).
  /// When sent to another isolate, this will cause the entire byte array to be copied.
  LlamaImage.fromBytes(Uint8List bytes)
      : _bytes = bytes,
        _path = null;

  /// Internal method to get the raw image bytes, either by reading the file
  /// or using the provided byte list.
  Uint8List _getImageBytes() {
    if (_path != null) {
      return File(_path).readAsBytesSync();
    }
    if (_bytes != null) {
      return _bytes;
    }
    // This should be unreachable due to the constructors.
    throw StateError('LlamaImage is empty. Use fromFile() or fromBytes().');
  }

  /// Internal method to convert the image data into a native `mtmd_bitmap`.
  /// This is called by the Llama class and handles the FFI conversion.
  ///
  /// This method is safe to be called inside any isolate, as it performs
  /// its own I/O if necessary.
  BitmapPointers toBitmap(llama_cpp lib, Allocator allocator) {
    // Get the raw bytes using our new internal method.
    final imageBytes = _getImageBytes();

    final decodedImage = img.decodeImage(imageBytes);
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
