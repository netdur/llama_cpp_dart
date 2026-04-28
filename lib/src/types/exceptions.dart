/// Base type for all exceptions thrown by the llama_cpp_dart binding.
///
/// Abstract rather than sealed so binding subsystems (e.g. chat templates,
/// future multimodal helpers) can introduce their own subtypes from their
/// own files.
abstract class LlamaException implements Exception {
  final String message;
  const LlamaException(this.message);

  @override
  String toString() => '$runtimeType: $message';
}

final class LlamaLibraryException extends LlamaException {
  const LlamaLibraryException(super.message);
}

final class LlamaModelLoadException extends LlamaException {
  const LlamaModelLoadException(super.message);
}

final class LlamaContextException extends LlamaException {
  const LlamaContextException(super.message);
}

final class LlamaTokenizeException extends LlamaException {
  const LlamaTokenizeException(super.message);
}

final class LlamaDecodeException extends LlamaException {
  final int code;
  const LlamaDecodeException(this.code, String message) : super(message);

  @override
  String toString() => 'LlamaDecodeException(code=$code): $message';
}
