import 'package:ffi/ffi.dart';
import 'package:llama_cpp_dart/llama_cpp.dart';

class Output {
  late llm_output native;

  Output();

  factory Output.fromNative(llm_output native) {
    Output output = Output();
    output.native = native;
    return output;
  }

  llm_output get() => native;

  String get text => native.text.cast<Utf8>().toDartString();
  int get token => native.token;
  bool get hasNext => native.has_next == 1;

  void dispose() {
    calloc.free(native.text);
  }
}
