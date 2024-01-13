import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'llama_cpp.dart';

class IntArray {
  late llm_int_array native;

  IntArray();

  llm_int_array get() => native;

  factory IntArray.fromNative(llm_int_array native) {
    IntArray output = IntArray();
    output.native = native;
    return output;
  }

  List<int> get data {
    return List<int>.generate(
        native.size, (i) => native.data.elementAt(i).value);
  }

  void dispose() {
    calloc.free(native.data);
  }
}
