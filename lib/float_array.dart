import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'llama_cpp.dart';

class FloatArray {
  late llm_float_array native;

  FloatArray();

  llm_float_array get() => native;

  factory FloatArray.fromNative(llm_float_array native) {
    FloatArray output = FloatArray();
    output.native = native;
    return output;
  }

  List<double> get data {
    return List<double>.generate(
        native.size, (i) => native.data.elementAt(i).value);
  }

  void dispose() {
    calloc.free(native.data);
  }
}
