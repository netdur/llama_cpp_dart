import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'float_array.dart';
import 'int_array.dart';

import 'gpt_params.dart';
import 'llama_cpp.dart';

class LLM {
  late llama_cpp _lib;
  late int _pointer;

  LLM() {
    _lib = llama_cpp(DynamicLibrary.process());
  }

  List<int> decode(String prompt) {
    Pointer<Char> converted = prompt.toNativeUtf8().cast<Char>();
    llm_int_array decoded = _lib.llm_encode(_pointer, converted);
    var array = IntArray.fromNative(decoded);
    calloc.free(converted);
    var data = array.data;
    array.dispose();
    return data;
  }

  List<double> embded(String prompt) {
    Pointer<Char> converted = prompt.toNativeUtf8().cast<Char>();
    llm_float_array features = _lib.llm_embed(_pointer, converted);
    var array = FloatArray.fromNative(features);
    calloc.free(converted);
    var data = array.data;
    array.dispose();
    return data;
  }

  void loadModel(GptParams gptParams) {
    _pointer = _lib.llm_load_model(gptParams.get());
  }

  void unloadModel() {
    _lib.llm_unload_model(_pointer);
  }

  String? getText(String prompt) {
    Pointer<Char> converted = prompt.toNativeUtf8().cast<Char>();
    Pointer<Char> text = _lib.llm_get_text(_pointer, converted);
    calloc.free(converted);
    String? result = pointerToString(text);
    malloc.free(text);
    return result;
  }

  void setTextIter(String prompt) {
    Pointer<Char> converted = prompt.toNativeUtf8().cast<Char>();
    _lib.llm_set_text_iter(_pointer, converted);
    calloc.free(converted);
  }

  (String?, bool, int) getNext() {
    Pointer<llm_output> output = _lib.llm_get_next(_pointer);
    String? text = pointerToString(output.ref.text);
    bool hasNext = output.ref.has_next == 1;
    int token = output.ref.token;
    malloc.free(output);
    return (text, hasNext, token);
  }

  Stream<String> prompt(String initialText) async* {
    setTextIter(initialText);
    while (true) {
      var (text, hasNext, _) = getNext();
      yield text ?? "";
      if (!hasNext) {
        break;
      }
    }
  }

  static String? pointerToString(Pointer<Char> pointer) {
    try {
      final units = <int>[];
      var len = 0;
      while (true) {
        final int char = pointer.cast<Int8>().elementAt(len++).value;
        if (char == 0) break;
        units.add(char);
      }
      return String.fromCharCodes(units);
    } catch (_) {
      return null;
    }
  }
}
