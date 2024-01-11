import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'llama_cpp.dart';

class LLM {
  late llama_cpp _lib;
  late int _pointer;

  LLM() {
    _lib = llama_cpp(DynamicLibrary.process());
  }

  llm_inference_parameters createInferenceParameters() {
    return _lib.llm_create_inference_parameters();
  }

  llm_gpt_params createGPTParams() {
    return _lib.llm_create_gpt_params();
  }

  void loadModel(String path) {
    llm_gpt_params parameters = _lib.llm_create_gpt_params();
    parameters.model = path.toNativeUtf8().cast<Char>();
    _pointer = _lib.llm_load_model(parameters);
  }

  void unloadModel() {
    _lib.llm_unload_model(_pointer);
  }

  String? getText(String prompt) {
    Pointer<Char> text =
        _lib.llm_get_text(_pointer, prompt.toNativeUtf8().cast<Char>());
    String? result = pointerToString(text);
    malloc.free(text);
    return result;
  }

  void setTextIter(String prompt) {
    _lib.llm_set_text_iter(_pointer, prompt.toNativeUtf8().cast<Char>());
  }

  (String?, bool, int) getNext() {
    Pointer<llm_output> output = _lib.llm_get_next(_pointer);
    String? text = pointerToString(output.ref.text);
    bool hasNext = output.ref.has_next == 1;
    int token = output.ref.token;
    malloc.free(output);
    return (text, hasNext, token);
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
