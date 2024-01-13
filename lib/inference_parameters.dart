import 'dart:convert';
import 'dart:ffi';
import 'package:ffi/ffi.dart';

import 'llama_cpp.dart';

class InferenceParameters {
  late llama_cpp _lib;
  late llm_inference_parameters _parameters;

  InferenceParameters() {
    _lib = llama_cpp(DynamicLibrary.process());
    _parameters = _lib.llm_create_inference_parameters();
  }

  llm_inference_parameters get() => _parameters;

  int get nPrev => _parameters.n_prev;
  set nPrev(int value) => _parameters.n_prev = value;

  int get nProbs => _parameters.n_probs;
  set nProbs(int value) => _parameters.n_probs = value;

  int get topK => _parameters.top_k;
  set topK(int value) => _parameters.top_k = value;

  double get topP => _parameters.top_p;
  set topP(double value) => _parameters.top_p = value;

  double get minP => _parameters.min_p;
  set minP(double value) => _parameters.min_p = value;

  double get tfsZ => _parameters.tfs_z;
  set tfsZ(double value) => _parameters.tfs_z = value;

  double get typicalP => _parameters.typical_p;
  set typicalP(double value) => _parameters.typical_p = value;

  double get temp => _parameters.temp;
  set temp(double value) => _parameters.temp = value;

  int get penaltyLastN => _parameters.penalty_last_n;
  set penaltyLastN(int value) => _parameters.penalty_last_n = value;

  double get penaltyRepeat => _parameters.penalty_repeat;
  set penaltyRepeat(double value) => _parameters.penalty_repeat = value;

  double get penaltyFreq => _parameters.penalty_freq;
  set penaltyFreq(double value) => _parameters.penalty_freq = value;

  double get penaltyPresent => _parameters.penalty_present;
  set penaltyPresent(double value) => _parameters.penalty_present = value;

  int get mirostat => _parameters.mirostat;
  set mirostat(int value) => _parameters.mirostat = value;

  double get mirostatTau => _parameters.mirostat_tau;
  set mirostatTau(double value) => _parameters.mirostat_tau = value;

  double get mirostatEta => _parameters.mirostat_eta;
  set mirostatEta(double value) => _parameters.mirostat_eta = value;

  bool get penalizeNL => _parameters.penalize_nl == 1;
  set penalizeNL(bool value) => _parameters.penalize_nl = value ? 1 : 0;

  String get samplersSequence {
    final bytes =
        List<int>.generate(6, (i) => _parameters.samplers_sequence[i]);
    return utf8.decode(bytes);
  }

  set samplersSequence(String value) {
    final units = utf8.encode(value).take(6).toList();
    for (int i = 0; i < units.length; i++) {
      _parameters.samplers_sequence[i] = units[i];
    }
    _parameters.samplers_sequence[6] = 0;
  }

  String get grammar {
    return _parameters.grammar.cast<Utf8>().toDartString();
  }

  set grammar(String value) {
    _parameters.grammar = value.toNativeUtf8().cast<Char>();
  }

  String get cfgNegativePrompt {
    return _parameters.cfg_negative_prompt.cast<Utf8>().toDartString();
  }

  set cfgNegativePrompt(String value) {
    _parameters.cfg_negative_prompt = value.toNativeUtf8().cast<Char>();
  }

  double get cfgScale => _parameters.cfg_scale;
  set cfgScale(double value) => _parameters.cfg_scale = value;

  bool get usePenaltyPromptTokens => _parameters.use_penalty_prompt_tokens == 1;
  set usePenaltyPromptTokens(bool value) =>
      _parameters.use_penalty_prompt_tokens = value ? 1 : 0;

  void dispose() {
    calloc.free(_parameters.grammar);
    calloc.free(_parameters.cfg_negative_prompt);
  }
}
