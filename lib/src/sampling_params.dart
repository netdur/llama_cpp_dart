// ignore_for_file: non_constant_identifier_names, camel_case_types

import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

class SamplingParams {
  int nPrev = 64;
  int nProbs = 0;
  int topK = 40;
  double topP = 0.95;
  double minP = 0.05;
  double tfsZ = 1.00;
  double typicalP = 1.00;
  double temp = 0.80;
  int penaltyLastN = 64;
  double penaltyRepeat = 1.10;
  double penaltyFreq = 0.00;
  double penaltyPresent = 0.00;
  int mirostat = 0;
  double mirostatTau = 5.00;
  double mirostatEta = 0.10;
  bool penalizeNl = true;
  String samplersSequence = "kfypmt";
  String grammar = '';
  String cfgNegativePrompt = '';
  double cfgScale = 1.0;
  Map<int, double> logitBias = {};
  List<int> penaltyPromptTokens = [];
  bool usePenaltyPromptTokens = false;

  SamplingParams();

  SamplingParams.fromJson(Map<String, dynamic> json) {
    nPrev = json['nPrev'] ?? nPrev;
    nProbs = json['nProbs'] ?? nProbs;
    topK = json['topK'] ?? topK;
    topP = json['topP'] ?? topP;
    minP = json['minP'] ?? minP;
    tfsZ = json['tfsZ'] ?? tfsZ;
    typicalP = json['typicalP'] ?? typicalP;
    temp = json['temp'] ?? temp;
    penaltyLastN = json['penaltyLastN'] ?? penaltyLastN;
    penaltyRepeat = json['penaltyRepeat'] ?? penaltyRepeat;
    penaltyFreq = json['penaltyFreq'] ?? penaltyFreq;
    penaltyPresent = json['penaltyPresent'] ?? penaltyPresent;
    mirostat = json['mirostat'] ?? mirostat;
    mirostatTau = json['mirostatTau'] ?? mirostatTau;
    mirostatEta = json['mirostatEta'] ?? mirostatEta;
    penalizeNl = json['penalizeNl'] ?? penalizeNl;
    samplersSequence = json['samplersSequence'] ?? samplersSequence;
    grammar = json['grammar'] ?? grammar;
    cfgNegativePrompt = json['cfgNegativePrompt'] ?? cfgNegativePrompt;
    cfgScale = json['cfgScale'] ?? cfgScale;
    logitBias = json['logitBias']?.cast<int, double>() ?? logitBias;
    penaltyPromptTokens =
        json['penaltyPromptTokens']?.cast<int>() ?? penaltyPromptTokens;
    usePenaltyPromptTokens =
        json['usePenaltyPromptTokens'] ?? usePenaltyPromptTokens;
  }

  Map<String, dynamic> toJson() {
    return {
      'nPrev': nPrev,
      'nProbs': nProbs,
      'topK': topK,
      'topP': topP,
      'minP': minP,
      'tfsZ': tfsZ,
      'typicalP': typicalP,
      'temp': temp,
      'penaltyLastN': penaltyLastN,
      'penaltyRepeat': penaltyRepeat,
      'penaltyFreq': penaltyFreq,
      'penaltyPresent': penaltyPresent,
      'mirostat': mirostat,
      'mirostatTau': mirostatTau,
      'mirostatEta': mirostatEta,
      'penalizeNl': penalizeNl,
      'samplersSequence': samplersSequence,
      'grammar': grammar,
      'cfgNegativePrompt': cfgNegativePrompt,
      'cfgScale': cfgScale,
      'logitBias': logitBias,
      'penaltyPromptTokens': penaltyPromptTokens,
      'usePenaltyPromptTokens': usePenaltyPromptTokens
    };
  }

  Pointer<llama_sampling_params> get() {
    final Pointer<llama_sampling_params> ptr =
        malloc.allocate<llama_sampling_params>(sizeOf<llama_sampling_params>());

    ptr.ref
      ..n_prev = nPrev
      ..n_probs = nProbs
      ..top_k = topK
      ..top_p = topP
      ..min_p = minP
      ..tfs_z = tfsZ
      ..typical_p = typicalP
      ..temp = temp
      ..penalty_last_n = penaltyLastN
      ..penalty_repeat = penaltyRepeat
      ..penalty_freq = penaltyFreq
      ..penalty_present = penaltyPresent
      ..mirostat = mirostat
      ..mirostat_tau = mirostatTau
      ..mirostat_eta = mirostatEta
      ..penalize_nl = penalizeNl
      ..cfg_scale = cfgScale
      ..samplers_sequence = samplersSequence.toNativeUtf8()
      ..grammar = grammar.toNativeUtf8()
      ..cfg_negative_prompt = cfgNegativePrompt.toNativeUtf8()
      ..cfg_scale = cfgScale
      ..use_penalty_prompt_tokens = usePenaltyPromptTokens;

    ptr.ref.penalty_prompt_tokens_length = penaltyPromptTokens.length;
    ptr.ref.penalty_prompt_tokens = calloc<Int32>(penaltyPromptTokens.length);
    final tokenList =
        ptr.ref.penalty_prompt_tokens.asTypedList(penaltyPromptTokens.length);
    tokenList.setAll(0, penaltyPromptTokens);

    ptr.ref.logit_bias_length = logitBias.length;
    ptr.ref.logit_bias_ints = calloc<Int32>(logitBias.length);
    ptr.ref.logit_bias_floats = calloc<Float>(logitBias.length);

    Int32List logitBiasInts =
        ptr.ref.logit_bias_ints.asTypedList(logitBias.length);
    Float32List logitBiasFloats =
        ptr.ref.logit_bias_floats.asTypedList(logitBias.length);

    int i = 0;
    logitBias.forEach((key, value) {
      logitBiasInts[i] = key;
      logitBiasFloats[i] = value;
      i++;
    });

    return ptr;
  }
}

final class llama_sampling_params extends Struct {
  @Int32()
  external int n_prev;

  @Int32()
  external int n_probs;

  @Int32()
  external int top_k;

  @Float()
  external double top_p;

  @Float()
  external double min_p;

  @Float()
  external double tfs_z;

  @Float()
  external double typical_p;

  @Float()
  external double temp;

  @Int32()
  external int penalty_last_n;

  @Float()
  external double penalty_repeat;

  @Float()
  external double penalty_freq;

  @Float()
  external double penalty_present;

  @Int32()
  external int mirostat;

  @Float()
  external double mirostat_tau;

  @Float()
  external double mirostat_eta;

  @Bool()
  external bool penalize_nl;

  external Pointer<Utf8> samplers_sequence;

  external Pointer<Utf8> grammar;

  external Pointer<Utf8> cfg_negative_prompt;

  @Double()
  external double cfg_scale;

  external Pointer<Int32> logit_bias_ints;

  external Pointer<Float> logit_bias_floats;

  @Int32()
  external int logit_bias_length;

  external Pointer<Int32> penalty_prompt_tokens;

  @Int32()
  external int penalty_prompt_tokens_length;

  @Bool()
  external bool use_penalty_prompt_tokens;

  void free() {
    if (samplers_sequence.address != 0) {
      calloc.free(samplers_sequence);
    }
    if (grammar.address != 0) {
      calloc.free(grammar);
    }
    if (cfg_negative_prompt.address != 0) {
      calloc.free(cfg_negative_prompt);
    }
    if (penalty_prompt_tokens.address != 0) {
      calloc.free(penalty_prompt_tokens);
    }
    if (logit_bias_ints.address != 0) {
      calloc.free(logit_bias_ints);
    }
    if (logit_bias_floats.address != 0) {
      calloc.free(logit_bias_floats);
    }
  }
}
