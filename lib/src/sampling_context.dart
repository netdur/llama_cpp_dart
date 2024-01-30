// ignore_for_file: camel_case_types

import 'dart:ffi';
import 'dart:math';
import 'package:ffi/ffi.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

import 'llama_cpp.dart';
import 'sampling_params.dart';

final class llama_sampling_context extends Struct {
  external llama_sampling_params params;
}

class SamplingContext {
  List<Pointer<llama_token>> _prev = [];
  final List<Pointer<llama_token_data>> _cur = [];

  SamplingParams? params;
  final Llama llama;

  SamplingContext(this.llama) {
    // result->grammar = nullptr;
    // _prev = List.generate(samplingParams.nPrev,
    //    (index) => calloc.allocate<llama_token>(sizeOf<llama_token>()));
  }

  void dispose() {
    // free grammae
  }

  void reset() {
    if (params != null) {
      _prev = List.generate(params!.nPrev,
          (index) => calloc.allocate<llama_token>(sizeOf<llama_token>()));
    }
    _cur.clear();
  }

  int last() {
    return _prev.last.value;
  }

  String prev(Llama llama, int n) {
    int size = _prev.length;
    n = min(n, size);
    String result = "";
    for (int i = size - n; i < size; i++) {
      result += llama.tokenToPiece(_prev[i].value);
    }
    return result;
  }

  void tfsZ(Pointer<llama_token_data_array> curP, int minKeep, int nVocab) {
    if (params == null) {
      throw Exception("sampling params is required");
    }
    var samplingParams = params!;

    var temp = samplingParams.temp;
    // double         dynatemp_range    = params.dynatemp_range;
    // const float         dynatemp_exponent = params.dynatemp_exponent;
    var topK = samplingParams.topK <= 0 ? nVocab : samplingParams.topK;
    var topP = samplingParams.topP;
    var minP = samplingParams.minP;
    var tfsZ = samplingParams.tfsZ;
    var typicalP = samplingParams.typicalP;
    var samplersSequence = samplingParams.samplersSequence;

    // for (auto s : samplersSequence) {
    for (var i = 0; i < samplersSequence.length; i++) {
      var s = samplersSequence[i];
      switch (s) {
        case 'k':
          llama.l.llama_sample_top_k(llama.context, curP, topK, minKeep);
          break;
        case 'f':
          llama.l.llama_sample_tail_free(llama.context, curP, tfsZ, minKeep);
          break;
        case 'y':
          llama.l.llama_sample_typical(llama.context, curP, typicalP, minKeep);
          break;
        case 'p':
          llama.l.llama_sample_top_p(llama.context, curP, topP, minKeep);
          break;
        case 'm':
          llama.l.llama_sample_min_p(llama.context, curP, minP, minKeep);
          break;
        case 't':
          /*
                if (dynatemp_range > 0) {
                    float dynatemp_min = std::max(0.0f, temp - dynatemp_range);
                    float dynatemp_max = std::max(0.0f, temp + dynatemp_range);
                    llama.l.llama_sample_entropy(llama.context, curP, dynatemp_min, dynatemp_max, dynatemp_exponent);
                } else {
                    llama.l.llama_sample_temp(llama.context, curP, temp);
                }
                */
          llama.l.llama_sample_temp(llama.context, curP, temp);
          break;
        default:
          break;
      }
    }
  }

  void queue(Pointer<llama_token_data_array> curP, int minKeep) {
    if (params == null) {
      throw Exception("sampling params is required");
    }
    var samplingParams = params!;

    int nVocab = llama.l.llama_n_vocab(llama.model);

    var temp = samplingParams.temp;
    // double         dynatemp_range    = params.dynatemp_range;
    // const float         dynatemp_exponent = params.dynatemp_exponent;
    var topK = samplingParams.topK <= 0 ? nVocab : samplingParams.topK;
    var topP = samplingParams.topP;
    var minP = samplingParams.minP;
    var tfsZ = samplingParams.tfsZ;
    var typicalP = samplingParams.typicalP;
    var samplersSequence = samplingParams.samplersSequence;

    // for (auto s : samplersSequence) {
    for (var i = 0; i < samplersSequence.length; i++) {
      var s = samplersSequence[i];
      switch (s) {
        case 'k':
          llama.l.llama_sample_top_k(llama.context, curP, topK, minKeep);
          break;
        case 'f':
          llama.l.llama_sample_tail_free(llama.context, curP, tfsZ, minKeep);
          break;
        case 'y':
          llama.l.llama_sample_typical(llama.context, curP, typicalP, minKeep);
          break;
        case 'p':
          llama.l.llama_sample_top_p(llama.context, curP, topP, minKeep);
          break;
        case 'm':
          llama.l.llama_sample_min_p(llama.context, curP, minP, minKeep);
          break;
        case 't':
          /*
                if (dynatemp_range > 0) {
                    float dynatemp_min = std::max(0.0f, temp - dynatemp_range);
                    float dynatemp_max = std::max(0.0f, temp + dynatemp_range);
                    llama.l.llama_sample_entropy(llama.context, curP, dynatemp_min, dynatemp_max, dynatemp_exponent);
                } else {
                    llama.l.llama_sample_temp(llama.context, curP, temp);
                }
                */
          llama.l.llama_sample_temp(llama.context, curP, temp);
          break;
        default:
          break;
      }
    }
  }

  int impl(
      Pointer<Int32> idx, bool isResampling, Pointer<llama_context>? ctxCfg) {
    if (params == null) {
      throw Exception("sampling params is required");
    }
    var samplingParams = params!;

    int nVocab = llama.l.llama_n_vocab(llama.model);

    var temp = samplingParams.temp;
    var penaltyLastN = samplingParams.penaltyLastN < 0
        ? samplingParams.nPrev
        : samplingParams.penaltyLastN;
    var penaltyRepeat = samplingParams.penaltyRepeat;
    var penaltyFreq = samplingParams.penaltyFreq;
    var penaltyPresent = samplingParams.penaltyPresent;
    var mirostat = samplingParams.mirostat;
    var mirostatTau = samplingParams.mirostatTau;
    var mirostatEta = samplingParams.mirostatEta;
    var penalizeNl = samplingParams.penalizeNl;

    var prev = _prev;
    List<Pointer<llama_token_data>> cur = _cur;

    int id = 0;

    var logits = llama.l.llama_get_logits_ith(llama.context, idx.value);

    List<double> originalLogits = [];

    if (!isResampling) {
      int nVocab = llama.l.llama_n_vocab(llama.model);

      originalLogits = List<double>.filled(nVocab, 0.0);

      for (int i = 0; i < nVocab; i++) {
        originalLogits[i] = logits.elementAt(i).value;
      }
    }

    for (var entry in samplingParams.logitBias.entries) {
      logits[entry.key] += entry.value;
    }

    if (ctxCfg != null) {
      Pointer<Float> logitsGuidance =
          llama.l.llama_get_logits_ith(ctxCfg, idx.value);
      llama.l.llama_sample_apply_guidance(
          llama.context, logits, logitsGuidance, samplingParams.cfgScale);
    }

    cur.clear();

    for (int tokenId = 0; tokenId < nVocab; tokenId++) {
      var data = calloc.allocate<llama_token_data>(sizeOf<llama_token_data>());
      data.ref.id = tokenId;
      data.ref.logit = logits[tokenId];
      data.ref.p = 0.0;
      cur.add(data);
    }

    final Pointer<llama_token_data> dataArray =
        calloc<llama_token_data>(cur.length);
    for (int i = 0; i < cur.length; i++) {
      dataArray.elementAt(i).ref.id = cur[i].ref.id;
      dataArray.elementAt(i).ref.logit = cur[i].ref.logit;
      dataArray.elementAt(i).ref.p = cur[i].ref.p;
    }

    final Pointer<llama_token_data_array> curP =
        calloc<llama_token_data_array>();
    curP.ref.data = dataArray;
    curP.ref.size = cur.length;
    curP.ref.sorted = false;

    List<Pointer<llama_token>> list = [];
    for (int i = 0; i < samplingParams.penaltyPromptTokens.length; i++) {
      Pointer<llama_token> val =
          calloc.allocate<llama_token>(sizeOf<llama_token>());
      val.value = samplingParams.penaltyPromptTokens[i];
    }

    List<Pointer<llama_token>> penaltyTokens =
        samplingParams.usePenaltyPromptTokens ? list : prev;

    int penaltyTokensUsedSize = min(penaltyTokens.length, penaltyLastN);

    if (penaltyTokensUsedSize != 0) {
      var nlLogit = logits[llama.l.llama_token_nl(llama.model)];

      Pointer<llama_token> sublistPtr =
          calloc.allocate<llama_token>(penaltyTokensUsedSize);
      for (int i = 0; i < penaltyTokensUsedSize; i++) {
        sublistPtr[i] =
            penaltyTokens[penaltyTokens.length - penaltyTokensUsedSize + i]
                .value;
      }

      llama.l.llama_sample_repetition_penalties(llama.context, curP, sublistPtr,
          penaltyTokensUsedSize, penaltyRepeat, penaltyFreq, penaltyPresent);

      calloc.free(sublistPtr);

      if (!penalizeNl) {
        for (int idx = 0; idx < curP.ref.size; idx++) {
          if (curP.ref.data[idx].id == llama.l.llama_token_nl(llama.model)) {
            curP.ref.data[idx].logit = nlLogit;
            break;
          }
        }
      }
    }

    //*
    if (temp < 0.0) {
      // greedy sampling, with probs
      llama.l.llama_sample_softmax(llama.context, curP);
      id = curP.ref.data[0].id;
    } else if (temp == 0.0) {
      // greedy sampling, no probs
      id = llama.l.llama_sample_token_greedy(llama.context, curP);
    } else {
      Pointer<Float> mirostatMu = calloc.allocate<Float>(sizeOf<Float>());
      if (mirostat == 1) {
        const int mirostatM = 100;
        llama.l.llama_sample_temp(llama.context, curP, temp);
        id = llama.l.llama_sample_token_mirostat(llama.context, curP,
            mirostatTau, mirostatEta, mirostatM, mirostatMu);
      } else if (mirostat == 2) {
        llama.l.llama_sample_temp(llama.context, curP, temp);
        id = llama.l.llama_sample_token_mirostat_v2(
            llama.context, curP, mirostatTau, mirostatEta, mirostatMu);
      } else {
        int minKeep = max(1, samplingParams.nProbs);
        queue(curP, minKeep);
        id = llama.l.llama_sample_token(llama.context, curP);
      }
    }
    //*/

    return id;
  }

  int sample(Pointer<Int32> idx, Pointer<llama_context>? ctxCfg) {
    return impl(idx, false, ctxCfg);
  }

  accept(int id) {
    if (_prev.isNotEmpty) {
      _prev.removeAt(0);
    }
    Pointer<llama_token> idx =
        calloc.allocate<llama_token>(sizeOf<llama_token>());
    _prev.add(idx);
  }
}
