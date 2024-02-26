// ignore_for_file: camel_case_types

import 'dart:ffi';
import 'dart:math';
import 'package:ffi/ffi.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

import 'llama_cpp.dart';

class SamplingContext {
  final List<Pointer<llama_token>> _prev = [];
  final List<Pointer<llama_token_data>> _cur = [];

  SamplingParams? params;
  final Llama llama;

  SamplingContext(this.llama) {
    // result->grammar = nullptr;
  }

  void dispose() {
    reset();
    // free grammae
  }

  void reset() {
    for (var ptr in _prev) {
      calloc.free(ptr);
    }
    _prev.clear();
    for (var ptr in _cur) {
      calloc.free(ptr);
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

    int id = 0;

    var logits = llama.l.llama_get_logits_ith(llama.context, idx.value);

    List<double> originalLogits = [];

    // never happens?
    if (!isResampling) {
      int nVocab = llama.l.llama_n_vocab(llama.model);

      originalLogits = List<double>.filled(nVocab, 0.0);

      for (int i = 0; i < nVocab; i++) {
        originalLogits[i] = logits[i];
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
      calloc.free(logitsGuidance);
    }

    for (var ptr in _cur) {
      calloc.free(ptr);
    }
    _cur.clear();

    for (int tokenId = 0; tokenId < nVocab; tokenId++) {
      var data = calloc.allocate<llama_token_data>(sizeOf<llama_token_data>());
      data.ref.id = tokenId;
      data.ref.logit = logits[tokenId];
      data.ref.p = 0.0;
      _cur.add(data);
    }

    final Pointer<llama_token_data> dataArray =
        calloc<llama_token_data>(_cur.length);
    for (int i = 0; i < _cur.length; i++) {
      // dataArray.elementAt(i).ref.id = _cur[i].ref.id;
      // dataArray.elementAt(i).ref.logit = _cur[i].ref.logit;
      // dataArray.elementAt(i).ref.p = _cur[i].ref.p;
      dataArray[i].id = _cur[i].ref.id;
      dataArray[i].logit = _cur[i].ref.logit;
      dataArray[i].p = _cur[i].ref.p;
    }

    final Pointer<llama_token_data_array> curP =
        calloc<llama_token_data_array>();
    curP.ref.data = dataArray;
    curP.ref.size = _cur.length;
    curP.ref.sorted = false;

    List<Pointer<llama_token>> list = [];
    for (int i = 0; i < samplingParams.penaltyPromptTokens.length; i++) {
      Pointer<llama_token> val =
          calloc.allocate<llama_token>(sizeOf<llama_token>());
      val.value = samplingParams.penaltyPromptTokens[i];
    }

    List<Pointer<llama_token>> penaltyTokens =
        samplingParams.usePenaltyPromptTokens ? list : _prev;

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
      calloc.free(mirostatMu);
    }

    calloc.free(curP);
    calloc.free(dataArray);

    return id;
  }

  int sample(Pointer<Int32> idx, Pointer<llama_context>? ctxCfg) {
    return impl(idx, false, ctxCfg);
  }

  accept(int id) {
    if (_prev.isNotEmpty) {
      calloc.free(_prev[0]);
      _prev.removeAt(0);
    }
    Pointer<llama_token> idx =
        calloc.allocate<llama_token>(sizeOf<llama_token>());
    idx.value = id;
    _prev.add(idx);
  }
}
