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
  List<Pointer<llama_token_data>> _cur = [];

  final SamplingParams samplingParams;
  final Llama llama;

  SamplingContext(this.llama, this.samplingParams) {
    // result->grammar = nullptr;
    _prev = List.generate(samplingParams.nPrev,
        (index) => calloc.allocate<llama_token>(sizeOf<llama_token>()));
  }

  void dispose() {
    // free grammae
  }

  void reset() {
    _prev = List.generate(samplingParams.nPrev,
        (index) => calloc.allocate<llama_token>(sizeOf<llama_token>()));
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

  int impl(int idx, bool is_resampling, Pointer<llama_context>? ctx_cfg) {
    int n_vocab = llama.l.llama_n_vocab(llama.model);

    var temp = samplingParams.temp;
    var penalty_last_n = samplingParams.penaltyLastN < 0
        ? samplingParams.nPrev
        : samplingParams.penaltyLastN;
    var penalty_repeat = samplingParams.penaltyRepeat;
    var penalty_freq = samplingParams.penaltyFreq;
    var penalty_present = samplingParams.penaltyPresent;
    var mirostat = samplingParams.mirostat;
    var mirostat_tau = samplingParams.mirostatTau;
    var mirostat_eta = samplingParams.mirostatEta;
    var penalize_nl = samplingParams.penalizeNl;

    var prev = _prev;
    var cur = _cur;

    int id = 0;

    var logits = llama.l.llama_get_logits_ith(llama.context, idx);

    List<double> original_logits = [];

    if (!is_resampling) {
      // Get the number of elements from the C++ function
      int n_vocab = llama.l.llama_n_vocab(llama.model);

      // Allocating memory for the Dart list
      original_logits = List<double>.filled(n_vocab, 0.0);

      // Copying data from the Pointer<Float> to the Dart list
      for (int i = 0; i < n_vocab; i++) {
        original_logits[i] = logits.elementAt(i).value;
      }
    }

    for (var entry in samplingParams.logitBias.entries) {
      logits[entry.key] += entry.value;
    }

    if (ctx_cfg != null) {
      Pointer<Float> logits_guidance =
          llama.l.llama_get_logits_ith(ctx_cfg, idx);
      llama.l.llama_sample_apply_guidance(
          llama.context, logits, logits_guidance, samplingParams.cfgScale);
    }

    cur.clear();

    for (int token_id = 0; token_id < n_vocab; token_id++) {
      var data = calloc.allocate<llama_token_data>(sizeOf<llama_token_data>());
      data.ref.id = token_id;
      data.ref.logit = logits[token_id];
      data.ref.p = 0.0;
      cur.add(data);
    }

    var cur_p = calloc<llama_token_data_array>();

    List<Pointer<llama_token>> list = [];
    for (int i = 0; i < samplingParams.penaltyPromptTokens.length; i++) {
      Pointer<llama_token> val =
          calloc.allocate<llama_token>(sizeOf<llama_token>());
      val.value = samplingParams.penaltyPromptTokens[i];
    }

    List<Pointer<llama_token>> penalty_tokens =
        samplingParams.usePenaltyPromptTokens ? list : prev;

    int penalty_tokens_used_size = min(penalty_tokens.length, penalty_last_n);

    if (penalty_tokens_used_size != 0) {
      var nl_logit = logits[llama.l.llama_token_nl(llama.model)];

      Pointer<llama_token> sublistPtr =
          calloc.allocate<llama_token>(penalty_tokens_used_size);
      for (int i = 0; i < penalty_tokens_used_size; i++) {
        sublistPtr[i] =
            penalty_tokens[penalty_tokens.length - penalty_tokens_used_size + i]
                .value;
      }

      llama.l.llama_sample_repetition_penalties(
          llama.context,
          cur_p,
          sublistPtr,
          penalty_tokens_used_size,
          penalty_repeat,
          penalty_freq,
          penalty_present);

      calloc.free(sublistPtr);

      if (!penalize_nl) {
        for (int idx = 0; idx < cur_p.ref.size; idx++) {
          if (cur_p.ref.data[idx].id == llama.l.llama_token_nl(llama.model)) {
            cur_p.ref.data[idx].logit = nl_logit;
            break;
          }
        }
      }
    }

    if (temp < 0.0) {
      // greedy sampling, with probs
      llama.l.llama_sample_softmax(llama.context, cur_p);
      id = cur_p.ref.data[0].id;
    } else if (temp == 0.0) {
      // greedy sampling, no probs
      id = llama.l.llama_sample_token_greedy(llama.context, cur_p);
    } else {
      Pointer<Float> mirostat_mu = calloc.allocate<Float>(sizeOf<Float>());
      if (mirostat == 1) {
        const int mirostat_m = 100;
        llama.l.llama_sample_temp(llama.context, cur_p, temp);
        id = llama.l.llama_sample_token_mirostat(llama.context, cur_p,
            mirostat_tau, mirostat_eta, mirostat_m, mirostat_mu);
      } else if (mirostat == 2) {
        llama.l.llama_sample_temp(llama.context, cur_p, temp);
        id = llama.l.llama_sample_token_mirostat_v2(
            llama.context, cur_p, mirostat_tau, mirostat_eta, mirostat_mu);
      } else {
        int min_keep = max(1, samplingParams.nProbs);
        queue(cur_p, min_keep);
        id = llama.l.llama_sample_token(llama.context, cur_p);
      }
    }

    return id;
  }

  int sample(int idx, Pointer<llama_context>? ctx_cfg) {
    return impl(idx, false, ctx_cfg);
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
