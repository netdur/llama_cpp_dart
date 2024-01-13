import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'inference_parameters.dart';
import 'llama_cpp.dart';

class GptParams {
  late llama_cpp _lib;
  late llm_gpt_params _parameters;
  late InferenceParameters _inferenceParameters;

  GptParams() {
    _lib = llama_cpp(DynamicLibrary.process());
    _parameters = _lib.llm_create_gpt_params();
  }

  llm_gpt_params get() => _parameters;

  InferenceParameters get inferenceParameters => _inferenceParameters;
  set cfgNegativePrompt(InferenceParameters inferenceParameters) {
    _inferenceParameters = inferenceParameters;
    _parameters.sparams = inferenceParameters.get();
  }

  int get seed => _parameters.seed;
  set seed(int value) => _parameters.seed = value;

  int get nThreads => _parameters.n_threads;
  set nThreads(int value) => _parameters.n_threads = value;

  int get nThreadsBatch => _parameters.n_threads_batch;
  set nThreadsBatch(int value) => _parameters.n_threads_batch = value;

  int get nPredict => _parameters.n_predict;
  set nPredict(int value) => _parameters.n_predict = value;

  int get nCtx => _parameters.n_ctx;
  set nCtx(int value) => _parameters.n_ctx = value;

  int get nBatch => _parameters.n_batch;
  set nBatch(int value) => _parameters.n_batch = value;

  int get nKeep => _parameters.n_keep;
  set nKeep(int value) => _parameters.n_keep = value;

  int get nDraft => _parameters.n_draft;
  set nDraft(int value) => _parameters.n_draft = value;

  int get nChunks => _parameters.n_chunks;
  set nChunks(int value) => _parameters.n_chunks = value;

  int get nParallel => _parameters.n_parallel;
  set nParallel(int value) => _parameters.n_parallel = value;

  int get nSequences => _parameters.n_sequences;
  set nSequences(int value) => _parameters.n_sequences = value;

  double get pAccept => _parameters.p_accept;
  set pAccept(double value) => _parameters.p_accept = value;

  double get pSplit => _parameters.p_split;
  set pSplit(double value) => _parameters.p_split = value;

  int get nGpuLayers => _parameters.n_gpu_layers;
  set nGpuLayers(int value) => _parameters.n_gpu_layers = value;

  int get nGpuLayersDraft => _parameters.n_gpu_layers_draft;
  set nGpuLayersDraft(int value) => _parameters.n_gpu_layers_draft = value;

  int get mainGpu => _parameters.main_gpu;
  set mainGpu(int value) => _parameters.main_gpu = value;

  List<double> get tensorSplit =>
      List<double>.generate(16, (i) => _parameters.tensor_split[i]);
  set tensorSplit(List<double> values) {
    for (int i = 0; i < 16; i++) {
      _parameters.tensor_split[i] = values[i];
    }
  }

  int get nBeams => _parameters.n_beams;
  set nBeams(int value) => _parameters.n_beams = value;

  double get ropeFreqBase => _parameters.rope_freq_base;
  set ropeFreqBase(double value) => _parameters.rope_freq_base = value;

  double get ropeFreqScale => _parameters.rope_freq_scale;
  set ropeFreqScale(double value) => _parameters.rope_freq_scale = value;

  double get yarnExtFactor => _parameters.yarn_ext_factor;
  set yarnExtFactor(double value) => _parameters.yarn_ext_factor = value;

  double get yarnAttnFactor => _parameters.yarn_attn_factor;
  set yarnAttnFactor(double value) => _parameters.yarn_attn_factor = value;

  double get yarnBetaFast => _parameters.yarn_beta_fast;
  set yarnBetaFast(double value) => _parameters.yarn_beta_fast = value;

  double get yarnBetaSlow => _parameters.yarn_beta_slow;
  set yarnBetaSlow(double value) => _parameters.yarn_beta_slow = value;

  int get yarnOrigCtx => _parameters.yarn_orig_ctx;
  set yarnOrigCtx(int value) => _parameters.yarn_orig_ctx = value;

  int get ropeScalingType => _parameters.rope_scaling_type;
  set ropeScalingType(int value) => _parameters.rope_scaling_type = value;

  String get model => _parameters.model.cast<Utf8>().toDartString();
  set model(String value) =>
      _parameters.model = value.toNativeUtf8().cast<Char>();

  String get modelDraft => _parameters.model_draft.cast<Utf8>().toDartString();
  set modelDraft(String value) =>
      _parameters.model_draft = value.toNativeUtf8().cast<Char>();

  String get modelAlias => _parameters.model_alias.cast<Utf8>().toDartString();
  set modelAlias(String value) =>
      _parameters.model_alias = value.toNativeUtf8().cast<Char>();

  String get prompt => _parameters.prompt.cast<Utf8>().toDartString();
  set prompt(String value) =>
      _parameters.prompt = value.toNativeUtf8().cast<Char>();

  String get promptFile => _parameters.prompt_file.cast<Utf8>().toDartString();
  set promptFile(String value) =>
      _parameters.prompt_file = value.toNativeUtf8().cast<Char>();

  String get pathPromptCache =>
      _parameters.path_prompt_cache.cast<Utf8>().toDartString();
  set pathPromptCache(String value) =>
      _parameters.path_prompt_cache = value.toNativeUtf8().cast<Char>();

  String get inputPrefix =>
      _parameters.input_prefix.cast<Utf8>().toDartString();
  set inputPrefix(String value) =>
      _parameters.input_prefix = value.toNativeUtf8().cast<Char>();

  String get inputSuffix =>
      _parameters.input_suffix.cast<Utf8>().toDartString();
  set inputSuffix(String value) =>
      _parameters.input_suffix = value.toNativeUtf8().cast<Char>();

  String get logdir => _parameters.logdir.cast<Utf8>().toDartString();
  set logdir(String value) =>
      _parameters.logdir = value.toNativeUtf8().cast<Char>();

  String get loraBase => _parameters.lora_base.cast<Utf8>().toDartString();
  set loraBase(String value) =>
      _parameters.lora_base = value.toNativeUtf8().cast<Char>();

  int get pplStride => _parameters.ppl_stride;
  set pplStride(int value) => _parameters.ppl_stride = value;

  int get pplOutputType => _parameters.ppl_output_type;
  set pplOutputType(int value) => _parameters.ppl_output_type = value;

  int get hellaswag => _parameters.hellaswag;
  set hellaswag(int value) => _parameters.hellaswag = value;

  int get hellaswagTasks => _parameters.hellaswag_tasks;
  set hellaswagTasks(int value) => _parameters.hellaswag_tasks = value;

  int get mulMatQ => _parameters.mul_mat_q;
  set mulMatQ(int value) => _parameters.mul_mat_q = value;

  int get randomPrompt => _parameters.random_prompt;
  set randomPrompt(int value) => _parameters.random_prompt = value;

  int get useColor => _parameters.use_color;
  set useColor(int value) => _parameters.use_color = value;

  int get interactive => _parameters.interactive;
  set interactive(int value) => _parameters.interactive = value;

  int get chatml => _parameters.chatml;
  set chatml(int value) => _parameters.chatml = value;

  int get promptCacheAll => _parameters.prompt_cache_all;
  set promptCacheAll(int value) => _parameters.prompt_cache_all = value;

  int get promptCacheRo => _parameters.prompt_cache_ro;
  set promptCacheRo(int value) => _parameters.prompt_cache_ro = value;

  int get embedding => _parameters.embedding;
  set embedding(int value) => _parameters.embedding = value;

  int get escape => _parameters.escape;
  set escape(int value) => _parameters.escape = value;

  int get interactiveFirst => _parameters.interactive_first;
  set interactiveFirst(int value) => _parameters.interactive_first = value;

  int get multilineInput => _parameters.multiline_input;
  set multilineInput(int value) => _parameters.multiline_input = value;

  int get simpleIo => _parameters.simple_io;
  set simpleIo(int value) => _parameters.simple_io = value;

  int get contBatching => _parameters.cont_batching;
  set contBatching(int value) => _parameters.cont_batching = value;

  int get inputPrefixBos => _parameters.input_prefix_bos;
  set inputPrefixBos(int value) => _parameters.input_prefix_bos = value;

  int get ignoreEos => _parameters.ignore_eos;
  set ignoreEos(int value) => _parameters.ignore_eos = value;

  int get instruct => _parameters.instruct;
  set instruct(int value) => _parameters.instruct = value;

  int get logitsAll => _parameters.logits_all;
  set logitsAll(int value) => _parameters.logits_all = value;

  int get useMmap => _parameters.use_mmap;
  set useMmap(int value) => _parameters.use_mmap = value;

  int get useMlock => _parameters.use_mlock;
  set useMlock(int value) => _parameters.use_mlock = value;

  int get numa => _parameters.numa;
  set numa(int value) => _parameters.numa = value;

  int get verbosePrompt => _parameters.verbose_prompt;
  set verbosePrompt(int value) => _parameters.verbose_prompt = value;

  int get infill => _parameters.infill;
  set infill(int value) => _parameters.infill = value;

  int get dumpKvCache => _parameters.dump_kv_cache;
  set dumpKvCache(int value) => _parameters.dump_kv_cache = value;

  int get noKvOffload => _parameters.no_kv_offload;
  set noKvOffload(int value) => _parameters.no_kv_offload = value;

  List<int> get cacheTypeK =>
      List<int>.generate(4, (i) => _parameters.cache_type_k[i]);
  set cacheTypeK(List<int> values) {
    for (int i = 0; i < 4; i++) {
      _parameters.cache_type_k[i] = values[i];
    }
  }

  List<int> get cacheTypeV =>
      List<int>.generate(4, (i) => _parameters.cache_type_v[i]);
  set cacheTypeV(List<int> values) {
    for (int i = 0; i < 4; i++) {
      _parameters.cache_type_v[i] = values[i];
    }
  }

  String get mmproj => _parameters.mmproj.cast<Utf8>().toDartString();
  set mmproj(String value) =>
      _parameters.mmproj = value.toNativeUtf8().cast<Char>();

  String get image => _parameters.image.cast<Utf8>().toDartString();
  set image(String value) =>
      _parameters.image = value.toNativeUtf8().cast<Char>();

  void dispose() {
    _inferenceParameters.dispose();

    calloc.free(_parameters.model);
    calloc.free(_parameters.model_draft);
    calloc.free(_parameters.model_alias);
    calloc.free(_parameters.prompt);
    calloc.free(_parameters.prompt_file);
    calloc.free(_parameters.path_prompt_cache);
    calloc.free(_parameters.input_prefix);
    calloc.free(_parameters.input_suffix);
    calloc.free(_parameters.logdir);
    calloc.free(_parameters.lora_base);
  }
}
