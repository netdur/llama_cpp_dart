/// Print metadata about a loaded GGUF model.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/Users/adel/Workspace/gguf/gemma-4-E2B-it-Q8_0.gguf \
///     dart run example/probes/inspect_model.dart
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  final libPath = '/Users/adel/Workspace/llama_cpp_dart/build/macos/install/lib/libllama.dylib'; // Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = '/Users/adel/Workspace/gguf/gemma-4-E2B-it-Q8_0.gguf'; // Platform.environment['LLAMA_CPP_DART_MODEL'];

  LlamaLibrary.load(path: libPath);
  final model = LlamaModel.load(ModelParams(path: modelPath, gpuLayers: 99));

  stdout.writeln('description : ${model.describe()}');
  stdout.writeln('parameters  : ${_human(model.nParams)}');
  stdout.writeln('size on disk: ${_humanBytes(model.sizeBytes)}');
  stdout.writeln('embedding   : ${model.nEmbd}');
  stdout.writeln('layers      : ${model.nLayer}');
  stdout.writeln('heads       : ${model.nHead} (kv: ${model.nHeadKv})');
  stdout.writeln('train ctx   : ${model.trainCtx}');
  stdout.writeln('vocab size  : ${model.vocab.nTokens}');
  stdout.writeln('encoder     : ${model.hasEncoder}');
  stdout.writeln('decoder     : ${model.hasDecoder}');
  stdout.writeln('recurrent   : ${model.isRecurrent}');

  model.dispose();
  LlamaLibrary.dispose();
}

String _human(int n) {
  const units = ['', 'K', 'M', 'B', 'T'];
  var v = n.toDouble();
  var i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return '${v.toStringAsFixed(2)}${units[i]}';
}

String _humanBytes(int n) {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  var v = n.toDouble();
  var i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return '${v.toStringAsFixed(2)} ${units[i]}';
}
