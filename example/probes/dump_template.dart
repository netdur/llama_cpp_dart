/// Dump the full embedded chat template from a model's GGUF metadata.
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB']!;
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL']!;
  LlamaLibrary.load(path: libPath);
  final model = LlamaModel.load(
    ModelParams(path: modelPath, gpuLayers: 0, vocabOnly: true),
  );

  final tmpl = ChatTemplate.fromModel(model);
  if (tmpl == null) {
    stdout.writeln('(no embedded template)');
  } else {
    stdout.writeln('--- length=${tmpl.length} ---');
    stdout.writeln(tmpl);
    stdout.writeln('--- end ---');
  }
  model.dispose();
}
