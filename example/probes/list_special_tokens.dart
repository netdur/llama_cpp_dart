/// Dump the model's special token ids and labels.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart run example/probes/list_special_tokens.dart
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  if (libPath == null || modelPath == null) {
    stderr.writeln(
      'set LLAMA_CPP_DART_LIB and LLAMA_CPP_DART_MODEL before running',
    );
    exit(2);
  }

  LlamaLibrary.load(path: libPath);
  final model = LlamaModel.load(
    ModelParams(path: modelPath, gpuLayers: 0, vocabOnly: true),
  );
  final vocab = model.vocab;
  final tokenizer = Tokenizer(vocab);

  final entries = <String, int>{
    'BOS': vocab.bos,
    'EOS': vocab.eos,
    'EOT': vocab.eot,
    'SEP': vocab.sep,
    'NL': vocab.nl,
    'PAD': vocab.pad,
    'MASK': vocab.mask,
  };

  stdout.writeln('vocab size: ${vocab.nTokens}');
  for (final entry in entries.entries) {
    final id = entry.value;
    if (id < 0) {
      stdout.writeln('  ${entry.key.padRight(5)} : (none)');
      continue;
    }
    final piece = tokenizer.decode(id).replaceAll('\n', r'\n');
    final flags = <String>[
      if (vocab.isEog(id)) 'eog',
      if (vocab.isControl(id)) 'control',
    ].join(',');
    stdout.writeln(
      '  ${entry.key.padRight(5)} : ${id.toString().padLeft(6)} '
      '"$piece"${flags.isEmpty ? '' : '  [$flags]'}',
    );
  }

  model.dispose();
  LlamaLibrary.dispose();
}
