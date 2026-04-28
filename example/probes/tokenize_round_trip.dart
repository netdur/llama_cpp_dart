/// Encode a string to tokens and decode it back, printing each piece.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart run example/probes/tokenize_round_trip.dart "Hello, world!"
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main(List<String> args) {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  if (libPath == null || modelPath == null) {
    stderr.writeln(
      'set LLAMA_CPP_DART_LIB and LLAMA_CPP_DART_MODEL before running',
    );
    exit(2);
  }

  final input =
      args.isEmpty ? 'The quick brown fox jumps over the lazy dog.' : args.join(' ');

  LlamaLibrary.load(path: libPath);
  final model = LlamaModel.load(
    ModelParams(path: modelPath, gpuLayers: 0, vocabOnly: true),
  );
  final tokenizer = Tokenizer(model.vocab);

  final tokens = tokenizer.encode(input);
  stdout.writeln('input  : $input');
  stdout.writeln('tokens : ${tokens.length}');
  for (final id in tokens) {
    final piece = tokenizer.decode(id);
    final escaped = piece
        .replaceAll('\n', r'\n')
        .replaceAll('\t', r'\t')
        .replaceAll('\r', r'\r');
    stdout.writeln('  ${id.toString().padLeft(6)}  "$escaped"');
  }

  final decoded = tokenizer.decodeAll(tokens);
  stdout.writeln('decoded: $decoded');
  stdout.writeln('match  : ${decoded.trim() == input.trim()}');

  model.dispose();
  LlamaLibrary.dispose();
}
