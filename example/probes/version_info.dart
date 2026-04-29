/// Print binding + llama.cpp pin info, plus the runtime system info.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///     dart run example/probes/version_info.dart
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  stdout.writeln('package            = ${LlamaVersion.package}');
  stdout.writeln('llamaCppCommit     = ${LlamaVersion.llamaCppCommit}');
  stdout.writeln('llamaCppCommitDate = ${LlamaVersion.llamaCppCommitDate}');

  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  if (libPath == null) {
    stdout.writeln('(set LLAMA_CPP_DART_LIB to also see runtime systemInfo)');
    return;
  }
  LlamaLibrary.load(path: libPath);
  stdout.writeln('systemInfo       = ${LlamaVersion.systemInfo()}');
}
