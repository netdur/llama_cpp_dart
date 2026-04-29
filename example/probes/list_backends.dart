/// Print every ggml-backend device the runtime exposed.
///
/// Use to verify that Hexagon / OpenCL / Metal actually loaded after
/// dropping in the relevant native library bundle.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///     dart run example/probes/list_backends.dart
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  if (libPath == null || libPath.isEmpty) {
    stderr.writeln('set LLAMA_CPP_DART_LIB to libllama.{dylib,so}');
    exit(2);
  }
  LlamaLibrary.load(path: libPath);

  final devices = LlamaBackends.list();
  if (devices.isEmpty) {
    stdout.writeln('(no devices reported — backends did not load)');
    return;
  }

  stdout.writeln('${devices.length} ggml-backend device(s):');
  for (var i = 0; i < devices.length; i++) {
    final d = devices[i];
    stdout.writeln('  [$i] ${d.name}');
    stdout.writeln('       type    : ${d.type.name}');
    stdout.writeln('       registry: ${d.registryName}');
    stdout.writeln('       desc    : ${d.description}');
    if (d.memoryTotal > 0) {
      final freeMib = (d.memoryFree / 1024 / 1024).toStringAsFixed(1);
      final totalMib = (d.memoryTotal / 1024 / 1024).toStringAsFixed(1);
      stdout.writeln('       memory  : $freeMib / $totalMib MiB free');
    }
  }

  final accelerators = devices.where((d) => d.isAccelerator).toList();
  stdout.writeln();
  stdout.writeln('accelerators: ${accelerators.length}');
  for (final a in accelerators) {
    stdout.writeln('  - ${a.registryName} (${a.name})');
  }
}
