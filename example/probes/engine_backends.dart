/// Spawn an engine and dump its device list. This is the round-trip
/// version of `list_backends.dart` — verifies the worker isolate
/// captures the device list and ships it back via EngineReadyResponse.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart run example/probes/engine_backends.dart
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB']!;
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL']!;

  final engine = await LlamaEngine.spawn(
    libraryPath: libPath,
    modelParams: ModelParams(path: modelPath, gpuLayers: 99),
    contextParams: const ContextParams(nCtx: 1024, nBatch: 256, nUbatch: 256),
  );

  stdout.writeln('engine devices (${engine.devices.length}):');
  for (final d in engine.devices) {
    stdout.writeln('  $d');
  }

  stdout.writeln();
  stdout.writeln('hasAccelerator      : ${engine.hasAccelerator}');
  stdout.writeln(
      'primary accelerator : ${engine.primaryAcceleratorName ?? "(none)"}');
  stdout.writeln('canShift            : ${engine.canShift}');
  stdout.writeln('multimodalLoaded    : ${engine.multimodalLoaded}');

  await engine.dispose();
}
