/// Save a chat session to disk, dispose the engine, restart, and resume
/// the conversation from the saved state.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart run example/probes/save_resume.dart
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  if (libPath == null || modelPath == null) {
    stderr.writeln(
      'set LLAMA_CPP_DART_LIB and LLAMA_CPP_DART_MODEL before running',
    );
    exit(2);
  }

  final tmpDir = Directory.systemTemp.createTempSync('llamacppdart_resume_');
  final statePath = '${tmpDir.path}/chat.lcdc';

  // ---- Phase 1: start a chat, generate, save ----
  stdout.writeln('phase 1: starting engine ...');
  var engine = await LlamaEngine.spawn(
    libraryPath: libPath,
    modelParams: ModelParams(path: modelPath, gpuLayers: 99),
    contextParams: const ContextParams(nCtx: 1024, nBatch: 256, nUbatch: 256),
  );

  if (engine.modelChatTemplate == null) {
    stderr.writeln('model has no chat template; this probe needs one');
    await engine.dispose();
    exit(1);
  }

  var chat = await engine.createChat();
  chat.addSystem('You are concise.');
  chat.addUser('Reply with the word "alpha" only.');

  stdout.write('<<< ');
  await for (final ev in chat.generate(
    sampler: SamplerParams.greedyDefault,
    maxTokens: 8,
  )) {
    if (ev is TokenEvent) stdout.write(ev.text);
  }
  stdout.writeln();

  await chat.saveState(statePath);
  final fileSize = File(statePath).statSync().size;
  stdout.writeln('saved $fileSize bytes to $statePath');
  stdout.writeln('messages so far: ${chat.messageCount}');

  await chat.dispose();
  await engine.dispose();

  // ---- Phase 2: spawn fresh engine, load state, continue ----
  stdout.writeln('\nphase 2: fresh engine, loading state ...');
  engine = await LlamaEngine.spawn(
    libraryPath: libPath,
    modelParams: ModelParams(path: modelPath, gpuLayers: 99),
    contextParams: const ContextParams(nCtx: 1024, nBatch: 256, nUbatch: 256),
  );
  chat = await engine.createChat();
  await chat.loadState(statePath);
  stdout.writeln('messages restored: ${chat.messageCount}');
  for (final m in chat.messages) {
    stdout.writeln('  ${m.role}: ${m.content}');
  }

  // Continue the conversation in the resumed session.
  chat.addUser('And now the word "beta".');
  stdout.write('<<< ');
  await for (final ev in chat.generate(
    sampler: SamplerParams.greedyDefault,
    maxTokens: 8,
  )) {
    if (ev is TokenEvent) stdout.write(ev.text);
  }
  stdout.writeln();
  stdout.writeln('messages now: ${chat.messageCount}');

  await chat.dispose();
  await engine.dispose();
  tmpDir.deleteSync(recursive: true);
}
