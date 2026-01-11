import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

Future<void> main() async {
  const modelPath = '/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf';
  if (modelPath.isEmpty) {
    print('Set modelPath in example/persist_name_session.dart before running.');
    exit(1);
  }

  // Setup library path (Critical for Mac).
  if (Platform.isMacOS) {
    Llama.libraryPath =
        "/Users/adel/Workspace/llama_cpp_dart/bin/MAC_ARM64/libllama.dylib";
  }

  final sessionHome = '${Directory.current.path}/sessions/persist_name_session';
  Directory(sessionHome).createSync(recursive: true);

  final contextParams = ContextParams()
    ..nSeqMax = 1
    ..nCtx = 2048;

  const sessionId = 'adel_session';
  final statePath = '$sessionHome/$sessionId.state';
  await _runSaveThenLoad(
    modelPath: modelPath,
    sessionId: sessionId,
    sessionHome: sessionHome,
    statePath: statePath,
    contextParams: contextParams,
  );

  await _runStatefulConcurrentRequests(modelPath: modelPath);
  await _runStatelessSingleRequest(modelPath: modelPath);
  await _runStatelessConcurrentRequests(modelPath: modelPath);
}

Future<void> _runSaveThenLoad({
  required String modelPath,
  required String sessionId,
  required String sessionHome,
  required String statePath,
  required ContextParams contextParams,
}) async {
  final stateFile = File(statePath);
  if (stateFile.existsSync()) {
    stateFile.deleteSync();
  }

  final writerService = LlamaService(
    modelPath,
    contextParams: contextParams,
    sessionHome: sessionHome,
    verbose: false,
  );

  try {
    writerService.createSession(sessionId);
    final response1 = await _runPrompt(
      writerService,
      sessionId,
      '<start_of_turn>user\nhello my name is adel!'
      '<end_of_turn>\n'
      '<start_of_turn>model\n',
    );
    print('First response: $response1');

    await writerService.saveSession(sessionId, statePath);
    print('Saved state: $statePath');
  } finally {
    await writerService.freeSession(sessionId);
    await writerService.dispose();
  }

  final readerService = LlamaService(
    modelPath,
    contextParams: contextParams,
    sessionHome: sessionHome,
    verbose: false,
  );

  try {
    final loaded = await readerService.loadSession(sessionId, statePath);
    if (!loaded) {
      throw StateError('Failed to load session from $statePath');
    }

    final response2 = await _runPrompt(
      readerService,
      sessionId,
      '<start_of_turn>user\nwhat is my name?'
      '<end_of_turn>\n'
      '<start_of_turn>model\n',
      clearHistory: false,
    );
    print('Second response: $response2');
  } finally {
    await readerService.freeSession(sessionId);
    await readerService.dispose();

    if (stateFile.existsSync()) {
      await stateFile.delete();
      print('Deleted state: ${stateFile.path}');
    }
  }
}

Future<String> _runPrompt(
  LlamaService service,
  String sessionId,
  String prompt, {
  bool clearHistory = true,
}) async {
  final buffer = StringBuffer();
  final subscription = service.generateText(sessionId).listen(buffer.write);

  await service.setPrompt(sessionId, prompt, clearHistory: clearHistory);

  while (service.status(sessionId) == LlamaStatus.generating) {
    await Future.delayed(const Duration(milliseconds: 10));
  }

  await subscription.cancel();
  return buffer.toString().trim();
}

String _prepareSessionHome(String label) {
  final path =
      '${Directory.current.path}/sessions/persist_name_session/$label';
  final dir = Directory(path);
  if (dir.existsSync()) {
    dir.deleteSync(recursive: true);
  }
  dir.createSync(recursive: true);
  return path;
}

Future<void> _runStatefulConcurrentRequests({
  required String modelPath,
}) async {
  print('\n--- Stateful: 2 concurrent requests ---');
  final sessionHome = _prepareSessionHome('stateful_concurrent');
  final contextParams = ContextParams()
    ..nSeqMax = 2
    ..nCtx = 2048;

  final service = LlamaService(
    modelPath,
    contextParams: contextParams,
    sessionHome: sessionHome,
    verbose: false,
  );

  try {
    service.createSession('stateful_1');
    service.createSession('stateful_2');

    final task1 = _runPrompt(
      service,
      'stateful_1',
      '<start_of_turn>user\nmy name is adel.'
      '<end_of_turn>\n'
      '<start_of_turn>model\n',
    );
    final task2 = _runPrompt(
      service,
      'stateful_2',
      '<start_of_turn>user\nmy favorite color is blue.'
      '<end_of_turn>\n'
      '<start_of_turn>model\n',
    );

    final results = await Future.wait([task1, task2]);
    print('Stateful 1 response: ${results[0]}');
    print('Stateful 2 response: ${results[1]}');
  } finally {
    await service.freeSession('stateful_1');
    await service.freeSession('stateful_2');
    await service.dispose();
  }
}

Future<void> _runStatelessSingleRequest({
  required String modelPath,
}) async {
  print('\n--- Stateless: 1 request ---');
  final sessionHome = _prepareSessionHome('stateless_single');
  final contextParams = ContextParams()
    ..nSeqMax = 1
    ..nCtx = 2048;

  final service = LlamaService(
    modelPath,
    contextParams: contextParams,
    sessionHome: sessionHome,
    verbose: false,
  );

  try {
    service.createSession('stateless_1');
    final response = await _runPrompt(
      service,
      'stateless_1',
      '<start_of_turn>user\nsay hello in one sentence.'
      '<end_of_turn>\n'
      '<start_of_turn>model\n',
    );
    print('Stateless response: $response');
  } finally {
    await service.freeSession('stateless_1');
    await service.dispose();
  }
}

Future<void> _runStatelessConcurrentRequests({
  required String modelPath,
}) async {
  print('\n--- Stateless: 2 concurrent requests ---');
  final sessionHome = _prepareSessionHome('stateless_concurrent');
  final contextParams = ContextParams()
    ..nSeqMax = 2
    ..nCtx = 2048;

  final service = LlamaService(
    modelPath,
    contextParams: contextParams,
    sessionHome: sessionHome,
    verbose: false,
  );

  try {
    service.createSession('stateless_1');
    service.createSession('stateless_2');

    final task1 = _runPrompt(
      service,
      'stateless_1',
      '<start_of_turn>user\nlist two fruits.'
      '<end_of_turn>\n'
      '<start_of_turn>model\n',
    );
    final task2 = _runPrompt(
      service,
      'stateless_2',
      '<start_of_turn>user\nname two animals.'
      '<end_of_turn>\n'
      '<start_of_turn>model\n',
    );

    final results = await Future.wait([task1, task2]);
    print('Stateless 1 response: ${results[0]}');
    print('Stateless 2 response: ${results[1]}');
  } finally {
    await service.freeSession('stateless_1');
    await service.freeSession('stateless_2');
    await service.dispose();
  }
}
