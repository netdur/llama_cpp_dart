import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import '../batch/batch.dart';
import '../chat/chat_template.dart';
import '../context/context.dart';
import '../ffi/library_loader.dart';
import '../generation/context_shift.dart';
import '../generation/event.dart';
import '../generation/stop.dart';
import '../model/model.dart';
import '../multimodal/multimodal_context.dart';
import '../sampling/sampler_factory.dart';
import '../sampling/sampler_params.dart';
import '../session/session.dart';
import '../session/state_codec.dart';
import '../tokenizer/tokenizer.dart';
import '../tokenizer/utf8_accumulator.dart';
import 'messages.dart';

/// Top-level entry point for the engine worker isolate.
///
/// Loads the native library, model, and context inside the worker; sends
/// [EngineReadyResponse] when ready or [EngineErrorResponse] if startup
/// failed. Then listens for [EngineCommand]s on the worker's receive port.
Future<void> runEngineWorker(EngineBootstrap bootstrap) async {
  final reply = bootstrap.replyPort;
  final commandRx = ReceivePort();

  final LlamaModel model;
  final LlamaContext context;

  try {
    if (bootstrap.useProcessSymbols) {
      LlamaLibrary.loadFromProcess();
    } else {
      LlamaLibrary.load(
        path: bootstrap.libraryPath,
        backendDirectory: bootstrap.backendDirectory,
      );
    }
    // NOTE: LlamaLog.silence() is NOT called here. Pointer.fromFunction
    // callbacks are bound to a single isolate and crash when ggml's
    // Metal backend re-initializes from another thread. Log silencing
    // for the worker arrives with the thread-safe NativeCallable rework.
    model = LlamaModel.load(bootstrap.modelParams);
    context = LlamaContext.create(model, bootstrap.contextParams);
  } catch (e, st) {
    reply.send(EngineErrorResponse(0, '$e\n$st'));
    commandRx.close();
    return;
  }

  MultimodalContext? mtmd;
  if (bootstrap.multimodalParams != null) {
    try {
      mtmd = MultimodalContext.init(
        model: model,
        params: bootstrap.multimodalParams!,
      );
    } catch (e, st) {
      reply.send(EngineErrorResponse(0, 'multimodal init failed: $e\n$st'));
      commandRx.close();
      return;
    }
  }

  final modelTemplate = ChatTemplate.fromModel(model);
  reply.send(EngineReadyResponse(
    0,
    commandRx.sendPort,
    modelChatTemplate: modelTemplate,
    multimodalLoaded: mtmd != null,
    supportsVision: mtmd?.supportsVision ?? false,
    supportsAudio: mtmd?.supportsAudio ?? false,
    audioSampleRate: mtmd?.audioSampleRate ?? -1,
    canShift: context.canShift,
  ));

  final state = _WorkerState(
    model: model,
    context: context,
    modelTemplate: modelTemplate,
    multimodal: mtmd,
  );
  final completer = Completer<void>();

  commandRx.listen((dynamic msg) {
    if (msg is! EngineCommand) return;
    if (msg is CancelCommand) {
      state.cancelledRequests.add(msg.targetRequestId);
      reply.send(EngineAckResponse(msg.requestId));
      return;
    }
    if (msg is ShutdownCommand) {
      _shutdown(state, msg.requestId, reply, commandRx, completer);
      return;
    }
    // Other commands run as fire-and-forget microtasks. The worker still
    // serializes generates via the inFlight slot.
    _dispatch(msg, state, reply);
  });

  await completer.future;
}

class _WorkerState {
  _WorkerState({
    required this.model,
    required this.context,
    required this.modelTemplate,
    required this.multimodal,
  });

  final LlamaModel model;
  final LlamaContext context;
  final String? modelTemplate;
  final MultimodalContext? multimodal;
  final Map<int, LlamaSession> sessions = <int, LlamaSession>{};
  final Set<int> cancelledRequests = <int>{};
  int? inFlightGenerateId;
}

Future<void> _dispatch(
  EngineCommand cmd,
  _WorkerState state,
  SendPort reply,
) async {
  try {
    switch (cmd) {
      case CreateSessionCommand():
        if (state.sessions.containsKey(cmd.sessionId)) {
          reply.send(EngineErrorResponse(
            cmd.requestId,
            'session ${cmd.sessionId} already exists',
          ));
          return;
        }
        state.sessions[cmd.sessionId] =
            LlamaSession(state.context, seqId: cmd.seqId);
        reply.send(EngineAckResponse(cmd.requestId));

      case DisposeSessionCommand():
        final session = state.sessions.remove(cmd.sessionId);
        session?.clear();
        reply.send(EngineAckResponse(cmd.requestId));

      case AppendTextCommand():
        final session = state.sessions[cmd.sessionId];
        if (session == null) {
          reply.send(EngineErrorResponse(
            cmd.requestId,
            'no session ${cmd.sessionId}',
          ));
          return;
        }
        session.appendText(
          cmd.text,
          addSpecial: cmd.addSpecial,
          parseSpecial: cmd.parseSpecial,
        );
        reply.send(EngineAckResponse(cmd.requestId));

      case ClearSessionCommand():
        final session = state.sessions[cmd.sessionId];
        if (session == null) {
          reply.send(EngineErrorResponse(
            cmd.requestId,
            'no session ${cmd.sessionId}',
          ));
          return;
        }
        session.clear();
        reply.send(EngineAckResponse(cmd.requestId));

      case GenerateCommand():
        await _runGenerate(cmd, state, reply);

      case GenerateChatCommand():
        await _runGenerateChat(cmd, state, reply);

      case SaveSessionStateCommand():
        _runSaveSessionState(cmd, state, reply);

      case LoadSessionStateCommand():
        _runLoadSessionState(cmd, state, reply);

      case CancelCommand():
      case ShutdownCommand():
        // Handled in the receive-port listener directly.
        break;
    }
  } catch (e, st) {
    reply.send(EngineErrorResponse(cmd.requestId, '$e\n$st'));
  }
}

Future<void> _runGenerate(
  GenerateCommand cmd,
  _WorkerState state,
  SendPort reply,
) async {
  if (!_claimGenerate(cmd.requestId, state, reply)) return;

  final session = state.sessions[cmd.sessionId];
  if (session == null) {
    reply.send(EngineErrorResponse(
      cmd.requestId,
      'no session ${cmd.sessionId}',
    ));
    state.inFlightGenerateId = null;
    return;
  }

  try {
    if (cmd.media.isNotEmpty) {
      final mtmd = state.multimodal;
      if (mtmd == null) {
        reply.send(EngineErrorResponse(
          cmd.requestId,
          'request includes media but engine has no multimodal context; '
          'pass MultimodalParams to LlamaEngine.spawn',
        ));
        state.inFlightGenerateId = null;
        return;
      }
      if (cmd.prompt == null || cmd.prompt!.isEmpty) {
        reply.send(EngineErrorResponse(
          cmd.requestId,
          'media generate requires a non-empty prompt with media markers',
        ));
        state.inFlightGenerateId = null;
        return;
      }
      // Reset KV; mtmd's prefill assumes a fresh sequence.
      session.clear();
      final newNPast = mtmd.evalChunks(
        llamaContext: state.context,
        prompt: cmd.prompt!,
        media: cmd.media,
        nPast: 0,
        seqId: session.seqId,
        nBatch: state.context.nBatch,
        addSpecial: cmd.addSpecial,
        parseSpecial: cmd.parseSpecial,
        logitsLast: true,
      );
      await _streamSampleAfterPrefill(
        requestId: cmd.requestId,
        state: state,
        reply: reply,
        sampler: cmd.sampler,
        maxTokens: cmd.maxTokens,
        seqId: session.seqId,
        initialPos: newNPast,
      );
    } else {
      if (cmd.prompt != null && cmd.prompt!.isNotEmpty) {
        session.appendText(
          cmd.prompt!,
          addSpecial: cmd.addSpecial,
          parseSpecial: cmd.parseSpecial,
        );
      }
      await _streamSessionGenerate(
        requestId: cmd.requestId,
        session: session,
        sampler: cmd.sampler,
        maxTokens: cmd.maxTokens,
        state: state,
        reply: reply,
        shiftPolicy: cmd.shiftPolicy,
        shift: cmd.shift,
      );
    }
  } catch (e, st) {
    reply.send(EngineErrorResponse(cmd.requestId, '$e\n$st'));
  } finally {
    state.inFlightGenerateId = null;
    state.cancelledRequests.remove(cmd.requestId);
  }
}

Future<void> _runGenerateChat(
  GenerateChatCommand cmd,
  _WorkerState state,
  SendPort reply,
) async {
  if (!_claimGenerate(cmd.requestId, state, reply)) return;

  final session = state.sessions[cmd.sessionId];
  if (session == null) {
    reply.send(EngineErrorResponse(
      cmd.requestId,
      'no session ${cmd.sessionId}',
    ));
    state.inFlightGenerateId = null;
    return;
  }

  final template = cmd.templateOverride ?? state.modelTemplate;
  if (template == null) {
    reply.send(EngineErrorResponse(
      cmd.requestId,
      'no chat template available; pass templateOverride or load a model with one',
    ));
    state.inFlightGenerateId = null;
    return;
  }

  try {
    final hasMedia = cmd.messages.any((m) => m.media.isNotEmpty);

    if (hasMedia) {
      final mtmd = state.multimodal;
      if (mtmd == null) {
        reply.send(EngineErrorResponse(
          cmd.requestId,
          'message contains media but engine has no multimodal context; '
          'pass MultimodalParams to LlamaEngine.spawn',
        ));
        state.inFlightGenerateId = null;
        return;
      }
      await _runGenerateChatMedia(
        cmd: cmd,
        template: template,
        mtmd: mtmd,
        state: state,
        reply: reply,
      );
    } else {
      final prompt = ChatTemplate.apply(
        template: template,
        messages: cmd.messages,
        addAssistant: true,
      );
      session.clear();
      session.appendText(prompt, addSpecial: false);
      await _streamSessionGenerate(
        requestId: cmd.requestId,
        session: session,
        sampler: cmd.sampler,
        maxTokens: cmd.maxTokens,
        state: state,
        reply: reply,
      );
    }
  } catch (e, st) {
    reply.send(EngineErrorResponse(cmd.requestId, '$e\n$st'));
  } finally {
    state.inFlightGenerateId = null;
    state.cancelledRequests.remove(cmd.requestId);
  }
}

Future<void> _runGenerateChatMedia({
  required GenerateChatCommand cmd,
  required String template,
  required MultimodalContext mtmd,
  required _WorkerState state,
  required SendPort reply,
}) async {
  // Render the chat into a single prompt; media markers stay inline in the
  // user message bodies for mtmd_tokenize to replace.
  final prompt = ChatTemplate.apply(
    template: template,
    messages: cmd.messages,
    addAssistant: true,
  );
  final flatMedia = [
    for (final m in cmd.messages) ...m.media,
  ];

  // Reset KV for this seq before the multimodal evaluator runs.
  final lib = LlamaLibrary.bindings;
  lib.llama_memory_seq_rm(
    lib.llama_get_memory(state.context.pointer),
    0,
    -1,
    -1,
  );

  final newNPast = mtmd.evalChunks(
    llamaContext: state.context,
    prompt: prompt,
    media: flatMedia,
    nPast: 0,
    seqId: 0,
    nBatch: state.context.nBatch,
    addSpecial: true,
    parseSpecial: true,
    logitsLast: true,
  );

  await _streamSampleAfterPrefill(
    requestId: cmd.requestId,
    state: state,
    reply: reply,
    sampler: cmd.sampler,
    maxTokens: cmd.maxTokens,
    seqId: 0,
    initialPos: newNPast,
  );
}

void _runSaveSessionState(
  SaveSessionStateCommand cmd,
  _WorkerState state,
  SendPort reply,
) {
  final session = state.sessions[cmd.sessionId];
  if (session == null) {
    reply.send(EngineErrorResponse(
      cmd.requestId,
      'no session ${cmd.sessionId}',
    ));
    return;
  }
  try {
    final raw = session.captureRawState();
    final tokens = session.tokens;
    final tokenBytes = Uint8List(tokens.length * 4)
      ..buffer.asInt32List().setAll(0, tokens);
    final metadata = _currentMetadata(
      state: state,
      session: session,
      tokensCount: tokens.length,
      tokensChecksum: fnv1a64(tokenBytes),
      extra: cmd.extra,
    );
    final fileBytes = encodeState(
      metadata: metadata,
      tokens: tokens,
      rawState: raw,
    );
    File(cmd.path).writeAsBytesSync(fileBytes, flush: true);
    reply.send(EngineAckResponse(cmd.requestId));
  } catch (e, st) {
    reply.send(EngineErrorResponse(cmd.requestId, '$e\n$st'));
  }
}

void _runLoadSessionState(
  LoadSessionStateCommand cmd,
  _WorkerState state,
  SendPort reply,
) {
  final session = state.sessions[cmd.sessionId];
  if (session == null) {
    reply.send(EngineErrorResponse(
      cmd.requestId,
      'no session ${cmd.sessionId}',
    ));
    return;
  }
  try {
    final fileBytes = File(cmd.path).readAsBytesSync();
    final decoded = decodeState(fileBytes);

    final actual = _currentMetadata(
      state: state,
      session: session,
      tokensCount: decoded.metadata.tokensCount,
      tokensChecksum: decoded.metadata.tokensChecksum,
    );
    verifyCompatible(decoded.metadata, actual);

    session.restoreRawState(
      decoded.rawState,
      decoded.tokens,
      decoded.metadata.kvHead,
    );

    reply.send(SessionStateLoadedResponse(
      cmd.requestId,
      extra: decoded.metadata.extra,
      tokensCount: decoded.metadata.tokensCount,
      kvHead: decoded.metadata.kvHead,
    ));
  } catch (e, st) {
    reply.send(EngineErrorResponse(cmd.requestId, '$e\n$st'));
  }
}

StateMetadata _currentMetadata({
  required _WorkerState state,
  required dynamic session,
  required int tokensCount,
  required int tokensChecksum,
  Map<String, Object?> extra = const <String, Object?>{},
}) {
  final mtmd = state.multimodal;
  return StateMetadata(
    codecVersion: stateCodecVersion,
    savedAt: DateTime.now().toUtc().toIso8601String(),
    wrapperVersion: '0.9.0-dev.0',
    modelPath: state.model.params.path,
    modelSizeBytes: state.model.sizeBytes,
    modelNParams: state.model.nParams,
    modelNEmbd: state.model.nEmbd,
    modelNLayer: state.model.nLayer,
    modelTrainCtx: state.model.trainCtx,
    nCtx: state.context.nCtx,
    nBatch: state.context.nBatch,
    nUbatch: state.context.nUbatch,
    nSeqMax: state.context.nSeqMax,
    embeddings: state.context.params.embeddings,
    mmprojPath: mtmd?.params.mmprojPath,
    mmprojSizeBytes:
        mtmd == null ? null : File(mtmd.params.mmprojPath).statSync().size,
    mmprojSupportsVision: mtmd?.supportsVision,
    mmprojSupportsAudio: mtmd?.supportsAudio,
    seqId: (session as dynamic).seqId as int,
    kvHead: (session as dynamic).kvHead as int,
    tokensCount: tokensCount,
    tokensChecksum: tokensChecksum,
    extra: extra,
  );
}

/// Sampling loop for paths where prefill was performed externally
/// (e.g. via mtmd_helper_eval_chunks). Mirrors [Generator]'s decode loop.
Future<void> _streamSampleAfterPrefill({
  required int requestId,
  required _WorkerState state,
  required SendPort reply,
  required SamplerParams sampler,
  required int maxTokens,
  required int seqId,
  required int initialPos,
}) async {
  final lib = LlamaLibrary.bindings;
  final ctx = state.context;
  final tokenizer = Tokenizer(state.model.vocab);
  final accumulator = Utf8Accumulator();
  final batch = LlamaBatch(ctx.nBatch);
  final samplerHandle = SamplerFactory.build(sampler);

  var pos = initialPos;
  var generated = 0;
  var cancelled = false;

  try {
    while (true) {
      if (state.cancelledRequests.remove(requestId)) {
        cancelled = true;
        break;
      }

      final token = samplerHandle.sample(ctx);
      samplerHandle.accept(token);
      final isEog = state.model.vocab.isEog(token);
      final bytes = tokenizer.encodeToken(token);

      if (isEog) {
        accumulator.clear();
        reply.send(EngineGenerationEvent(
          requestId,
          TokenEvent(id: token, bytes: bytes, text: '', position: pos),
        ));
        reply.send(EngineGenerationEvent(
          requestId,
          DoneEvent(
            reason: const StopEog(),
            generatedCount: generated + 1,
            committedPosition: pos,
          ),
        ));
        break;
      }

      final text = accumulator.accept(bytes);
      reply.send(EngineGenerationEvent(
        requestId,
        TokenEvent(id: token, bytes: bytes, text: text, position: pos),
      ));
      generated++;

      if (generated >= maxTokens) {
        reply.send(EngineGenerationEvent(
          requestId,
          DoneEvent(
            reason: const StopMaxTokens(),
            generatedCount: generated,
            committedPosition: pos,
            trailingText: accumulator.flush(),
          ),
        ));
        break;
      }

      batch.clear();
      batch.add(token, pos, [seqId], wantLogits: true);
      final rc = lib.llama_decode(ctx.pointer, batch.raw);
      if (rc != 0) {
        reply.send(EngineErrorResponse(
          requestId,
          'llama_decode failed during multimodal sampling: rc=$rc',
        ));
        return;
      }
      pos++;
    }

    if (cancelled) {
      reply.send(EngineGenerationCancelled(requestId));
    } else {
      reply.send(EngineGenerationFinished(requestId));
    }
  } finally {
    samplerHandle.dispose();
    batch.dispose();
  }
}

bool _claimGenerate(int requestId, _WorkerState state, SendPort reply) {
  if (state.inFlightGenerateId != null) {
    reply.send(EngineErrorResponse(
      requestId,
      'another generate is already in flight (request ${state.inFlightGenerateId})',
    ));
    return false;
  }
  state.inFlightGenerateId = requestId;
  return true;
}

Future<void> _streamSessionGenerate({
  required int requestId,
  required LlamaSession session,
  required SamplerParams sampler,
  required int maxTokens,
  required _WorkerState state,
  required SendPort reply,
  ContextShiftPolicy shiftPolicy = ContextShiftPolicy.off,
  ContextShift shift = ContextShift.defaults,
}) async {
  var cancelled = false;
  await for (final event in session.generate(
    sampler: sampler,
    maxTokens: maxTokens,
    shiftPolicy: shiftPolicy,
    shift: shift,
  )) {
    if (state.cancelledRequests.remove(requestId)) {
      cancelled = true;
      break;
    }
    reply.send(EngineGenerationEvent(requestId, event));
    if (event is DoneEvent) break;
  }
  if (cancelled) {
    reply.send(EngineGenerationCancelled(requestId));
  } else {
    reply.send(EngineGenerationFinished(requestId));
  }
}

void _shutdown(
  _WorkerState state,
  int requestId,
  SendPort reply,
  ReceivePort commandRx,
  Completer<void> completer,
) {
  for (final session in state.sessions.values) {
    try {
      session.clear();
    } catch (_) {/* ignore */}
  }
  state.sessions.clear();
  // We deliberately do NOT dispose the model or context here. With multiple
  // engines in a process, freeing one worker's model/context can crash the
  // other worker's outstanding operations because the backend is shared and
  // some teardown paths touch process-global state. The OS reclaims memory
  // on process exit; that is good enough for M3.
  LlamaLibrary.dispose();
  reply.send(EngineShutdownComplete(requestId));
  commandRx.close();
  completer.complete();
}
