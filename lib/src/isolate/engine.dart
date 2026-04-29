import 'dart:async';
import 'dart:isolate';

import '../chat/chat_message.dart';
import '../context/context_params.dart';
import '../ffi/backends.dart';
import '../generation/context_shift.dart';
import '../generation/event.dart';
import '../model/model_params.dart';
import '../multimodal/media.dart';
import '../multimodal/multimodal_params.dart';
import '../sampling/sampler_params.dart';
import '../types/exceptions.dart';
import 'messages.dart';
import 'worker.dart';

/// Off-thread llama.cpp inference engine.
///
/// Spawns a worker isolate that owns the native model and context. The
/// main isolate gets a thin handle that posts commands and consumes
/// streaming events.
final class LlamaEngine {
  final Isolate _isolate;
  final SendPort _commandPort;
  final ReceivePort _responsePort;
  late final StreamSubscription<dynamic> _responseSub;

  /// Default chat template embedded in the loaded model's GGUF metadata,
  /// or `null` if the model has none. When `null`, [createChat] requires
  /// the caller to supply a `templateOverride` on every generate call.
  String? get modelChatTemplate => _modelChatTemplate;
  String? _modelChatTemplate;

  /// True when the worker successfully loaded a multimodal projector.
  bool get multimodalLoaded => _multimodalLoaded;
  bool _multimodalLoaded = false;

  /// True if the projector handles image input.
  bool get supportsVision => _supportsVision;
  bool _supportsVision = false;

  /// True if the projector handles audio input.
  bool get supportsAudio => _supportsAudio;
  bool _supportsAudio = false;

  /// Audio sample rate the projector expects (e.g. 16000), or `-1` if
  /// audio isn't supported.
  int get audioSampleRate => _audioSampleRate;
  int _audioSampleRate = -1;

  /// True if the loaded context's memory backend supports
  /// `llama_memory_seq_add` (i.e., position shifting). When false, callers
  /// must keep `ContextShiftPolicy.off` and handle context overflow at
  /// the application level. False on recurrent caches and most iSWA
  /// configurations (e.g. some Qwen3, Gemma 3 with SWA).
  bool get canShift => _canShift;
  bool _canShift = true;

  /// Snapshot of every ggml-backend device the runtime loaded inside
  /// the worker isolate. Use to tell whether Hexagon / OpenCL / Metal
  /// is actually available: if there's no entry whose `registryName`
  /// is `Hexagon`, the Hexagon AAR's DSP libs didn't load. The list
  /// is captured once at engine spawn and never changes thereafter.
  List<BackendDevice> get devices => _devices;
  List<BackendDevice> _devices = const <BackendDevice>[];

  /// Convenience: true if any non-CPU device is loaded.
  bool get hasAccelerator =>
      _devices.any((d) => d.type != BackendDeviceType.cpu);

  /// Convenience: name of the highest-priority accelerator (Hexagon
  /// before OpenCL before Metal before integrated GPU). `null` if the
  /// runtime is CPU-only.
  String? get primaryAcceleratorName {
    const order = [
      BackendDeviceType.accel, // Hexagon, ANE
      BackendDeviceType.gpu, // discrete GPU
      BackendDeviceType.igpu, // OpenCL on Adreno, Apple iGPU, etc.
    ];
    for (final t in order) {
      for (final d in _devices) {
        if (d.type == t) return d.name;
      }
    }
    return null;
  }

  final Map<int, Completer<EngineResponse>> _pending =
      <int, Completer<EngineResponse>>{};
  final Map<int, StreamController<GenerationEvent>> _streams =
      <int, StreamController<GenerationEvent>>{};
  final Set<int> _sessionIds = <int>{};
  int _nextRequestId = 1;
  int _nextSessionId = 1;
  bool _disposed = false;

  LlamaEngine._(this._isolate, this._commandPort, this._responsePort);

  /// Spawn a worker isolate that loads llama.cpp from a dylib path.
  ///
  /// For iOS / macOS apps where `llama.xcframework` was embedded via
  /// Xcode, use [spawnFromProcess] instead — no path is needed.
  static Future<LlamaEngine> spawn({
    required String libraryPath,
    required ModelParams modelParams,
    required ContextParams contextParams,
    MultimodalParams? multimodalParams,
    String? backendDirectory,
  }) {
    return _spawnInternal(
      libraryPath: libraryPath,
      useProcessSymbols: false,
      backendDirectory: backendDirectory,
      modelParams: modelParams,
      contextParams: contextParams,
      multimodalParams: multimodalParams,
    );
  }

  static Future<LlamaEngine> _spawnInternal({
    required String libraryPath,
    required bool useProcessSymbols,
    String? backendDirectory,
    required ModelParams modelParams,
    required ContextParams contextParams,
    MultimodalParams? multimodalParams,
  }) async {
    final responsePort = ReceivePort();
    final ready = Completer<EngineResponse>();
    LlamaEngine? engine;

    // Single listener for the whole port lifetime. Pre-ready it satisfies
    // the bootstrap completer; post-ready it dispatches to the engine.
    final sub = responsePort.listen((dynamic msg) {
      final live = engine;
      if (live != null) {
        live._handleResponse(msg);
        return;
      }
      if (msg is EngineReadyResponse && !ready.isCompleted) {
        ready.complete(msg);
      } else if (msg is EngineErrorResponse && !ready.isCompleted) {
        ready.completeError(LlamaLibraryException(msg.message));
      }
    });

    final isolate = await Isolate.spawn<EngineBootstrap>(
      runEngineWorker,
      EngineBootstrap(
        replyPort: responsePort.sendPort,
        libraryPath: libraryPath,
        useProcessSymbols: useProcessSymbols,
        modelParams: modelParams,
        contextParams: contextParams,
        multimodalParams: multimodalParams,
        backendDirectory: backendDirectory,
      ),
      errorsAreFatal: true,
      debugName: 'llama_cpp_dart.worker',
    );

    final EngineResponse response;
    try {
      response = await ready.future;
    } catch (_) {
      await sub.cancel();
      responsePort.close();
      isolate.kill(priority: Isolate.immediate);
      rethrow;
    }
    final readyResponse = response as EngineReadyResponse;

    final built =
        LlamaEngine._(isolate, readyResponse.commandPort, responsePort);
    built._modelChatTemplate = readyResponse.modelChatTemplate;
    built._multimodalLoaded = readyResponse.multimodalLoaded;
    built._supportsVision = readyResponse.supportsVision;
    built._supportsAudio = readyResponse.supportsAudio;
    built._audioSampleRate = readyResponse.audioSampleRate;
    built._canShift = readyResponse.canShift;
    built._devices = readyResponse.devices;
    built._responseSub = sub;
    engine = built;
    return built;
  }

  /// Spawn a worker isolate using llama.cpp symbols already linked into
  /// the running process. Use this on iOS / macOS app builds that embed
  /// `llama.xcframework` — Xcode static-links the framework into the
  /// app binary so there's no `.dylib` to dlopen.
  ///
  /// `dart test` and standalone Dart CLI runs should use [spawn] with a
  /// dylib path instead.
  static Future<LlamaEngine> spawnFromProcess({
    required ModelParams modelParams,
    required ContextParams contextParams,
    MultimodalParams? multimodalParams,
  }) async {
    return _spawnInternal(
      libraryPath: '<process>',
      useProcessSymbols: true,
      modelParams: modelParams,
      contextParams: contextParams,
      multimodalParams: multimodalParams,
    );
  }

  /// Create a chat handle backed by a fresh session. The handle keeps a
  /// list of [ChatMessage]s and renders them via the model's embedded
  /// chat template on each [EngineChat.generate].
  Future<EngineChat> createChat({int seqId = 0}) async {
    _ensureAlive();
    final session = await createSession(seqId: seqId);
    return EngineChat._(this, session);
  }

  /// Create a session bound to this engine. Sessions hold their own token
  /// history and KV-cache cursor.
  ///
  /// [seqId] must be `< nSeqMax` of the engine's context. Defaults to `0`,
  /// matching the single-session model. Apps with `nSeqMax > 1` must pass
  /// distinct ids explicitly.
  Future<EngineSession> createSession({int seqId = 0}) async {
    _ensureAlive();
    final sessionId = _nextSessionId++;
    await _request<EngineAckResponse>((id) => CreateSessionCommand(
          id,
          sessionId: sessionId,
          seqId: seqId,
        ));
    _sessionIds.add(sessionId);
    return EngineSession._(this, sessionId, seqId);
  }

  /// Shut down the worker isolate. Cancels all in-flight streams.
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;

    for (final c in _streams.values) {
      if (!c.isClosed) {
        c.addError(const LlamaLibraryException('LlamaEngine disposed'));
        await c.close();
      }
    }
    _streams.clear();

    final id = _nextRequestId++;
    final completer = Completer<EngineResponse>();
    _pending[id] = completer;
    _commandPort.send(ShutdownCommand(id));
    try {
      await completer.future.timeout(const Duration(seconds: 2), onTimeout: () {
        return EngineShutdownComplete(id);
      });
    } catch (_) {/* tear down regardless */}
    _pending.clear();

    await _responseSub.cancel();
    _responsePort.close();
    _isolate.kill(priority: Isolate.beforeNextEvent);
  }

  Future<T> _request<T extends EngineResponse>(
    EngineCommand Function(int id) build,
  ) async {
    _ensureAlive();
    final id = _nextRequestId++;
    final completer = Completer<EngineResponse>();
    _pending[id] = completer;
    _commandPort.send(build(id));
    final response = await completer.future;
    if (response is EngineErrorResponse) {
      throw LlamaLibraryException(response.message);
    }
    return response as T;
  }

  Stream<GenerationEvent> _generate({
    required int sessionId,
    required String? prompt,
    required bool addSpecial,
    required bool parseSpecial,
    required SamplerParams sampler,
    required int maxTokens,
    List<LlamaMedia> media = const <LlamaMedia>[],
    ContextShiftPolicy shiftPolicy = ContextShiftPolicy.off,
    ContextShift shift = ContextShift.defaults,
  }) {
    return _streamGenerate(
      build: (id) => GenerateCommand(
        id,
        sessionId: sessionId,
        prompt: prompt,
        addSpecial: addSpecial,
        parseSpecial: parseSpecial,
        sampler: sampler,
        maxTokens: maxTokens,
        media: media,
        shiftPolicy: shiftPolicy,
        shift: shift,
      ),
    );
  }

  Stream<GenerationEvent> _generateChat({
    required int sessionId,
    required List<ChatMessage> messages,
    required SamplerParams sampler,
    required int maxTokens,
    String? templateOverride,
  }) {
    return _streamGenerate(
      build: (id) => GenerateChatCommand(
        id,
        sessionId: sessionId,
        messages: messages,
        sampler: sampler,
        maxTokens: maxTokens,
        templateOverride: templateOverride,
      ),
    );
  }

  Stream<GenerationEvent> _streamGenerate({
    required EngineCommand Function(int id) build,
  }) {
    _ensureAlive();
    final id = _nextRequestId++;
    late StreamController<GenerationEvent> controller;
    var sentCancel = false;

    controller = StreamController<GenerationEvent>(
      onCancel: () {
        if (sentCancel || _disposed) return;
        sentCancel = true;
        _commandPort.send(
          CancelCommand(_nextRequestId++, targetRequestId: id),
        );
        _streams.remove(id);
      },
    );

    _streams[id] = controller;
    _commandPort.send(build(id));
    return controller.stream;
  }

  Future<void> _disposeSession(int sessionId) async {
    if (_disposed) return;
    if (!_sessionIds.contains(sessionId)) return;
    await _request<EngineAckResponse>(
      (id) => DisposeSessionCommand(id, sessionId: sessionId),
    );
    _sessionIds.remove(sessionId);
  }

  Future<void> _appendText(
    int sessionId,
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) async {
    await _request<EngineAckResponse>(
      (id) => AppendTextCommand(
        id,
        sessionId: sessionId,
        text: text,
        addSpecial: addSpecial,
        parseSpecial: parseSpecial,
      ),
    );
  }

  Future<void> _clearSession(int sessionId) async {
    await _request<EngineAckResponse>(
      (id) => ClearSessionCommand(id, sessionId: sessionId),
    );
  }

  Future<void> _saveSessionState({
    required int sessionId,
    required String path,
    required Map<String, Object?> extra,
  }) async {
    await _request<EngineAckResponse>(
      (id) => SaveSessionStateCommand(
        id,
        sessionId: sessionId,
        path: path,
        extra: extra,
      ),
    );
  }

  Future<SessionStateLoadedResponse> _loadSessionState({
    required int sessionId,
    required String path,
  }) async {
    return _request<SessionStateLoadedResponse>(
      (id) => LoadSessionStateCommand(
        id,
        sessionId: sessionId,
        path: path,
      ),
    );
  }

  void _handleResponse(dynamic msg) {
    if (msg is! EngineResponse) return;
    final id = msg.requestId;

    if (msg is EngineGenerationEvent) {
      final c = _streams[id];
      if (c != null && !c.isClosed) c.add(msg.event);
      return;
    }
    if (msg is EngineGenerationFinished || msg is EngineGenerationCancelled) {
      final c = _streams.remove(id);
      if (c != null && !c.isClosed) c.close();
      return;
    }
    if (msg is EngineErrorResponse) {
      final c = _streams.remove(id);
      if (c != null && !c.isClosed) {
        c.addError(LlamaLibraryException(msg.message));
        c.close();
        return;
      }
      final pending = _pending.remove(id);
      if (pending != null && !pending.isCompleted) pending.complete(msg);
      return;
    }
    final pending = _pending.remove(id);
    if (pending != null && !pending.isCompleted) pending.complete(msg);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw const LlamaLibraryException('LlamaEngine has been disposed.');
    }
  }
}

/// A handle to a session running inside the engine worker isolate.
final class EngineSession {
  final LlamaEngine _engine;
  final int sessionId;
  final int seqId;
  bool _disposed = false;

  EngineSession._(this._engine, this.sessionId, this.seqId);

  /// Append text to the session's history without generating.
  Future<void> append(
    String text, {
    bool addSpecial = false,
    bool parseSpecial = true,
  }) async {
    _ensureAlive();
    await _engine._appendText(
      sessionId,
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }

  /// Clear the session's tokens and KV cache.
  Future<void> clear() async {
    _ensureAlive();
    await _engine._clearSession(sessionId);
  }

  /// Generate tokens. If [prompt] is non-null and non-empty it is appended
  /// before generation; otherwise generation continues from the existing
  /// history (which must contain pending uncommitted tokens).
  ///
  /// When [media] is non-empty the prompt must contain one media marker
  /// per item (default `<__media__>`) and the engine must have been
  /// spawned with `multimodalParams`. Media generation resets the session's
  /// KV cache before encoding.
  ///
  /// [shiftPolicy] controls what happens when the KV cache is one slot
  /// from full. `off` (default) raises a [LlamaDecodeException]; `auto`
  /// performs a llama-server-style context shift mid-stream. `auto`
  /// requires `engine.canShift == true` and is incompatible with media.
  Stream<GenerationEvent> generate({
    String? prompt,
    bool addSpecial = false,
    bool parseSpecial = true,
    SamplerParams sampler = const SamplerParams(),
    int maxTokens = 256,
    List<LlamaMedia> media = const <LlamaMedia>[],
    ContextShiftPolicy shiftPolicy = ContextShiftPolicy.off,
    ContextShift shift = ContextShift.defaults,
  }) {
    _ensureAlive();
    return _engine._generate(
      sessionId: sessionId,
      prompt: prompt,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
      sampler: sampler,
      maxTokens: maxTokens,
      media: media,
      shiftPolicy: shiftPolicy,
      shift: shift,
    );
  }

  /// Persist the session's KV state + token history to [path]. The file
  /// includes wrapper metadata so [loadState] can verify compatibility on
  /// reload. [extra] is application-defined and round-tripped verbatim.
  Future<void> saveState(
    String path, {
    Map<String, Object?> extra = const <String, Object?>{},
  }) async {
    _ensureAlive();
    await _engine._saveSessionState(
      sessionId: sessionId,
      path: path,
      extra: extra,
    );
  }

  /// Restore a previously saved state, returning the application-defined
  /// `extra` map that was attached at save time. Throws if the file isn't
  /// compatible with the running engine.
  Future<Map<String, Object?>> loadState(String path) async {
    _ensureAlive();
    final response = await _engine._loadSessionState(
      sessionId: sessionId,
      path: path,
    );
    return response.extra;
  }

  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    await _engine._disposeSession(sessionId);
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('EngineSession has been disposed.');
    }
  }
}

/// A multi-turn chat handle backed by an [EngineSession].
///
/// Holds a list of [ChatMessage]s. Each call to [generate] renders the
/// full message list via the model's embedded chat template, runs it
/// through the worker, and appends the assistant's reply to the history
/// when generation completes.
///
/// The underlying session's KV cache is cleared at the start of each
/// turn — incremental KV reuse across turns is a future optimization.
final class EngineChat {
  final LlamaEngine _engine;
  final EngineSession _session;
  final List<ChatMessage> _messages = <ChatMessage>[];
  bool _disposed = false;

  EngineChat._(this._engine, this._session);

  /// Read-only view of the conversation so far.
  List<ChatMessage> get messages => List.unmodifiable(_messages);

  /// Number of messages currently in history.
  int get messageCount => _messages.length;

  /// Underlying session id, useful for diagnostics.
  int get sessionId => _session.sessionId;

  /// Add a system message to the conversation.
  void addSystem(String content) {
    _ensureAlive();
    _messages.add(ChatMessage.system(content));
  }

  /// Add a user message to the conversation.
  ///
  /// If [media] is non-empty and [content] does not already include media
  /// markers (`<__media__>` by default), the markers are auto-prepended in
  /// the order matching [media]. The engine must have been spawned with
  /// `multimodalParams` for media to be processed.
  void addUser(
    String content, {
    List<LlamaMedia> media = const <LlamaMedia>[],
    String marker = '<__media__>',
  }) {
    _ensureAlive();
    var actualContent = content;
    if (media.isNotEmpty && !content.contains(marker)) {
      final markers = List<String>.filled(media.length, marker).join('\n');
      actualContent = markers.isEmpty
          ? content
          : (content.isEmpty ? markers : '$markers\n$content');
    }
    _messages.add(
      ChatMessage(role: 'user', content: actualContent, media: media),
    );
  }

  /// Add an assistant message to the conversation. Useful for seeding
  /// few-shot examples; the engine appends its own assistant reply after
  /// each [generate].
  void addAssistant(String content) {
    _ensureAlive();
    _messages.add(ChatMessage.assistant(content));
  }

  /// Add a generic message with arbitrary [role] (e.g. `tool`).
  void addMessage(ChatMessage message) {
    _ensureAlive();
    _messages.add(message);
  }

  /// Drop all messages.
  void clearHistory() {
    _ensureAlive();
    _messages.clear();
  }

  /// Render the current message list and stream the assistant's reply.
  ///
  /// On successful completion the assistant's full reply is appended to
  /// [messages] as `ChatMessage.assistant(...)`. If the stream is cancelled
  /// or errors, the partial reply is appended instead.
  ///
  /// [templateOverride] supplies a Jinja template string to use instead of
  /// the model's embedded one. Required when the model has no template.
  Stream<GenerationEvent> generate({
    SamplerParams sampler = const SamplerParams(),
    int maxTokens = 512,
    String? templateOverride,
  }) async* {
    _ensureAlive();
    if (_messages.isEmpty) {
      throw StateError('EngineChat has no messages; addUser/addSystem first.');
    }

    final replyBuf = StringBuffer();
    var committed = false;

    void commitReply() {
      if (committed) return;
      committed = true;
      _messages.add(ChatMessage.assistant(replyBuf.toString()));
    }

    try {
      await for (final event in _engine._generateChat(
        sessionId: _session.sessionId,
        messages: List<ChatMessage>.unmodifiable(_messages),
        sampler: sampler,
        maxTokens: maxTokens,
        templateOverride: templateOverride,
      )) {
        switch (event) {
          case TokenEvent():
            replyBuf.write(event.text);
            yield event;
          case ShiftEvent():
            yield event;
          case DoneEvent():
            if (event.trailingText.isNotEmpty) {
              replyBuf.write(event.trailingText);
            }
            commitReply();
            yield event;
        }
      }
      // Stream closed without DoneEvent (cancel or error). Still preserve
      // whatever was generated so the conversation reflects reality.
      commitReply();
    } catch (_) {
      commitReply();
      rethrow;
    }
  }

  /// Persist the chat (KV state + token history + message list) to [path].
  Future<void> saveState(String path) async {
    _ensureAlive();
    await _engine._saveSessionState(
      sessionId: _session.sessionId,
      path: path,
      extra: <String, Object?>{
        'messages': _messages.map((m) => m.toJson()).toList(),
      },
    );
  }

  /// Reload a previously saved state. Replaces the current message list
  /// and KV cache. Throws [LlamaLibraryException] (wrapping the worker's
  /// `LlamaStateException`) if the file isn't compatible.
  Future<void> loadState(String path) async {
    _ensureAlive();
    final response = await _engine._loadSessionState(
      sessionId: _session.sessionId,
      path: path,
    );
    final raw = response.extra['messages'];
    _messages.clear();
    if (raw is List) {
      for (final m in raw) {
        if (m is Map) {
          _messages.add(ChatMessage.fromJson(m.cast<String, Object?>()));
        }
      }
    }
  }

  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    await _session.dispose();
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('EngineChat has been disposed.');
    }
  }
}
