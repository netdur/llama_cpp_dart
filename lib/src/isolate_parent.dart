import 'dart:async';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:typed_isolate/typed_isolate.dart';

class CompletionEvent {
  final String promptId;
  final bool success;
  CompletionEvent(this.promptId, this.success);
}

class LlamaParent {
  StreamController<String> _controller = StreamController<String>.broadcast();
  final _parent = IsolateParent<LlamaCommand, LlamaResponse>();

  StreamSubscription<LlamaResponse>? _subscription;
  // bool _isDone = false;
  bool _isGenerating = false;

  // Add status tracking
  LlamaStatus _status = LlamaStatus.uninitialized;
  LlamaStatus get status => _status;

  List<Map<String, dynamic>> messages = [];

  final LlamaLoad loadCommand;
  final PromptFormat? formatter;

  Completer<void>? _readyCompleter;

  Completer<List<double>>? _embeddingCompleter;

  LlamaParent(this.loadCommand, [this.formatter]);

  Stream<String> get stream => _controller.stream;
  bool get isGenerating => _isGenerating;

  // Add a completion stream
  final _completionController = StreamController<CompletionEvent>.broadcast();
  Stream<CompletionEvent> get completions => _completionController.stream;

  // Track current prompt ID
  String _currentPromptId = "";

  void _onData(LlamaResponse data) {
    if (_embeddingCompleter != null && !_embeddingCompleter!.isCompleted) {
      if (data.embeddings != null) {
        // Success
        _embeddingCompleter!.complete(data.embeddings!);
      } else {
        // Assume error if embeddings are null (use the error field if available)
        _embeddingCompleter!.completeError(
            LlamaException(data.error ?? "Unknown error getting embeddings"));
      }
      _embeddingCompleter = null; // Reset completer
      // Update status if provided in the response (optional refinement)
      if (data.status != null) {
        _status = data.status!;
      }
      return; // IMPORTANT: Stop processing here for embedding responses
    }

    if (data.status != null) {
      _status = data.status!;
      if (data.status == LlamaStatus.ready && !_readyCompleter!.isCompleted) {
        _readyCompleter!.complete();
      }
    }

    if (data.text.isNotEmpty) {
      _controller.add(data.text);
    }

    if (data.isDone) {
      _isGenerating = false;
      _completionController.add(CompletionEvent(_currentPromptId, true));
    }
  }

  Future<void> init() async {
    _readyCompleter = Completer<void>();

    // _isDone = false;
    _isGenerating = false;
    _status = LlamaStatus.uninitialized;
    _parent.init();

    // Cancel any existing subscription first
    await _subscription?.cancel();
    _subscription = _parent.stream.listen(_onData);

    await _parent.spawn(LlamaChild());

    _parent.sendToChild(
        data: LlamaInit(
            Llama.libraryPath,
            loadCommand.modelParams,
            loadCommand.contextParams,
            loadCommand.samplingParams,
            formatter ?? loadCommand.format),
        id: 1);

    _parent.sendToChild(data: loadCommand, id: 1);

    // Wait for model to be ready
    // Note: This assumes the LlamaChild will send a response with ready status
    // _status = LlamaStatus.ready;
    await _readyCompleter!.future;
    // _readyCompleter!.complete();
  }

  // Reset internal state for new generation
  Future<void> _reset() async {
    // If we're currently generating, stop it first
    if (_isGenerating) {
      await _stopGeneration();
    }

    // _isDone = false;

    // Recreate the stream controller if it was closed
    if (_controller.isClosed) {
      _controller = StreamController<String>.broadcast();

      // Reattach the subscription
      await _subscription?.cancel();
      _subscription = _parent.stream.listen(_onData);
    }

    // Clear the llama context
    _parent.sendToChild(id: 1, data: LlamaClear());

    // Set status to ready after clearing
    _status = LlamaStatus.ready;

    // Wait for the clear to complete
    await Future.delayed(Duration(milliseconds: 200));
  }

  // Stop any ongoing generation
  Future<void> _stopGeneration() async {
    if (_isGenerating) {
      _parent.sendToChild(id: 1, data: LlamaStop());
      _isGenerating = false;

      // Set status to ready after stopping
      _status = LlamaStatus.ready;

      // Give a moment for the stop to process
      await Future.delayed(Duration(milliseconds: 200));
    }
  }

  // Enhanced sendPrompt that auto-resets and handles ongoing generations
  Future<String> sendPrompt(String prompt) async {
    if (loadCommand.contextParams.embeddings) {
      throw StateError(
          "This LlamaParent instance is configured for embeddings only and cannot generate text.");
    }

    // Auto-reset before sending a new prompt
    await _reset();

    // Generate a unique ID for this prompt (or accept one as parameter)
    _currentPromptId = DateTime.now().millisecondsSinceEpoch.toString();

    // Mark that we're now generating and update status
    _isGenerating = true;
    _status = LlamaStatus.generating;

    final formattedPrompt = messages.isEmpty
        ? (formatter?.formatPrompt(prompt) ?? prompt)
        : (formatter?.formatMessages(messages) ?? prompt);

    _parent.sendToChild(id: 1, data: LlamaPrompt(formattedPrompt));

    return _currentPromptId;
  }

  // Explicit stop method (can be called by user if needed)
  Future<void> stop() async {
    await _stopGeneration();
  }

  Future<void> dispose() async {
    // _isDone = true;
    _isGenerating = false;
    _status = LlamaStatus.disposed;

    await _subscription?.cancel();

    // Check if controller is already closed before trying to close it
    if (!_controller.isClosed) {
      await _controller.close();
    }

    _parent.sendToChild(id: 1, data: LlamaClear());

    _parent.dispose();
  }

  Future<List<double>> getEmbeddings(String prompt,
      {bool addBos = true, bool normalize = true}) async {
    // 1. Check configuration: Must be an embedding instance
    if (!loadCommand.contextParams.embeddings) {
      throw StateError(
          "This LlamaParent instance is configured for text generation only and cannot get embeddings.");
    }

    // 2. Basic Status Check (use whatever check was implicitly working before for sendPrompt/init)
    if (_status != LlamaStatus.ready) {
      // If init needs awaiting (based on your original usage)
      if (_readyCompleter != null && !_readyCompleter!.isCompleted) {
        await _readyCompleter!.future;
      }
      // Check again after potential wait
      if (_status != LlamaStatus.ready) {
        throw StateError('Llama is not ready. Current status: $_status');
      }
    }

    // 3. Prevent concurrent embedding requests (still useful within embedding mode)
    if (_embeddingCompleter != null && !_embeddingCompleter!.isCompleted) {
      throw StateError('Another embedding request is already in progress.');
    }

    // 4. Create completer and send command
    _embeddingCompleter = Completer<List<double>>();
    _parent.sendToChild(
      id: 1,
      data: LlamaGetEmbeddings(prompt, addBos: addBos, normalize: normalize),
    );

    // 5. Return the future (add timeout if desired)
    // return _embeddingCompleter!.future.timeout(Duration(seconds: 60)); // Optional
    return _embeddingCompleter!.future;
  }
}
