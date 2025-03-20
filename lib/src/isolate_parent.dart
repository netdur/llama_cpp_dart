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
  bool _isDone = false;
  bool _isGenerating = false;

  List<Map<String, dynamic>> messages = [];

  final LlamaLoad loadCommand;
  final PromptFormat? formatter;

  LlamaParent(this.loadCommand, [this.formatter]);

  Stream<String> get stream => _controller.stream;
  bool get isGenerating => _isGenerating;

  // Add a completion stream
  final _completionController = StreamController<CompletionEvent>.broadcast();
  Stream<CompletionEvent> get completions => _completionController.stream;

  // Track current prompt ID
  String _currentPromptId = "";

  void _onData(LlamaResponse data) {
    // print("data.isDone ${data.isDone}");

    // First check if we're already done to avoid processing more data
    if (_isDone) {
      return;
    }

    // Mark as done if this is the final response
    if (data.isDone) {
      _isDone = true;
      _isGenerating = false;

      // Emit completion event
      _completionController.add(CompletionEvent(_currentPromptId, true));
    }

    // Process the response text regardless
    _parseResponse(data.text);

    /*
    // Close the controller only after processing the last piece of text
    if (_isDone) {
      // Use a small delay to ensure all processing is complete
      Future.delayed(Duration.zero, () {
        if (!_controller.isClosed) {
          _controller.close();
        }
      });
    }
    */
  }

  void _parseResponse(String response) {
    // Check if controller is already closed
    if (_controller.isClosed) {
      return;
    }

    final processed =
        formatter == null ? response : formatter!.filterResponse(response);
    if (processed != null) {
      _controller.add(processed);
    }
  }

  Future<void> init() async {
    _isDone = false;
    _isGenerating = false;
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
  }

  // Reset internal state for new generation
  Future<void> _reset() async {
    // If we're currently generating, stop it first
    if (_isGenerating) {
      await _stopGeneration();
    }

    _isDone = false;

    // Recreate the stream controller if it was closed
    if (_controller.isClosed) {
      _controller = StreamController<String>.broadcast();

      // Reattach the subscription
      await _subscription?.cancel();
      _subscription = _parent.stream.listen(_onData);
    }

    // Clear the llama context
    _parent.sendToChild(id: 1, data: LlamaClear());

    // Wait for the clear to complete
    await Future.delayed(Duration(milliseconds: 200));
  }

  // Stop any ongoing generation
  Future<void> _stopGeneration() async {
    if (_isGenerating) {
      _parent.sendToChild(id: 1, data: LlamaStop());
      _isGenerating = false;

      // Give a moment for the stop to process
      await Future.delayed(Duration(milliseconds: 200));
    }
  }

  // Enhanced sendPrompt that auto-resets and handles ongoing generations
  Future<String> sendPrompt(String prompt) async {
    // Auto-reset before sending a new prompt
    await _reset();

    // Generate a unique ID for this prompt (or accept one as parameter)
    _currentPromptId = DateTime.now().millisecondsSinceEpoch.toString();

    // Mark that we're now generating
    _isGenerating = true;

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
    _isDone = true;
    _isGenerating = false;

    await _subscription?.cancel();

    // Check if controller is already closed before trying to close it
    if (!_controller.isClosed) {
      await _controller.close();
    }

    _parent.sendToChild(id: 1, data: LlamaClear());

    _parent.dispose();
  }
}
