import 'dart:async';
import 'dart:typed_data';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// A scope that filters responses from LlamaParent for specific prompt IDs
class LlamaScope {
  final LlamaParent _parent;
  final String id = "scope_${DateTime.now().microsecondsSinceEpoch}";
  final Set<String> _promptIds = {};
  final StreamController<String> _streamController =
      StreamController<String>.broadcast();
  final StreamController<CompletionEvent> _completionController =
      StreamController<CompletionEvent>.broadcast();

  /// Stream of text generated for prompts sent through this scope
  Stream<String> get stream => _streamController.stream;

  /// Stream of completion events for prompts sent through this scope
  Stream<CompletionEvent> get completions => _completionController.stream;

  /// Create a new scope for the given parent
  LlamaScope(this._parent);

  /// Send a prompt to the model and track its ID in this scope
  Future<String> sendPrompt(String prompt) async {
    final promptId = await _parent.sendPrompt(prompt, scope: this);
    _promptIds.add(promptId);
    return promptId;
  }

  /// Send a prompt with images to the model and track its ID in this scope
  Future<String> sendPromptWithImages(
      String prompt, List<LlamaImage> images) async {
    final promptId =
        await _parent.sendPromptWithImages(prompt, images, scope: this);
    _promptIds.add(promptId);
    return promptId;
  }

  /// Handle a response from the parent
  void handleResponse(LlamaResponse response) {
    if (response.promptId != null &&
        _promptIds.contains(response.promptId) &&
        response.text.isNotEmpty) {
      _streamController.add(response.text);
    }
  }

  /// Handle a completion event from the parent
  void handleCompletion(CompletionEvent event) {
    if (_promptIds.contains(event.promptId)) {
      _completionController.add(event);
      _promptIds.remove(event.promptId);
    }
  }

  /// Add a prompt ID to this scope (used internally)
  void addPromptId(String promptId) {
    _promptIds.add(promptId);
  }

  /// Stop generation for this scope only.
  Future<void> stop({bool alsoCancelQueued = true}) {
    return _parent.cancelScope(this,
        cancelInFlight: true, cancelQueued: alsoCancelQueued);
  }

  /// Captures the current KV Cache of this scope from VRAM.
  /// Returns the serialized state as a byte array (Tier 2).
  Future<Uint8List> saveState() async {
    return await _parent.saveState(this);
  }

  /// Restores a serialized state from RAM into VRAM for this scope.
  Future<void> loadState(Uint8List data) async {
    await _parent.loadState(this, data);
  }

  /// Restores a session directly from a Disk file into VRAM (Tier 3).
  Future<void> loadSession(String path) async {
    await _parent.loadSession(this, path);
  }

  /// Dispose of resources and FREE THE SLOT.
  Future<void> dispose() async {
    if (!_streamController.isClosed) await _streamController.close();
    if (!_completionController.isClosed) await _completionController.close();
    
    await _parent.disposeScope(this);
  }
}
