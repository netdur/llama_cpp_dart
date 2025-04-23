/// Event signaling the completion of a prompt generation task.
class CompletionEvent {
  /// The unique identifier of the request that completed.
  /// Will be null if the completion corresponds to a prompt sent via
  /// the global `LlamaParent.sendPrompt` method.
  final String? requestId;

  /// Indicates whether the generation completed successfully or ended in error.
  final bool success;

  CompletionEvent({
    required this.requestId,
    required this.success,
  });

  @override
  String toString() {
    return 'CompletionEvent(requestId: $requestId, success: $success)';
  }
}
