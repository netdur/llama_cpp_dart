/// Event signaling the completion of a prompt generation task.
class CompletionEvent {
  /// The unique identifier of the prompt/request that completed.
  final String promptId;

  /// Indicates whether the generation completed successfully.
  final bool success;

  /// If success is false, this contains the error message.
  final String? error;

  CompletionEvent(this.promptId, this.success, [this.error]);

  @override
  String toString() {
    return 'CompletionEvent(promptId: $promptId, success: $success, error: $error)';
  }
}