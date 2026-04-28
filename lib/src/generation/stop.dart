/// Reason a [Generator] stopped emitting tokens.
sealed class StopReason {
  const StopReason();
}

/// The sampled token was an end-of-generation token.
final class StopEog extends StopReason {
  const StopEog();
  @override
  String toString() => 'StopEog';
}

/// The configured `maxTokens` budget was reached.
final class StopMaxTokens extends StopReason {
  const StopMaxTokens();
  @override
  String toString() => 'StopMaxTokens';
}

/// The consumer cancelled the stream.
final class StopUserAbort extends StopReason {
  const StopUserAbort();
  @override
  String toString() => 'StopUserAbort';
}
