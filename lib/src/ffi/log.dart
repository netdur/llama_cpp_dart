import 'dart:ffi';

import 'library_loader.dart';

/// Control over llama.cpp / ggml log output.
///
/// llama.cpp logs unconditionally to stderr by default. [silence] redirects
/// every log call to a no-op C function; [useDefault] restores stderr.
///
/// Custom Dart-side log callbacks are intentionally not exposed yet — the
/// underlying logger fires from arbitrary threads, which is unsafe for
/// `Pointer.fromFunction` callbacks. A thread-safe variant arrives with
/// the engine-isolate work in M3.
final class LlamaLog {
  LlamaLog._();

  static Pointer<NativeFunction<_LogCallbackC>>? _silentCallback;

  /// Suppress all llama.cpp / ggml logging. Idempotent.
  static void silence() {
    _silentCallback ??= Pointer.fromFunction<_LogCallbackC>(_noop);
    final cb = _silentCallback!;
    LlamaLibrary.bindings.llama_log_set(cb, nullptr);
    LlamaLibrary.bindings.ggml_log_set(cb, nullptr);
  }

  /// Restore the default behavior (logs printed to stderr by llama.cpp).
  static void useDefault() {
    LlamaLibrary.bindings
        .llama_log_set(nullptr.cast<NativeFunction<_LogCallbackC>>(), nullptr);
    LlamaLibrary.bindings
        .ggml_log_set(nullptr.cast<NativeFunction<_LogCallbackC>>(), nullptr);
  }

  static void _noop(int level, Pointer<Char> text, Pointer<Void> userData) {
    // Intentionally empty. Must not touch any Dart objects: this may be
    // invoked from a non-Dart worker thread where dereferencing the Dart
    // runtime is unsafe.
  }
}

typedef _LogCallbackC = Void Function(
  UnsignedInt level,
  Pointer<Char> text,
  Pointer<Void> userData,
);
