import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import 'library_loader.dart';

/// Control over llama.cpp / ggml log output.
///
/// llama.cpp logs unconditionally to stderr by default. [silence] mutes the
/// log callback; [useDefault] restores it; [captureToFile] redirects the
/// process-wide stderr `FILE*` to a file so the host app can read llama.cpp /
/// ggml log lines (which otherwise vanish on Android, where stderr is not
/// connected to logcat).
///
/// Custom Dart-side log callbacks are intentionally not exposed: the
/// underlying logger fires from arbitrary threads, which is unsafe for
/// `Pointer.fromFunction` callbacks that touch the Dart heap. File capture
/// is the safe, portable substitute — apps can tail the file and forward
/// lines to logcat / `debugPrint`.
final class LlamaLog {
  LlamaLog._();

  static Pointer<NativeFunction<_LogCallbackC>>? _silentCallback;
  static String? _captureFilePath;

  /// Currently active stderr-capture file path, or `null` if not capturing.
  static String? get captureFilePath => _captureFilePath;

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

  /// Redirect the process-wide stderr `FILE*` to [path] so llama.cpp / ggml
  /// log lines (which always go to stderr) become readable to the host app.
  /// Idempotent: a second call replaces the active capture file. Call
  /// [restoreStderr] to revert.
  ///
  /// On Android the host app should tail this file (e.g. via a periodic
  /// read loop) and forward lines to `debugPrint` / logcat — neither
  /// `print` nor `stderr.writeln` reach logcat by default.
  ///
  /// [append] mode keeps prior content; otherwise the file is truncated.
  ///
  /// Implicitly calls [useDefault] so log output reaches stderr.
  static void captureToFile(String path, {bool append = false}) {
    useDefault();

    final libc = _libc;
    final stderrFp = _stderrFilePtr(libc);
    if (stderrFp == null || stderrFp == nullptr) {
      throw const LlamaLogException(
          'could not resolve stderr FILE* for capture');
    }

    final pathPtr = path.toNativeUtf8();
    final modePtr = (append ? 'a' : 'w').toNativeUtf8();
    try {
      final freopen = libc.lookupFunction<
          Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Void>),
          Pointer<Void> Function(
              Pointer<Utf8>, Pointer<Utf8>, Pointer<Void>)>('freopen');
      final result = freopen(pathPtr, modePtr, stderrFp);
      if (result == nullptr) {
        throw LlamaLogException('freopen($path) failed for stderr capture');
      }
      _captureFilePath = path;

      // _IONBF=2: make stderr unbuffered so log lines are visible immediately.
      libc.lookupFunction<
          Int32 Function(Pointer<Void>, Pointer<Char>, Int32, IntPtr),
          int Function(Pointer<Void>, Pointer<Char>, int, int)>('setvbuf')(
        stderrFp,
        nullptr,
        2,
        0,
      );
    } finally {
      malloc.free(pathPtr);
      malloc.free(modePtr);
    }
  }

  /// Undo a prior [captureToFile]. On POSIX this re-opens stderr against
  /// `/dev/null` (the original tty/pipe is not recoverable across `freopen`);
  /// the host app's own `stderr.writeln` continues to work via Dart's own
  /// fd 2, which is independent of the C `FILE*` slot.
  ///
  /// Idempotent — no-op if capture was never installed.
  static void restoreStderr() {
    if (_captureFilePath == null) return;
    final libc = _libc;
    final stderrFp = _stderrFilePtr(libc);
    if (stderrFp == null || stderrFp == nullptr) {
      _captureFilePath = null;
      return;
    }
    final devNull = '/dev/null'.toNativeUtf8();
    final mode = 'w'.toNativeUtf8();
    try {
      final freopen = libc.lookupFunction<
          Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Void>),
          Pointer<Void> Function(
              Pointer<Utf8>, Pointer<Utf8>, Pointer<Void>)>('freopen');
      freopen(devNull, mode, stderrFp);
    } finally {
      malloc.free(devNull);
      malloc.free(mode);
    }
    _captureFilePath = null;
  }

  static void _noop(int level, Pointer<Char> text, Pointer<Void> userData) {
    // Intentionally empty. Must not touch any Dart objects: this may be
    // invoked from a non-Dart worker thread where dereferencing the Dart
    // runtime is unsafe.
  }

  static DynamicLibrary get _libc {
    if (Platform.isWindows) {
      throw const LlamaLogException(
          'stderr capture is not supported on Windows');
    }
    if (Platform.isAndroid) return DynamicLibrary.open('libc.so');
    return DynamicLibrary.process();
  }

  static Pointer<Void>? _stderrFilePtr(DynamicLibrary libc) {
    final symbol =
        (Platform.isMacOS || Platform.isIOS) ? '__stderrp' : 'stderr';
    try {
      return libc.lookup<Pointer<Void>>(symbol).cast<Pointer<Void>>().value;
    } on ArgumentError {
      return null;
    }
  }
}

class LlamaLogException implements Exception {
  final String message;
  const LlamaLogException(this.message);
  @override
  String toString() => 'LlamaLogException: $message';
}

typedef _LogCallbackC = Void Function(
  UnsignedInt level,
  Pointer<Char> text,
  Pointer<Void> userData,
);
