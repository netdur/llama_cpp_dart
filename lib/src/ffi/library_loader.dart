import 'dart:ffi';
import 'dart:io' show File, Platform;

import 'package:ffi/ffi.dart';

import '../types/exceptions.dart';
import 'bindings.dart';

/// Process-wide owner of the loaded llama.cpp dynamic library.
///
/// Call [load] exactly once before using any other API. The path must point
/// at the platform-specific shared library produced by `tool/build_native.sh`
/// or downloaded from a release bundle.
final class LlamaLibrary {
  LlamaLibrary._();

  static LlamaBindings? _bindings;
  static String? _libraryPath;
  static String? _libraryDir;
  static DynamicLibrary? _llamaLib;
  static DynamicLibrary? _mtmdLib;

  /// Lazily-resolved bindings. Throws if [load] has not been called.
  static LlamaBindings get bindings {
    final b = _bindings;
    if (b == null) {
      throw const LlamaLibraryException(
        'LlamaLibrary.load() must be called before using the binding.',
      );
    }
    return b;
  }

  static bool get isLoaded => _bindings != null;
  static String? get libraryPath => _libraryPath;
  static String? get libraryDirectory => _libraryDir;

  /// Load the native library and initialize llama / ggml backends.
  ///
  /// [path] is the absolute path to `libllama.dylib` / `libllama.so`.
  /// [backendDirectory] points at the directory holding the ggml backend
  /// dylibs (`libggml-metal.dylib`, `libggml-cpu.dylib`, ...). Defaults to
  /// the directory containing [path].
  ///
  /// Idempotent: subsequent calls with the same [path] are no-ops.
  /// Calling with a different path while a library is already loaded
  /// throws [LlamaLibraryException].
  static void load({required String path, String? backendDirectory}) {
    if (_bindings != null) {
      if (_libraryPath == path) return;
      throw LlamaLibraryException(
        'A different llama library is already loaded: $_libraryPath',
      );
    }

    final DynamicLibrary llama;
    try {
      llama = DynamicLibrary.open(path);
    } on ArgumentError catch (e) {
      throw LlamaLibraryException('failed to open $path: $e');
    }
    _llamaLib = llama;

    // Try to open libmtmd alongside libllama. mtmd is optional — engines
    // without multimodal can run without it. Symbols starting with
    // `mtmd_` are routed to this handle when the lookup runs.
    //
    // Resolution rules:
    //   1. If [path] is a basename (no slash), assume the OS will resolve
    //      libraries by name from the app's lib path (Android jniLibs,
    //      Linux LD_LIBRARY_PATH). Open libmtmd by basename too.
    //   2. Otherwise look in the same directory as libllama (or the
    //      explicit [backendDirectory] override) and open by full path.
    DynamicLibrary? mtmd;
    final dir = backendDirectory ?? _dirOf(path);
    final mtmdName = _mtmdFileName();
    if (mtmdName != null) {
      if (dir == null) {
        // Basename mode (typical Android usage).
        try {
          mtmd = DynamicLibrary.open(mtmdName);
        } on ArgumentError {
          // mtmd not present in this build; multimodal calls will throw.
        }
      } else {
        final mtmdPath = '$dir${Platform.pathSeparator}$mtmdName';
        if (File(mtmdPath).existsSync()) {
          try {
            mtmd = DynamicLibrary.open(mtmdPath);
          } on ArgumentError {
            // Skip silently; mtmd is best-effort.
          }
        }
      }
    }
    _mtmdLib = mtmd;

    final bindings = LlamaBindings.fromLookup(_routedLookup);
    bindings.llama_backend_init();

    if (dir != null) {
      final dirPtr = dir.toNativeUtf8(allocator: calloc);
      try {
        bindings.ggml_backend_load_all_from_path(dirPtr.cast());
      } finally {
        calloc.free(dirPtr);
      }
    } else {
      bindings.ggml_backend_load_all();
    }

    _bindings = bindings;
    _libraryPath = path;
    _libraryDir = dir;
  }

  /// True when the optional libmtmd dylib was successfully opened
  /// alongside libllama. False means multimodal calls will fail at lookup.
  static bool get isMultimodalAvailable => _mtmdLib != null;

  static Pointer<T> _routedLookup<T extends NativeType>(String name) {
    if (name.startsWith('mtmd_')) {
      final m = _mtmdLib;
      if (m == null) {
        throw LlamaLibraryException(
          'mtmd symbol "$name" requested but libmtmd is not loaded. '
          'Build with --with-mtmd and ensure libmtmd is shipped beside libllama.',
        );
      }
      return m.lookup<T>(name);
    }
    return _llamaLib!.lookup<T>(name);
  }

  static String? _mtmdFileName() {
    if (Platform.isMacOS) return 'libmtmd.dylib';
    if (Platform.isLinux || Platform.isAndroid) return 'libmtmd.so';
    if (Platform.isWindows) return 'mtmd.dll';
    return null;
  }

  /// Resolve symbols from the running process instead of opening a
  /// separate dylib.
  ///
  /// Use this when the llama.cpp static archive (e.g. an iOS xcframework
  /// slice or a macOS app that links `llama.framework` directly) was
  /// linked into the app binary by Xcode. There's no `.dylib` to dlopen —
  /// the symbols are already part of the process. Equivalent to
  /// `LlamaLibrary.load(path: '...')` but without a path.
  ///
  /// Idempotent. Mixing this with [load] in the same isolate throws.
  static void loadFromProcess() {
    if (_bindings != null) {
      if (_libraryPath == r'<process>') return;
      throw LlamaLibraryException(
        'A different llama library is already loaded: $_libraryPath',
      );
    }

    final lib = DynamicLibrary.process();
    _llamaLib = lib;
    _mtmdLib = lib; // same process namespace; mtmd is statically linked too

    final bindings = LlamaBindings.fromLookup(_routedLookup);
    bindings.llama_backend_init();
    bindings.ggml_backend_load_all();

    _bindings = bindings;
    _libraryPath = r'<process>';
    _libraryDir = null;
  }

  /// Drop the cached bindings reference for this isolate.
  ///
  /// Does NOT call `llama_backend_free` — backend init is process-global
  /// and freeing it from one isolate while another still holds models
  /// crashes the second isolate's `llama_free`. Process exit reclaims the
  /// backend state.
  static void dispose() {
    if (_bindings == null) return;
    _bindings = null;
    _libraryPath = null;
    _libraryDir = null;
    _llamaLib = null;
    _mtmdLib = null;
  }

  /// Default file name for the llama library on the current platform.
  /// Useful when writing test helpers; not used by [load].
  static String defaultFileName() {
    if (Platform.isMacOS) return 'libllama.dylib';
    if (Platform.isIOS) return 'llama.framework/llama';
    if (Platform.isAndroid || Platform.isLinux) return 'libllama.so';
    if (Platform.isWindows) return 'llama.dll';
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }

  static String? _dirOf(String path) {
    final i = path.lastIndexOf(Platform.pathSeparator);
    if (i < 0) return null;
    return path.substring(0, i);
  }
}
