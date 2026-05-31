/// De-risk probe: can we resolve the C++-mangled pre-norm hidden-state
/// staging symbols through Dart FFI and get sane vectors back?
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///   LLAMA_CPP_DART_MODEL=/path/to/model.gguf \
///     dart run tool/probe_prenorm.dart
library;

import 'dart:ffi';
import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

// Itanium-mangled names (clang/gcc). On macOS `nm` shows a leading '_' which
// dlsym strips, so the lookup name starts at '_Z'.
const _setName = '_Z29llama_set_embeddings_pre_normP13llama_contextbb';
const _getIthName = '_Z33llama_get_embeddings_pre_norm_ithP13llama_contexti';

typedef _SetNative = Void Function(Pointer<Void>, Bool, Bool);
typedef _SetDart = void Function(Pointer<Void>, bool, bool);
typedef _GetIthNative = Pointer<Float> Function(Pointer<Void>, Int32);
typedef _GetIthDart = Pointer<Float> Function(Pointer<Void>, int);

void main() {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB'];
  final modelPath = Platform.environment['LLAMA_CPP_DART_MODEL'];
  if (libPath == null || modelPath == null) {
    stderr.writeln('set LLAMA_CPP_DART_LIB and LLAMA_CPP_DART_MODEL');
    exit(2);
  }

  LlamaLibrary.load(path: libPath);
  LlamaLog.silence();

  final dylib = DynamicLibrary.open(libPath);
  final setPreNorm = dylib.lookupFunction<_SetNative, _SetDart>(_setName);
  final getIth = dylib.lookupFunction<_GetIthNative, _GetIthDart>(_getIthName);
  stdout.writeln('resolved set_embeddings_pre_norm + get_..._ith ✓');

  final model = LlamaModel.load(ModelParams(path: modelPath, gpuLayers: 99));
  final ctx = LlamaContext.create(
    model,
    const ContextParams(nCtx: 512, nBatch: 512, nUbatch: 512),
  );
  final nEmbd = model.nEmbd;
  stdout.writeln('n_embd = $nEmbd');

  // Turn on pre-norm embedding output for ALL tokens (masked=false).
  setPreNorm(ctx.pointer.cast(), true, false);

  final encoded = Tokenizer(model.vocab)
      .encode('The ocean is', addSpecial: true, parseSpecial: true);
  stdout.writeln('prompt tokens = $encoded');

  final batch = LlamaBatch(encoded.length);
  for (var i = 0; i < encoded.length; i++) {
    batch.add(encoded[i], i, [0], wantLogits: true);
  }
  final rc = LlamaLibrary.bindings.llama_decode(ctx.pointer, batch.raw);
  stdout.writeln('decode rc = $rc');

  if (rc == 0) {
    final h = getIth(ctx.pointer.cast(), encoded.length - 1);
    if (h == nullptr) {
      stdout.writeln('get_..._ith returned NULL ✗');
    } else {
      final view = h.asTypedList(nEmbd);
      var allZero = true, bad = false, norm = 0.0;
      for (final v in view) {
        if (v != 0.0) allZero = false;
        if (v.isNaN || v.isInfinite) bad = true;
        norm += v * v;
      }
      stdout.writeln('hidden[0..7] = ${view.sublist(0, 8)}');
      stdout.writeln('allZero=$allZero  anyNaN/Inf=$bad  L2norm=$norm');
      stdout.writeln(allZero || bad
          ? 'RESULT: suspicious hidden state ✗'
          : 'RESULT: sane non-zero finite hidden state ✓');
    }
  }

  batch.dispose();
  ctx.dispose();
  model.dispose();
  LlamaLibrary.dispose();
}
