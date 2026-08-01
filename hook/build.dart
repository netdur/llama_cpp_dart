import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

// pub.dev only accepts `build.dart` / `link.dart` under `hook/`, so the
// extraction helper lives in lib/ and is imported by package URI.
import 'package:llama_cpp_dart/src/hook/android_native_assets.dart';

const _bundledAarPath = 'native/android/llama-cpp-dart.aar';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets ||
        input.config.code.targetOS != OS.android) {
      return;
    }

    final bundleAndroid = input.userDefines['bundle_android'];
    if (bundleAndroid is! bool?) {
      throw const FormatException(
        'hooks.user_defines.llama_cpp_dart.bundle_android must be a boolean.',
      );
    }
    if (bundleAndroid == false) return;

    final architecture = input.config.code.targetArchitecture;
    final abi = androidAbiFor(architecture);
    if (abi == null) {
      stderr.writeln(
        'llama_cpp_dart: skipping unsupported Android architecture '
        '$architecture.',
      );
      return;
    }

    final customAar = input.userDefines.path('android_aar');
    final aar = customAar ?? input.packageRoot.resolve(_bundledAarPath);
    final aarFile = File.fromUri(aar);
    if (!await aarFile.exists()) {
      throw StateError(
        'llama_cpp_dart Android AAR not found at ${aarFile.path}. '
        'Published releases include it automatically. From a source checkout, '
        'run tool/stage_android_native_assets.sh first.',
      );
    }
    output.dependencies.add(aarFile.uri);

    final libraries = await extractAndroidLibraries(
      aar: aarFile.uri,
      abi: abi,
      outputDirectory: input.outputDirectoryShared,
    );
    if (libraries.isEmpty) {
      // The default artifact intentionally ships arm64 only. Leaving the other
      // Android variants empty lets Flutter build an arm64-only app bundle.
      stderr.writeln(
        'llama_cpp_dart: ${aarFile.path} has no $abi libraries; '
        'this target will not contain llama.cpp.',
      );
      return;
    }

    for (final library in libraries) {
      output.assets.code.add(
        CodeAsset(
          package: input.packageName,
          name: library.pathSegments.last,
          linkMode: DynamicLoadingBundled(),
          file: library,
        ),
      );
    }
  });
}
