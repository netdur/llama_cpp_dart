import 'dart:io';

import 'package:archive/archive.dart';
import 'package:code_assets/code_assets.dart';

/// The ABI directory used by Android AARs for [architecture].
String? androidAbiFor(Architecture architecture) => switch (architecture) {
  Architecture.arm => 'armeabi-v7a',
  Architecture.arm64 => 'arm64-v8a',
  Architecture.ia32 => 'x86',
  Architecture.x64 => 'x86_64',
  _ => null,
};

/// Extracts all shared libraries for [abi] from [aar] into [outputDirectory].
///
/// Only direct children of `jni/<abi>/` are accepted, so archive paths cannot
/// escape the hook output directory. The returned URIs are ordered by file
/// name for reproducible hook output.
Future<List<Uri>> extractAndroidLibraries({
  required Uri aar,
  required String abi,
  required Uri outputDirectory,
}) async {
  final aarFile = File.fromUri(aar);
  final archive = ZipDecoder().decodeBytes(await aarFile.readAsBytes());
  final prefix = 'jni/$abi/';
  final entries = archive.where((entry) {
    if (!entry.isFile || !entry.name.startsWith(prefix)) return false;
    final relativeName = entry.name.substring(prefix.length);
    return relativeName.endsWith('.so') && !relativeName.contains('/');
  }).toList()..sort((left, right) => left.name.compareTo(right.name));

  if (entries.isEmpty) return const <Uri>[];
  if (!entries.any((entry) => entry.name == '${prefix}libllama.so')) {
    throw FormatException(
      'The Android AAR contains libraries for $abi, but not libllama.so.',
    );
  }

  final extracted = <Uri>[];
  for (final entry in entries) {
    final fileName = entry.name.substring(prefix.length);
    final output = outputDirectory.resolve('$abi/$fileName');
    final bytes = entry.readBytes();
    if (bytes == null) {
      throw FormatException(
        'Could not read ${entry.name} from the Android AAR.',
      );
    }
    final outputFile = File.fromUri(output);
    await outputFile.create(recursive: true);
    await outputFile.writeAsBytes(bytes, flush: true);
    extracted.add(outputFile.uri);
  }
  return extracted;
}
