import 'dart:io';

import 'package:archive/archive.dart';
import 'package:code_assets/code_assets.dart';
import 'package:test/test.dart';

import '../hook/android_native_assets.dart';

void main() {
  test('maps Dart architectures to Android AAR ABIs', () {
    expect(androidAbiFor(Architecture.arm), 'armeabi-v7a');
    expect(androidAbiFor(Architecture.arm64), 'arm64-v8a');
    expect(androidAbiFor(Architecture.ia32), 'x86');
    expect(androidAbiFor(Architecture.x64), 'x86_64');
    expect(androidAbiFor(Architecture.riscv64), isNull);
  });

  test('extracts only direct shared libraries for the requested ABI', () async {
    final temp = await Directory.systemTemp.createTemp('llama-aar-test-');
    addTearDown(() => temp.delete(recursive: true));
    final aar = File('${temp.path}/llama.aar');
    final archive = Archive()
      ..add(ArchiveFile.bytes('jni/arm64-v8a/libllama.so', <int>[1, 2]))
      ..add(ArchiveFile.bytes('jni/arm64-v8a/libggml.so', <int>[3]))
      ..add(ArchiveFile.bytes('jni/arm64-v8a/nested/libbad.so', <int>[4]))
      ..add(ArchiveFile.bytes('jni/x86_64/libllama.so', <int>[5]));
    await aar.writeAsBytes(ZipEncoder().encodeBytes(archive));

    final output = Directory('${temp.path}/output')..createSync();
    final libraries = await extractAndroidLibraries(
      aar: aar.uri,
      abi: 'arm64-v8a',
      outputDirectory: output.uri,
    );

    expect(libraries.map((uri) => uri.pathSegments.last), <String>[
      'libggml.so',
      'libllama.so',
    ]);
    expect(await File.fromUri(libraries.last).readAsBytes(), <int>[1, 2]);
  });

  test('rejects an ABI without libllama', () async {
    final temp = await Directory.systemTemp.createTemp('llama-aar-test-');
    addTearDown(() => temp.delete(recursive: true));
    final aar = File('${temp.path}/llama.aar');
    final archive = Archive()
      ..add(ArchiveFile.bytes('jni/arm64-v8a/libggml.so', <int>[1]));
    await aar.writeAsBytes(ZipEncoder().encodeBytes(archive));

    expect(
      () => extractAndroidLibraries(
        aar: aar.uri,
        abi: 'arm64-v8a',
        outputDirectory: Directory('${temp.path}/output').uri,
      ),
      throwsFormatException,
    );
  });
}
