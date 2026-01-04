import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  test('library imports correctly', () {
    // Just verify we can import the library without errors.
    expect(LlamaSplitMode.none, isNotNull);
  });
}
