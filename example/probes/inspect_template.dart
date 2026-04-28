/// Dump what `llama_chat_apply_template` renders for various templates.
///
///   LLAMA_CPP_DART_LIB=$(pwd)/build/macos/install/lib/libllama.dylib \
///     dart run example/probes/inspect_template.dart
library;

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  final libPath = Platform.environment['LLAMA_CPP_DART_LIB']!;
  LlamaLibrary.load(path: libPath);

  const messages = [
    ChatMessage.system('You are concise.'),
    ChatMessage.user('What is 2+2?'),
  ];

  for (final entry in {
    'gemma sentinel': KnownChatTemplates.gemma,
    'chatml sentinel': KnownChatTemplates.chatml,
    'llama3 sentinel': KnownChatTemplates.llama3,
    'gemma full': '<start_of_turn>{{ role }}\n{{ content }}<end_of_turn>',
  }.entries) {
    stdout.writeln('==== ${entry.key}: "${entry.value}"');
    try {
      final out = ChatTemplate.apply(
        template: entry.value,
        messages: messages,
        addAssistant: true,
      );
      stdout.writeln(out);
    } catch (e) {
      stdout.writeln('failed: $e');
    }
    stdout.writeln('====');
  }
}
