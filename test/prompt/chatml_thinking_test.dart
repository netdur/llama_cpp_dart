import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_dart/src/prompt/chatml_thinking_format.dart';

void main() {
  group('ChatMLThinkingFormat', () {
    late ChatMLThinkingFormat format;

    setUp(() {
      format = ChatMLThinkingFormat();
    });

    test('formatPrompt formats single user message correctly', () {
      final prompt = 'Hello, how are you?';
      final expected =
          '<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n<|thought|>';

      expect(format.formatPrompt(prompt), equals(expected));
    });

    test('formatMessages formats conversation history correctly', () {
      final messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is 2+2?'},
        {'role': 'assistant', 'content': '<|thought|>Thinking...\nIt is 4.'},
        {'role': 'user', 'content': 'Thanks!'}
      ];

      // Note: Assistant history uses standard header, not the <|thought|> one.
      final expected =
          '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
          '<|im_start|>user\nWhat is 2+2?<|im_end|>\n'
          '<|im_start|>assistant\n<|thought|>Thinking...\nIt is 4.<|im_end|>\n'
          '<|im_start|>user\nThanks!<|im_end|>\n'
          '<|im_start|>assistant\n<|thought|>';

      expect(format.formatMessages(messages), equals(expected));
    });

    test(
        'formatMessages properly ends with thinking tag if last message is user',
        () {
      final messages = [
        {'role': 'user', 'content': 'Hello'}
      ];
      final expected =
          '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<|thought|>';
      expect(format.formatMessages(messages), equals(expected));
    });
  });
}
