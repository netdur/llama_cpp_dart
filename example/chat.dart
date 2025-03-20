// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/chat.dart';

void main() async {
  try {
    final chatMLFormat = ChatMLFormat();
    final chatHistory = ChatHistory();

    print('Initializing chat...\n');

    // Initialize system prompt
    chatHistory.addMessage(
      role: Role.system,
      content: """
Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.

Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.
""",
    );

    print('Loading model...');
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    final llama = Llama(
      "/Users/adel/Downloads/gemma-3-12b-it-Q4_K_M.gguf",
      ModelParams(),
      ContextParams(),
      SamplerParams(),
    );
    print('Model loaded successfully\n');

    const questions = [
      "What was OKT3 originally sourced from?",
      "What was the company called?",
    ];

    for (var i = 0; i < questions.length; i++) {
      print('Q${i + 1}: ${questions[i]}');
      print('A${i + 1}: ', terminator: '');
      await processQuestion(
        llama: llama,
        chatHistory: chatHistory,
        chatMLFormat: chatMLFormat,
        question: questions[i],
      );
    }

    llama.dispose();
    print('Chat session ended.');
  } catch (e) {
    print("\nError occurred: $e");
  }
}

Future<void> processQuestion({
  required Llama llama,
  required ChatHistory chatHistory,
  required ChatMLFormat chatMLFormat,
  required String question,
}) async {
  chatHistory.addMessage(role: Role.user, content: question);
  llama.setPrompt(chatHistory.exportFormat(ChatFormat.chatml));

  final responseBuffer = StringBuffer();

  while (true) {
    final (token, done) = llama.getNext();
    final chunk = chatMLFormat.filterResponse(token);

    if (chunk != null) {
      stdout.write(token);
      responseBuffer.write(token);
    }
    if (done) break;
  }

  chatHistory.addMessage(
    role: Role.assistant,
    content: responseBuffer.toString().trim(),
  );
  stdout.writeln('\n');
}

// Helper function for print with optional terminator
void print(String message, {String terminator = '\n'}) {
  stdout.write(message + terminator);
}

const int megaByte = 1024 * 1024;
