// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/chat.dart';
import 'package:llama_cpp_dart/src/chatml_format.dart';

void main() async {
  try {
    final chatMLFormat = ChatMLFormat();
    final chatHistory = ChatHistory();

    chatHistory.addMessage(
      role: Role.system,
      content:
          """You are playing the role of a friendly village shopkeeper in a medieval fantasy setting. Your shop sells basic supplies and equipment. You are an elderly person who has run this shop for decades and knows everyone in the village. You speak in a warm, casual manner with a slight rural accent""",
    );
    chatHistory.addMessage(
        role: Role.user,
        content:
            "pushes open the creaky door, snow falling from my cloak Hello? Is anyone here?");
    chatHistory.addMessage(
        role: Role.assistant,
        content:
            "Looks up from sweeping the wooden floor, adjusting my wire-rimmed spectacles Ah, welcome, welcome! Don't get many strangers round these parts, 'specially not in weather like this. Sets the broom against the wall What can I help you with, traveler?");
    chatHistory.addMessage(
        role: Role.user,
        content:
            "shivers and pulls my cloak tighter I need a lantern - mine broke on the road here. Do you have any for sale?");
    chatHistory.addMessage(
        role: Role.assistant,
        content:
            "Shuffles behind the counter with a knowing smile Aye, got just what you need. Reaches up to a dusty shelf Got two types here - sturdy iron ones for three silver, or these fancy brass ones for five. Holds both up The brass burns a bit brighter, but the iron'll take more of a beating if you're doing rough traveling.");
    chatHistory.addMessage(
        role: Role.user,
        content:
            "reaches for the iron lantern, coins jingling I'll take the iron one - seems more practical for the road ahead. pauses Say, you wouldn't happen to know of any safe places to rest for the night?");

    print('Loading model...');
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    final llama = Llama(
      "/Users/adel/Downloads/Qwen2-7B-Multilingual-RP-Q4_K_M.gguf",
      ModelParams(),
      ContextParams(),
      SamplerParams(),
    );

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

    llama.dispose();
  } catch (e) {
    print("\nError occurred: $e");
  }
}

void print(String message, {String terminator = '\n'}) {
  stdout.write(message + terminator);
}

const int megaByte = 1024 * 1024;
