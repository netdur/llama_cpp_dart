// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

// ==========================================================
// TOOLS
// ==========================================================

/// Tool to get the current time.
String getCurrentTime() {
  final now = DateTime.now();
  final formattedTime = DateFormat('HH:mm').format(now);
  return "The current time is $formattedTime.";
}

/// Tool to get the weather for a city.
Future<String> getWeather(String city) async {
  try {
    final url = Uri.parse('https://wttr.in/$city?format=%C,+%t');
    final response = await http.get(url);
    if (response.statusCode == 200) {
      final body = response.body.trim();
      if (body.contains(',')) {
        return "The weather in $city is ${body.split(',')[0]} with a temperature of ${body.split(',')[1]}.";
      }
      return "Could not parse weather for $city. Response: $body";
    } else {
      return "Sorry, I couldn't get the weather for $city.";
    }
  } catch (e) {
    return "An error occurred while fetching weather: $e";
  }
}

final Map<String, Function> toolBox = {
  'getCurrentTime': getCurrentTime,
  'getWeather': getWeather,
};

// ==========================================================
// PROMPT ENGINEERING
// ==========================================================

String buildToolDescriptions() {
  return """
- `getCurrentTime()`: Use this tool to get the current time. It takes no arguments.
- `getWeather(city: string)`: Use this tool to get the weather for a specific city.
""";
}

String buildSystemPrompt() {
  return """
You are an expert at choosing the right function to call to answer a user's question. You have access to the following tools:
${buildToolDescriptions()}

Your goal is to respond with ONLY a JSON object that represents the function call(s) needed.
- The JSON should be an array of objects.
- Each object must have a "tool_name" and an "arguments" map.
- If no tool is needed, or if you don't have enough information (e.g., the user asks for weather but doesn't name a city), respond with an empty JSON array: `[]`.
- Do NOT add any other text, explanation, or conversation.

Example:
User: What's the weather like in Paris?
Your response:
[
  {"tool_name": "getWeather", "arguments": {"city": "Paris"}}
]

User: what time is it?
Your response:
[
  {"tool_name": "getCurrentTime", "arguments": {}}
]

User: Tell me a joke.
Your response:
[]
""";
}

// ==========================================================
// MAIN AGENT LOGIC
// ==========================================================

Future<void> main() async {
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/Qwen3VL-8B-Instruct-Q8_0.gguf";

    final modelParams = ModelParams()..nGpuLayers = -1;
    final contextParams = ContextParams()
      ..nPredict = -1
      ..nCtx = 4096;
    final samplerParams = SamplerParams()..temp = 0.1;

    final llama = Llama(
      modelPath,
      modelParams: modelParams,
      contextParams: contextParams,
      samplerParams: samplerParams,
      verbose: false,
    );

    while (true) {
      stdout.write("\nAsk me something (or type 'exit'): ");
      final userInput = stdin.readLineSync();

      if (userInput == null || userInput.toLowerCase() == 'exit') {
        break;
      }

      print("🧠 Thinking...");

      final history = ChatHistory()
        ..addMessage(role: Role.user, content: buildSystemPrompt())
        ..addMessage(role: Role.user, content: userInput)
        ..addMessage(role: Role.assistant, content: '');

      llama.setPrompt(history.exportFormat(ChatFormat.gemini,
          leaveLastAssistantOpen: true));

      final responseBuffer = StringBuffer();
      await for (final token in llama.generateText()) {
        responseBuffer.write(token);
      }
      final llmResponse = responseBuffer.toString().trim();

      print("✅ LLM's Plan (raw response): $llmResponse");

      try {
        final jsonRegex =
            RegExp(r'```json\s*(\[.*\])\s*```|(\[.*\])', dotAll: true);
        final match = jsonRegex.firstMatch(llmResponse);

        if (match == null) {
          print(
              "🤖 I couldn't decide on a tool to use. Here's my raw thought: $llmResponse");
          continue;
        }

        final jsonString = match.group(1) ?? match.group(2)!;
        final List<dynamic> toolCalls = jsonDecode(jsonString);

        if (toolCalls.isEmpty) {
          print(
              "🤖 I don't have a tool for that. Please ask me about the time or weather.");
          continue;
        }

        for (var call in toolCalls) {
          final toolName = call['tool_name'];
          final arguments = call['arguments'] as Map<String, dynamic>;
          final tool = toolBox[toolName];

          if (tool != null) {
            print("🛠️  Executing tool: $toolName with args: $arguments");
            if (toolName == 'getWeather') {
              final result = await getWeather(arguments['city']);
              print("✔️  Result: $result");
            } else if (toolName == 'getCurrentTime') {
              final result = getCurrentTime();
              print("✔️  Result: $result");
            }
          } else {
            print("❌ Error: Tool '$toolName' not found in my toolbox.");
          }
        }
      } catch (e) {
        print("❌ Error processing the LLM's plan: $e");
        print("   My plan was: $llmResponse");
      }
    }

    llama.dispose();
    print("Goodbye!");
  } catch (e) {
    print("\nFatal Error: ${e.toString()}");
  }
}
