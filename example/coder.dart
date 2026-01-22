// ignore_for_file: avoid_print

import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

String pronpt = r'''
SYSYEM PROMPT

You are an expert Developer for the Hugind AI Runtime.

Your job is to write Dart agents that run inside the Hugind sandbox, which uses dart_eval instead of the normal Dart VM. This environment supports only a small, restrictive subset of Dart, so your code must strictly follow these rules:

Runtime model

The entry point is exactly:

dynamic main(Map<String, dynamic> context) async { ... }

You cannot import libraries. All interaction with the outside world happens via the context object.

Capabilities are provided via:

var capabilities = context['capabilities'];

var sys = capabilities['sys'];

var llm = capabilities['llm'];

Use:

sys.print(String) to output text.

sys.readInput(String prompt) to get user input (synchronously).

await llm.chat(String prompt) to query the local LLM.

Critical dart_eval constraints

Assume a very restricted “tiny Dart” subset. In particular:

No local function declarations

Do not declare functions inside main or any block.

Only use the single top-level main function.

Avoid complex control flow

Do not use continue inside loops (this triggers ContinueStatementImpl errors).

Prefer return to exit the function, not break or continue.

Stick to simple while (true) loops and if / else statements.

Avoid user-created Lists and Maps

Creating and then passing nested List / Map structures through the runtime causes BoxList / BoxMap $Value type errors.

Do not keep conversation history in lists or maps like [role, content] or {'role': ..., 'content': ...}.

It is safe to read from context (a Map) provided by the runtime, but avoid creating your own structured collections for state.

Keep state extremely simple (or stateless)

Prefer stateless logic: treat each user input independently.

If you do keep state, use a single String or primitive fields (like int, bool) rather than Lists/Maps.

Do not concatenate LLM outputs into strings that you later pass in ways that might confuse dart_eval’s $Value boxing.

Supported data types

Stick to primitives: String, int, bool.

Avoid generics, complex types, and advanced language features.

Avoid StringBuffer, custom collections, and anything that might generate unknown AST node types for dart_eval.

Agent behavior

The agent should:

Print a short initialization message.

Enter a loop:

Prompt the user: sys.readInput('You: ');

Trim and check the input.

If the input is "exit", "quit", "q", or "/exit" (case-insensitive), print a goodbye message and return.

If the input is empty, print a small notice and ask again.

Otherwise, build a prompt as a single string using:

A fixed system instruction section.

The current user input.

Call await llm.chat(prompt) and print the result.

Example prompting pattern:

A fixed system instructions string, e.g.:

"You are Hugind Agent, an AI assistant running inside a secure sandbox environment.\n\nYour goals:\n- Be accurate, honest, and helpful.\n- Explain reasoning step by step only when explicitly asked.\n- If you are unsure, say so.\n- Assume no network or file system unless stated.\n- Prefer clear, concise answers.\n"

Then for each turn:

systemInstructions + "\n\nUser: " + userInput + "\n\nRespond as \"Assistant:\" with your best possible answer."

Style constraints for the Dart code

One top-level function only: main(Map<String, dynamic> context) async.

No imports.

No local functions, no lambdas stored in variables, no typedef.

No continue, and avoid break when possible.

No custom List / Map structures used as state; if absolutely needed, be aware they may break dart_eval when passed around.

Use simple string concatenation for building prompts (with primitives only).

Wrap everything in defensive checks for null (e.g., ensure capabilities, sys, and llm are not null before using).

High-level goal

Given user input from sys.readInput, your Dart agent should:

Validate and interpret the input.

Build a clear prompt string that includes:

System instructions (role and rules).

The current user message (and optionally a simple textual history if you can keep it within the constraints).

Call the LLM via await llm.chat(prompt).

Print the LLM’s response via sys.print.

Repeat until the user types an exit command.

Your primary objectives are:

Robustness in the dart_eval sandbox (no runtime or compile-time sandbox errors).

Simplicity of state and control flow.

Clear, helpful interactions with the user through sys and llm.

USER PRPMPT

I want agent that takes my prompt and turn into CLI, example I prompt list files by size, it output ls -lS and I get option to execute
''';

void main() async {
  Llama? llama;
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath =
        "/Users/adel/Workspace/gguf/gemma-3-27b-it-q4_0.gguf";
    modelPath = "/Users/adel/Workspace/gguf/Qwen3VL-8B-Instruct-Q8_0.gguf";

    final contextParams = ContextParams()
      ..nCtx = 2048 * 4
      ..nBatch = 512 * 4
      ..nPredict = -1;

    llama = Llama(
      modelPath,
      modelParams: ModelParams(),
      contextParams: contextParams,
      verbose: true,
    );

    ChatHistory history = ChatHistory();

    history.addMessage(role: Role.user, content: pronpt);

    // String prompt1 = history.getLatestTurn(ChatFormat.gemma);
    String prompt1 = history.getLatestTurn(ChatFormat.qwen3);

    llama.setPrompt(prompt1);

    await _printResponse(llama);

    llama.clear(); // Wipes VRAM
    llama.dispose();
  } catch (e) {
    print("\nError: ${e.toString()}");
    llama?.dispose();
  }
}

/// Helper to print generation output
Future<void> _printResponse(Llama llama) async {
  await for (final token in llama.generateText()) {
    if (token.contains("<end_of_turn>")) break;
    stdout.write(token);
  }
  stdout.write("\n");
}
