import 'dart:async';
import 'dart:isolate';

import 'package:typed_isolate/typed_isolate.dart';

import "llama.dart";
import 'alpaca_format.dart';
import 'chatml_format.dart';
import 'prompt_format.dart';

import "isolate_child.dart";
import "isolate_types.dart";

class LlamaParent extends IsolateParent<LlamaCommand, LlamaResponse> {
  final _controller = StreamController<String>.broadcast();

  List<Map<String, dynamic>> messages = [];

  final LlamaLoad loadCommand;
  LlamaParent({
    required this.loadCommand,
  });

  Stream<String> get stream => _controller.stream;

  Isolate? _modelIsolate;

  @override
  void onData(LlamaResponse data, Object id) {
    if (data.isDone) _controller.close();
    _parseResponse(data.text);
  }

  void _parseResponse(String response) {
    final processed = switch (loadCommand.modelParams.format) {
      PromptFormatType.raw => response,
      PromptFormatType.alpaca => AlpacaFormat().filterResponse(response),
      PromptFormatType.chatml => ChatMLFormat().filterResponse(response),
    };
    if (processed != null) _controller.add(processed);
  }

  @override
  void init() async {
    super.init();
    _modelIsolate = await spawn(LlamaChild());
    send(data: LlamaInit(Llama.libraryPath), id: 1);
    send(data: loadCommand, id: 1);
  }

  void sendPrompt(String prompt) async {
    final formattedPrompt = messages.isEmpty
      ? switch (loadCommand.modelParams.format) {
        PromptFormatType.raw => prompt,
        PromptFormatType.alpaca => AlpacaFormat().formatPrompt(prompt),
        PromptFormatType.chatml => ChatMLFormat().formatPrompt(prompt),
      } : switch (loadCommand.modelParams.format) {
        PromptFormatType.raw => prompt,
        PromptFormatType.alpaca => AlpacaFormat().formatMessages(messages),
        PromptFormatType.chatml => ChatMLFormat().formatMessages(messages),
      };
    send(id: 1, data: LlamaPrompt(formattedPrompt));
  }

  void stop() async {
    send(id: 1, data: LlamaStop());
  }

  @override
  Future<void> dispose([int priority = Isolate.beforeNextEvent]) async {
    send(id: 1, data: LlamaClear());
    _modelIsolate?.kill(priority: Isolate.immediate);
    _controller.close();
    super.dispose(priority);
  }
}