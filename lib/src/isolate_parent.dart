import 'dart:async';
import 'dart:isolate';

import 'package:typed_isolate/typed_isolate.dart';

import "llama.dart";
import 'alpaca_format.dart';
import 'chatml_format.dart';
import 'prompt_format.dart';

import "isolate_child.dart";
import "isolate_types.dart";

extension on PromptFormatType {
  PromptFormat? get formatter => switch (this) {
    PromptFormatType.raw => null,
    PromptFormatType.alpaca => AlpacaFormat(),
    PromptFormatType.chatml => ChatMLFormat(),
  };
}

class LlamaParent extends IsolateParent<LlamaCommand, LlamaResponse> {
  final _controller = StreamController<String>.broadcast();

  // TODO: What type is this supposed to be?
  List<Map<String, dynamic>> messages = [];

  final LlamaLoad loadCommand;
  final PromptFormat? formatter;
  LlamaParent(this.loadCommand) :
    formatter = loadCommand.modelParams.format.formatter;

  Stream<String> get stream => _controller.stream;

  @override
  void onData(LlamaResponse data, Object id) {
    if (data.isDone) _controller.close();
    _parseResponse(data.text);
  }

  void _parseResponse(String response) {
    final processed = formatter == null ? response : formatter!.filterResponse(response);
    if (processed != null) _controller.add(processed);
  }

  @override
  void init() async {
    super.init();
    await spawn(LlamaChild());
    send(data: LlamaInit(Llama.libraryPath), id: 1);
    send(data: loadCommand, id: 1);
  }

  void sendPrompt(String prompt) async {
    final formattedPrompt = messages.isEmpty
      ? formatter?.formatPrompt(prompt) ?? prompt
      : formatter?.formatMessages(messages) ?? prompt;
    send(id: 1, data: LlamaPrompt(formattedPrompt));
  }

  void stop() async {
    send(id: 1, data: LlamaStop());
  }

  @override
  Future<void> dispose([int priority = Isolate.beforeNextEvent]) async {
    send(id: 1, data: LlamaClear());
    _controller.close();
    super.dispose(priority);
  }
}
