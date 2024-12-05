import 'dart:async';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:typed_isolate/typed_isolate.dart';

class LlamaParent {
  final _controller = StreamController<String>.broadcast();
  final _parent = IsolateParent<LlamaCommand, LlamaResponse>();

  StreamSubscription<LlamaResponse>? _subscription;

  List<Map<String, dynamic>> messages = [];

  final LlamaLoad loadCommand;
  final PromptFormat? formatter;
  LlamaParent(this.loadCommand, [PromptFormat? formatter])
      : formatter = formatter ?? loadCommand.format;

  Stream<String> get stream => _controller.stream;

  void _onData(LlamaResponse data) {
    if (data.isDone) _controller.close();
    _parseResponse(data.text);
  }

  void _parseResponse(String response) {
    final processed =
        formatter == null ? response : formatter!.filterResponse(response);
    if (processed != null) _controller.add(processed);
  }

  Future<void> init() async {
    _parent.init();
    _subscription = _parent.stream.listen(_onData);
    await _parent.spawn(LlamaChild());
    _parent.sendToChild(
        data: LlamaInit(
            Llama.libraryPath,
            loadCommand.modelParams,
            loadCommand.contextParams,
            loadCommand.samplingParams,
            loadCommand.format),
        id: 1);
    _parent.sendToChild(data: loadCommand, id: 1);
  }

  void sendPrompt(String prompt) async {
    final formattedPrompt = messages.isEmpty
        ? formatter?.formatPrompt(prompt) ?? prompt
        : formatter?.formatMessages(messages) ?? prompt;
    _parent.sendToChild(id: 1, data: LlamaPrompt(formattedPrompt));
  }

  void stop() => _parent.sendToChild(id: 1, data: LlamaStop());

  Future<void> dispose() async {
    await _subscription?.cancel();
    await _controller.close();
    _parent.sendToChild(id: 1, data: LlamaClear());
    _parent.dispose();
  }
}
