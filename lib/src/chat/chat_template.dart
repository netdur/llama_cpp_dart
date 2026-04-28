import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import '../model/model.dart';
import '../types/exceptions.dart';
import 'chat_message.dart';

/// Thin Dart wrapper over `llama_chat_apply_template` and
/// `llama_model_chat_template`.
///
/// All work is delegated to llama.cpp — this binding does not parse Jinja.
/// llama.cpp recognizes a fixed set of common templates by pattern (ChatML,
/// Llama 2/3, Gemma, Mistral, Phi, Qwen, etc.). Custom templates that don't
/// match a known pattern fail; in that case pass an explicit [template]
/// string for one of the supported families, or render the prompt by hand.
final class ChatTemplate {
  ChatTemplate._();

  /// Read the chat template embedded in [model]'s GGUF metadata.
  ///
  /// [name] selects a non-default template variant (e.g. `tool_use`); pass
  /// `null` to fetch the default template. Returns `null` if the model has
  /// no template at that key.
  static String? fromModel(LlamaModel model, {String? name}) {
    final b = LlamaLibrary.bindings;
    final namePtr = name == null
        ? Pointer<Char>.fromAddress(0)
        : name.toNativeUtf8(allocator: calloc).cast<Char>();
    try {
      final result = b.llama_model_chat_template(model.pointer, namePtr);
      if (result == nullptr) return null;
      return result.cast<Utf8>().toDartString();
    } finally {
      if (namePtr != nullptr) calloc.free(namePtr);
    }
  }

  /// List the chat template names llama.cpp knows about. Useful as a
  /// fallback when a model has no embedded template.
  static List<String> builtinNames() {
    final b = LlamaLibrary.bindings;
    // Probe required size first.
    final probe = b.llama_chat_builtin_templates(nullptr, 0);
    if (probe <= 0) return const <String>[];
    final out = calloc<Pointer<Char>>(probe);
    try {
      final n = b.llama_chat_builtin_templates(out, probe);
      return List<String>.generate(
        n,
        (i) => out[i].cast<Utf8>().toDartString(),
        growable: false,
      );
    } finally {
      calloc.free(out);
    }
  }

  /// Apply [template] to [messages] and return the rendered prompt string.
  ///
  /// [addAssistant] appends the prefix that begins an assistant message —
  /// set to `true` when the next thing the model should produce is the
  /// assistant's reply.
  static String apply({
    required String template,
    required List<ChatMessage> messages,
    bool addAssistant = true,
  }) {
    if (messages.isEmpty) {
      throw const ChatTemplateException('messages must be non-empty');
    }
    final b = LlamaLibrary.bindings;

    final tmplPtr = template.toNativeUtf8(allocator: calloc);
    final msgArray = calloc<llama_chat_message>(messages.length);
    final stringArenas = <Pointer<Utf8>>[];

    try {
      for (var i = 0; i < messages.length; i++) {
        final rolePtr = messages[i].role.toNativeUtf8(allocator: calloc);
        final contentPtr = messages[i].content.toNativeUtf8(allocator: calloc);
        stringArenas
          ..add(rolePtr)
          ..add(contentPtr);
        msgArray[i].role = rolePtr.cast();
        msgArray[i].content = contentPtr.cast();
      }

      final probe = b.llama_chat_apply_template(
        tmplPtr.cast(),
        msgArray,
        messages.length,
        addAssistant,
        nullptr,
        0,
      );
      if (probe < 0) {
        throw ChatTemplateException(
          'llama_chat_apply_template probe failed (rc=$probe). '
          'The template may not be one of llama.cpp\'s recognized formats.',
        );
      }
      if (probe == 0) return '';

      final cap = probe + 1; // room for trailing NUL just in case
      final buf = calloc<Char>(cap);
      try {
        final n = b.llama_chat_apply_template(
          tmplPtr.cast(),
          msgArray,
          messages.length,
          addAssistant,
          buf,
          cap,
        );
        if (n < 0) {
          throw ChatTemplateException(
            'llama_chat_apply_template failed (rc=$n)',
          );
        }
        return buf.cast<Utf8>().toDartString(length: n);
      } finally {
        calloc.free(buf);
      }
    } finally {
      calloc.free(tmplPtr);
      for (final p in stringArenas) {
        calloc.free(p);
      }
      calloc.free(msgArray);
    }
  }
}

/// Thrown when chat template rendering fails.
final class ChatTemplateException extends LlamaException {
  const ChatTemplateException(super.message);
}
