import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../ffi/bindings.dart';
import '../ffi/library_loader.dart';
import '../model/vocab.dart';
import '../types/exceptions.dart';

/// Encode/decode between strings and token ids using a [LlamaVocab].
///
/// Stateless and cheap to construct. Hold one alongside the model.
final class Tokenizer {
  final LlamaVocab vocab;
  const Tokenizer(this.vocab);

  /// Tokenize [text].
  ///
  /// [addSpecial] adds BOS/special prefix tokens when the vocab requests it.
  /// [parseSpecial] interprets `<...>` style strings as special tokens.
  List<int> encode(
    String text, {
    bool addSpecial = true,
    bool parseSpecial = true,
  }) {
    if (text.isEmpty) return const <int>[];
    final b = LlamaLibrary.bindings;
    final bytes = utf8.encode(text);

    final textPtr = calloc<Char>(bytes.length);
    try {
      textPtr.cast<Uint8>().asTypedList(bytes.length).setAll(0, bytes);

      // Probe with a 0-sized buffer; result is `-needed` if too small.
      final probe = b.llama_tokenize(
        vocab.pointer,
        textPtr,
        bytes.length,
        nullptr,
        0,
        addSpecial,
        parseSpecial,
      );

      if (probe >= 0) {
        // Empty input was caught above; an exact 0-fit here is unexpected
        // but harmless. Anything else indicates a misuse.
        if (probe == 0) return const <int>[];
        throw LlamaTokenizeException(
          'tokenize probe returned positive count: $probe',
        );
      }

      final needed = -probe;
      final out = calloc<llama_token>(needed);
      try {
        final n = b.llama_tokenize(
          vocab.pointer,
          textPtr,
          bytes.length,
          out,
          needed,
          addSpecial,
          parseSpecial,
        );
        if (n < 0) {
          throw LlamaTokenizeException('tokenize failed: $n');
        }
        return List<int>.generate(n, (i) => out[i], growable: false);
      } finally {
        calloc.free(out);
      }
    } finally {
      calloc.free(textPtr);
    }
  }

  /// Detokenize a single [token] into its UTF-8 piece.
  ///
  /// Some pieces are partial UTF-8 sequences; for streaming consumption use
  /// the byte-level form via [encodeToken] and accumulate manually.
  String decode(int token, {bool unparseSpecial = true}) {
    return decodeAll([token], unparseSpecial: unparseSpecial);
  }

  /// Detokenize a list of tokens into a UTF-8 string.
  String decodeAll(
    List<int> tokens, {
    bool removeSpecial = false,
    bool unparseSpecial = true,
  }) {
    if (tokens.isEmpty) return '';
    final b = LlamaLibrary.bindings;

    final tokensPtr = calloc<llama_token>(tokens.length);
    try {
      for (var i = 0; i < tokens.length; i++) {
        tokensPtr[i] = tokens[i];
      }
      final probe = b.llama_detokenize(
        vocab.pointer,
        tokensPtr,
        tokens.length,
        nullptr,
        0,
        removeSpecial,
        unparseSpecial,
      );
      if (probe >= 0) {
        if (probe == 0) return '';
        throw LlamaTokenizeException(
          'detokenize probe returned positive count: $probe',
        );
      }
      final needed = -probe;
      final out = calloc<Char>(needed);
      try {
        final n = b.llama_detokenize(
          vocab.pointer,
          tokensPtr,
          tokens.length,
          out,
          needed,
          removeSpecial,
          unparseSpecial,
        );
        if (n < 0) {
          throw LlamaTokenizeException('detokenize failed: $n');
        }
        return utf8.decode(
          out.cast<Uint8>().asTypedList(n),
          allowMalformed: true,
        );
      } finally {
        calloc.free(out);
      }
    } finally {
      calloc.free(tokensPtr);
    }
  }

  /// Raw byte form of a single token's piece — useful when streaming, where
  /// piece boundaries do not align with UTF-8 codepoints.
  Uint8List encodeToken(int token, {bool special = true}) {
    final b = LlamaLibrary.bindings;
    final probe = b.llama_token_to_piece(
      vocab.pointer,
      token,
      nullptr,
      0,
      0,
      special,
    );
    if (probe >= 0) {
      if (probe == 0) return Uint8List(0);
      throw LlamaTokenizeException(
        'token_to_piece probe returned positive count: $probe',
      );
    }
    final needed = -probe;
    final out = calloc<Char>(needed);
    try {
      final n = b.llama_token_to_piece(
        vocab.pointer,
        token,
        out,
        needed,
        0,
        special,
      );
      if (n < 0) {
        throw LlamaTokenizeException('token_to_piece failed: $n');
      }
      return Uint8List.fromList(out.cast<Uint8>().asTypedList(n));
    } finally {
      calloc.free(out);
    }
  }
}
