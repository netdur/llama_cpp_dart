import 'dart:ffi';

import 'package:ffi/ffi.dart';

class PendingItem {
  final int? token;
  final Pointer<Float>? nativeValues; // embeddings in native memory
  final int nTokens;
  int embdOffsetTokens;

  PendingItem.token(this.token)
      : nativeValues = null,
        nTokens = 1,
        embdOffsetTokens = 0;

  PendingItem.embedding(this.nativeValues, this.nTokens,
      {this.embdOffsetTokens = 0})
      : token = null;

  bool get isEmbedding => nativeValues != null;
  int get remainingEmbeddingTokens => nTokens - embdOffsetTokens;

  void dispose() {
    if (nativeValues != null && nativeValues != nullptr) {
      malloc.free(nativeValues!);
    }
  }
}
