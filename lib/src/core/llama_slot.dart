import 'dart:ffi';

import 'llama_cpp.dart';
import 'service/utf8_accumulator.dart';

/// Holds the state for a specific user/conversation slot.
class LlamaSlot {
  final Pointer<llama_context> context;
  int nPos = 0;
  int nPrompt = 0;
  int nGeneratedTotal = 0;
  Utf8Accumulator accumulator = Utf8Accumulator();

  LlamaSlot(this.context);
}
