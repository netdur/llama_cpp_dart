import 'sequence_filter.dart';

class ChatMLFilter extends SequenceFilter {
  ChatMLFilter()
      : super(['<|im_start|>user', '<|im_start|>assistant', '<|im_end|>']);
}
