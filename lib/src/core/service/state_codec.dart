import 'dart:typed_data';

class StatePayload {
  final Uint8List payload;
  final int? nPos;
  final int? nKeep;
  final bool hasHeader;

  StatePayload(this.payload, this.nPos, this.nKeep, this.hasHeader);
}
