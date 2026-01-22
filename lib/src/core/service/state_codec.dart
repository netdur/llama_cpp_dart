import 'dart:typed_data';

class StatePayload {
  final Uint8List payload;
  final int? nPos;
  final int? nKeep;
  final bool hasHeader;

  StatePayload(this.payload, this.nPos, this.nKeep, this.hasHeader);
}

class StateCodec {
  static const int headerSize = 16;
  static const int magic = 0x44415254; // "DART"
  static const int version = 1;

  static StatePayload decode(Uint8List bytes) {
    if (bytes.length < headerSize) {
      return StatePayload(bytes, null, null, false);
    }

    final header = ByteData.sublistView(bytes, 0, headerSize);
    final magicValue = header.getUint32(0, Endian.little);
    final versionValue = header.getUint32(4, Endian.little);
    if (magicValue != magic || versionValue != version) {
      return StatePayload(bytes, null, null, false);
    }

    final nPos = header.getUint32(8, Endian.little);
    final nKeep = header.getUint32(12, Endian.little);
    final payload = bytes.sublist(headerSize);
    return StatePayload(payload, nPos, nKeep, true);
  }

  static Uint8List encode(
    Uint8List payload, {
    required int nPos,
    required int nKeep,
  }) {
    final header = ByteData(headerSize);
    header.setUint32(0, magic, Endian.little);
    header.setUint32(4, version, Endian.little);
    header.setUint32(8, nPos, Endian.little);
    header.setUint32(12, nKeep, Endian.little);

    final allBytes = Uint8List(headerSize + payload.length);
    allBytes.setAll(0, header.buffer.asUint8List());
    allBytes.setAll(headerSize, payload);
    return allBytes;
  }
}
