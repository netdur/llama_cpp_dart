import 'dart:convert';
import 'dart:typed_data';

import '../types/exceptions.dart';

/// Wire format version of the state codec. Bumped when the binary layout
/// changes incompatibly. Files written with a different version are
/// rejected on load.
const int stateCodecVersion = 1;

const _magicHead = 'LCDC';
const _magicTail = 'LCDc';

/// Thrown when a state file can't be parsed or doesn't match the runtime.
final class LlamaStateException extends LlamaException {
  /// Discriminator for what went wrong; lets callers branch (e.g. ask the
  /// user to re-load the same model vs. delete the file).
  final LlamaStateError reason;

  const LlamaStateException(this.reason, super.message);
}

enum LlamaStateError {
  /// File header didn't match — corrupted or wrong file type.
  badMagic,

  /// File was written by a different codec version we can't read.
  unsupportedVersion,

  /// File ended before all sections were read.
  truncated,

  /// JSON metadata was malformed.
  badMetadata,

  /// Token list length / checksum didn't match metadata.
  tokensMismatch,

  /// Loaded into a different model than the one that produced the file.
  modelMismatch,

  /// Loaded into a context with smaller `nCtx` than the saved KV needs.
  contextTooSmall,

  /// Saved file used a multimodal projector but the runtime has none, or
  /// vice-versa, or the projectors don't match.
  multimodalMismatch,

  /// `llama_state_seq_set_data` rejected the buffer.
  llamaRejected,
}

/// Snapshot of the engine-side identity that produced a state file. Used
/// both as the metadata payload at save time and the comparison target
/// at load time.
final class StateMetadata {
  final int codecVersion;
  final String savedAt; // ISO-8601 UTC
  final String wrapperVersion;

  // Model identity. Path is informational; sizeBytes and nParams form the
  // structural fingerprint. Two different GGUFs with the same path won't
  // match if their sizes or param counts differ.
  final String modelPath;
  final int modelSizeBytes;
  final int modelNParams;
  final int modelNEmbd;
  final int modelNLayer;
  final int modelTrainCtx;

  // Context params that materially change KV layout.
  final int nCtx;
  final int nBatch;
  final int nUbatch;
  final int nSeqMax;
  final bool embeddings;

  // Multimodal identity (null when no projector was loaded).
  final String? mmprojPath;
  final int? mmprojSizeBytes;
  final bool? mmprojSupportsVision;
  final bool? mmprojSupportsAudio;

  // Session state.
  final int seqId;
  final int kvHead;
  final int tokensCount;
  final int tokensChecksum;

  // Caller-supplied opaque payload (e.g. chat message history). JSON
  // round-trippable.
  final Map<String, Object?> extra;

  const StateMetadata({
    required this.codecVersion,
    required this.savedAt,
    required this.wrapperVersion,
    required this.modelPath,
    required this.modelSizeBytes,
    required this.modelNParams,
    required this.modelNEmbd,
    required this.modelNLayer,
    required this.modelTrainCtx,
    required this.nCtx,
    required this.nBatch,
    required this.nUbatch,
    required this.nSeqMax,
    required this.embeddings,
    required this.mmprojPath,
    required this.mmprojSizeBytes,
    required this.mmprojSupportsVision,
    required this.mmprojSupportsAudio,
    required this.seqId,
    required this.kvHead,
    required this.tokensCount,
    required this.tokensChecksum,
    this.extra = const <String, Object?>{},
  });

  Map<String, Object?> toJson() => {
        'codec_version': codecVersion,
        'saved_at': savedAt,
        'wrapper_version': wrapperVersion,
        'model': {
          'path': modelPath,
          'size_bytes': modelSizeBytes,
          'n_params': modelNParams,
          'n_embd': modelNEmbd,
          'n_layer': modelNLayer,
          'train_ctx': modelTrainCtx,
        },
        'context': {
          'n_ctx': nCtx,
          'n_batch': nBatch,
          'n_ubatch': nUbatch,
          'n_seq_max': nSeqMax,
          'embeddings': embeddings,
        },
        'multimodal': mmprojPath == null
            ? null
            : {
                'path': mmprojPath,
                'size_bytes': mmprojSizeBytes,
                'vision': mmprojSupportsVision,
                'audio': mmprojSupportsAudio,
              },
        'session': {
          'seq_id': seqId,
          'kv_head': kvHead,
          'tokens_count': tokensCount,
          'tokens_checksum': tokensChecksum,
        },
        'extra': extra,
      };

  factory StateMetadata.fromJson(Map<String, Object?> json) {
    final model = json['model']! as Map<String, Object?>;
    final ctx = json['context']! as Map<String, Object?>;
    final mm = json['multimodal'] as Map<String, Object?>?;
    final s = json['session']! as Map<String, Object?>;
    return StateMetadata(
      codecVersion: json['codec_version']! as int,
      savedAt: json['saved_at']! as String,
      wrapperVersion: json['wrapper_version']! as String,
      modelPath: model['path']! as String,
      modelSizeBytes: model['size_bytes']! as int,
      modelNParams: model['n_params']! as int,
      modelNEmbd: model['n_embd']! as int,
      modelNLayer: model['n_layer']! as int,
      modelTrainCtx: model['train_ctx']! as int,
      nCtx: ctx['n_ctx']! as int,
      nBatch: ctx['n_batch']! as int,
      nUbatch: ctx['n_ubatch']! as int,
      nSeqMax: ctx['n_seq_max']! as int,
      embeddings: ctx['embeddings']! as bool,
      mmprojPath: mm == null ? null : mm['path'] as String?,
      mmprojSizeBytes: mm == null ? null : mm['size_bytes'] as int?,
      mmprojSupportsVision: mm == null ? null : mm['vision'] as bool?,
      mmprojSupportsAudio: mm == null ? null : mm['audio'] as bool?,
      seqId: s['seq_id']! as int,
      kvHead: s['kv_head']! as int,
      tokensCount: s['tokens_count']! as int,
      tokensChecksum: s['tokens_checksum']! as int,
      extra:
          (json['extra'] as Map<String, Object?>?) ?? const <String, Object?>{},
    );
  }
}

/// Decoded state file ready to apply to a context.
final class DecodedState {
  final StateMetadata metadata;
  final List<int> tokens;
  final Uint8List rawState;

  const DecodedState({
    required this.metadata,
    required this.tokens,
    required this.rawState,
  });
}

/// Encode a state file from its parts.
///
/// The actual KV bytes are produced by `llama_state_seq_get_data`; the
/// caller hands them in as [rawState].
Uint8List encodeState({
  required StateMetadata metadata,
  required List<int> tokens,
  required Uint8List rawState,
}) {
  if (tokens.length != metadata.tokensCount) {
    throw LlamaStateException(
      LlamaStateError.tokensMismatch,
      'metadata.tokensCount=${metadata.tokensCount} '
      'but tokens.length=${tokens.length}',
    );
  }

  final metaJson = utf8.encode(jsonEncode(metadata.toJson()));
  final tokenBytes = Uint8List(tokens.length * 4)
    ..buffer.asInt32List().setAll(0, tokens);

  final bytes = BytesBuilder(copy: false)
    ..add(ascii.encode(_magicHead))
    .._addUint16(stateCodecVersion)
    .._addUint16(0)
    .._addUint32(metaJson.length)
    ..add(metaJson)
    .._addUint32(tokens.length)
    ..add(tokenBytes)
    .._addUint64(rawState.length)
    ..add(rawState)
    ..add(ascii.encode(_magicTail));

  return bytes.takeBytes();
}

/// Decode a state file. Throws [LlamaStateException] on any inconsistency.
DecodedState decodeState(Uint8List file) {
  final reader = _Reader(file);
  final head = reader.readAscii(4);
  if (head != _magicHead) {
    throw const LlamaStateException(
      LlamaStateError.badMagic,
      'magic header mismatch — not a llama_cpp_dart state file',
    );
  }
  final version = reader.readUint16();
  reader.readUint16(); // reserved
  if (version != stateCodecVersion) {
    throw LlamaStateException(
      LlamaStateError.unsupportedVersion,
      'state codec version $version is not supported '
      '(this build expects $stateCodecVersion)',
    );
  }

  final metaLen = reader.readUint32();
  final metaBytes = reader.readBytes(metaLen);
  late final StateMetadata metadata;
  try {
    final json = jsonDecode(utf8.decode(metaBytes)) as Map<String, Object?>;
    metadata = StateMetadata.fromJson(json);
  } on FormatException catch (e) {
    throw LlamaStateException(
      LlamaStateError.badMetadata,
      'metadata JSON decode failed: $e',
    );
  } catch (e) {
    throw LlamaStateException(
      LlamaStateError.badMetadata,
      'metadata schema mismatch: $e',
    );
  }

  final tokenCount = reader.readUint32();
  if (tokenCount != metadata.tokensCount) {
    throw LlamaStateException(
      LlamaStateError.tokensMismatch,
      'token count in body ($tokenCount) does not match metadata '
      '(${metadata.tokensCount})',
    );
  }
  final tokens = <int>[];
  for (var i = 0; i < tokenCount; i++) {
    tokens.add(reader.readInt32());
  }
  final tokenBytes = Uint8List.fromList(
    Uint8List(tokenCount * 4)..buffer.asInt32List().setAll(0, tokens),
  );
  final actualChecksum = fnv1a64(tokenBytes);
  if (actualChecksum != metadata.tokensChecksum) {
    throw LlamaStateException(
      LlamaStateError.tokensMismatch,
      'token checksum mismatch (expected ${metadata.tokensChecksum}, '
      'got $actualChecksum)',
    );
  }

  final stateLen = reader.readUint64();
  final rawState = reader.readBytes(stateLen);

  final tail = reader.readAscii(4);
  if (tail != _magicTail) {
    throw const LlamaStateException(
      LlamaStateError.truncated,
      'trailer magic mismatch — file truncated or corrupted',
    );
  }

  return DecodedState(
    metadata: metadata,
    tokens: tokens,
    rawState: rawState,
  );
}

/// Compute an FNV-1a 64-bit hash. Stable for sanity-checking; not crypto.
int fnv1a64(Uint8List bytes) {
  var h = 0xcbf29ce484222325;
  for (final b in bytes) {
    h ^= b;
    h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF;
  }
  return h;
}

/// Verify a fresh [actual] [StateMetadata] against [saved] from a file.
/// Throws [LlamaStateException] if anything material differs.
void verifyCompatible(StateMetadata saved, StateMetadata actual) {
  if (saved.modelSizeBytes != actual.modelSizeBytes ||
      saved.modelNParams != actual.modelNParams ||
      saved.modelNEmbd != actual.modelNEmbd ||
      saved.modelNLayer != actual.modelNLayer) {
    throw LlamaStateException(
      LlamaStateError.modelMismatch,
      'state was saved against a different model '
      '(saved: ${saved.modelPath}, ${saved.modelSizeBytes} bytes, '
      '${saved.modelNParams} params; '
      'now: ${actual.modelPath}, ${actual.modelSizeBytes} bytes, '
      '${actual.modelNParams} params)',
    );
  }
  if (actual.nCtx < saved.kvHead) {
    throw LlamaStateException(
      LlamaStateError.contextTooSmall,
      'current nCtx=${actual.nCtx} cannot hold the saved KV at '
      'position ${saved.kvHead} — increase ContextParams.nCtx',
    );
  }
  final savedHasMm = saved.mmprojPath != null;
  final actualHasMm = actual.mmprojPath != null;
  if (savedHasMm != actualHasMm) {
    throw LlamaStateException(
      LlamaStateError.multimodalMismatch,
      savedHasMm
          ? 'state was saved with a multimodal projector but the engine has none'
          : 'engine has a multimodal projector but the saved state had none',
    );
  }
  if (savedHasMm && saved.mmprojSizeBytes != actual.mmprojSizeBytes) {
    throw LlamaStateException(
      LlamaStateError.multimodalMismatch,
      'multimodal projector identity differs '
      '(saved: ${saved.mmprojSizeBytes} bytes; '
      'now: ${actual.mmprojSizeBytes} bytes)',
    );
  }
}

class _Reader {
  _Reader(this._bytes);
  final Uint8List _bytes;
  int _i = 0;

  int get _remaining => _bytes.length - _i;

  void _ensure(int n) {
    if (_remaining < n) {
      throw const LlamaStateException(
        LlamaStateError.truncated,
        'file ended unexpectedly while reading',
      );
    }
  }

  String readAscii(int n) {
    _ensure(n);
    final s = ascii.decode(_bytes.sublist(_i, _i + n));
    _i += n;
    return s;
  }

  int readUint16() {
    _ensure(2);
    final v = _bytes.buffer.asByteData().getUint16(_i, Endian.little);
    _i += 2;
    return v;
  }

  int readUint32() {
    _ensure(4);
    final v = _bytes.buffer.asByteData().getUint32(_i, Endian.little);
    _i += 4;
    return v;
  }

  int readInt32() {
    _ensure(4);
    final v = _bytes.buffer.asByteData().getInt32(_i, Endian.little);
    _i += 4;
    return v;
  }

  int readUint64() {
    _ensure(8);
    final v = _bytes.buffer.asByteData().getUint64(_i, Endian.little);
    _i += 8;
    return v;
  }

  Uint8List readBytes(int n) {
    _ensure(n);
    final out = Uint8List.sublistView(_bytes, _i, _i + n);
    _i += n;
    return Uint8List.fromList(out);
  }
}

extension on BytesBuilder {
  void _addUint16(int v) {
    final b = ByteData(2)..setUint16(0, v, Endian.little);
    add(b.buffer.asUint8List());
  }

  void _addUint32(int v) {
    final b = ByteData(4)..setUint32(0, v, Endian.little);
    add(b.buffer.asUint8List());
  }

  void _addUint64(int v) {
    final b = ByteData(8)..setUint64(0, v, Endian.little);
    add(b.buffer.asUint8List());
  }
}
