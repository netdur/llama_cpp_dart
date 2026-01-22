import 'dart:async';
import 'dart:collection';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../llama_types.dart';
import '../llama_cpp.dart';
import 'pending_item.dart';
import 'utf8_accumulator.dart';

class SessionEvent {
  final int requestId;
  final String text;
  SessionEvent(this.requestId, this.text);
}

enum SessionTier {
  hot, // In VRAM (Active)
  warm, // In RAM (Uint8List snapshot)
  cold, // On Disk (File path)
}

class TokenUsage {
  int promptTokens = 0;
  int completionTokens = 0;
  final DateTime startTime = DateTime.now();
  DateTime? endTime;

  int get total => promptTokens + completionTokens;

  double get tokensPerSecond {
    final end = endTime ?? DateTime.now();
    final duration = end.difference(startTime).inMilliseconds / 1000.0;
    if (duration <= 0) return 0.0;
    return completionTokens / duration;
  }
}

class ServiceSession {
  final String id;
  int? seqId;
  final Pointer<llama_sampler> sampler;

  int requestId = 0;
  int bufferRequestId = 0;

  LlamaStatus status = LlamaStatus.uninitialized;

  Object? lastError;
  StackTrace? lastStackTrace;
  DateTime? lastErrorTime;

  Pointer<Char> decodeBuf = nullptr;
  int decodeCapacity = 0;
  Utf8Accumulator accumulator = Utf8Accumulator();

  final ListQueue<PendingItem> pendingItems = ListQueue<PendingItem>();

  late final StreamController<SessionEvent> controller;

  // requestId -> completion
  final Map<int, Completer<void>> requestCompleters = {};

  int nPos = 0;
  int nPromptTokens = 0;
  int nGenerated = 0;
  int nKeep = 0; // Number of tokens to preserve (System Prompt)
  TokenUsage usage = TokenUsage();

  SessionTier tier = SessionTier.warm; // Start warm (waiting for slot)
  Uint8List? stateBuffer;
  String? coldFilePath;
  DateTime lastActiveTime = DateTime.now();

  // Output buffering
  final StringBuffer outputBuffer = StringBuffer();
  DateTime lastFlushTime = DateTime.now();

  bool isDisposed = false;

  ServiceSession({
    required this.id,
    required this.seqId,
    required this.sampler,
  }) {
    // Broadcast stream with replay of buffered output when a listener attaches.
    controller = StreamController<SessionEvent>.broadcast(
      onListen: () {
        // If we have buffered output for the last request, flush it now.
        if (outputBuffer.isNotEmpty) {
          flush(force: true, requestId: bufferRequestId);
        }
      },
    );
  }

  Stream<SessionEvent> get stream => controller.stream;

  void flush({bool force = false, required int requestId}) {
    if (outputBuffer.isEmpty) return;

    // If no listeners, keep buffering (bounded).
    if (!controller.hasListener) {
      bufferRequestId = requestId;
      if (outputBuffer.length > 5 * 1024 * 1024) {
        outputBuffer.clear();
        outputBuffer.write(
          "[Error: Output buffer exceeded 5MB with no listeners. Stream dropped.]",
        );
      }
      return;
    }

    final now = DateTime.now();
    if (force ||
        outputBuffer.length > 1024 ||
        now.difference(lastFlushTime).inMilliseconds > 50) {
      if (!controller.isClosed) {
        controller.add(SessionEvent(requestId, outputBuffer.toString()));
      }
      outputBuffer.clear();
      lastFlushTime = now;
    }
  }

  void completeGeneration(int requestId) {
    // Ensure pending buffer is emitted (or stays buffered for replay).
    flush(force: true, requestId: requestId);

    final c = requestCompleters[requestId];
    if (c != null && !c.isCompleted) c.complete();

    // Drop old completers to avoid leaks.
    requestCompleters.removeWhere((k, _) => k < requestId);
  }

  void dispose(llama_cpp lib) {
    if (isDisposed) return;
    isDisposed = true;

    if (sampler != nullptr) {
      lib.llama_sampler_free(sampler);
    }
    if (decodeBuf != nullptr) {
      malloc.free(decodeBuf);
      decodeBuf = nullptr;
    }
    controller.close();
  }
}
