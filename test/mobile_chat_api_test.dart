import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:llama_cpp_dart/src/isolate/messages.dart';
import 'package:test/test.dart';

void main() {
  group('ContextParams.mobile', () {
    test('uses memory-conscious defaults', () {
      final params = ContextParams.mobile();

      expect(params.nCtx, 1024);
      expect(params.nBatch, 128);
      expect(params.nUbatch, 128);
      expect(params.typeK, KvCacheType.f16);
      expect(params.typeV, KvCacheType.f16);
      expect(params.nSeqMax, 1);
    });

    test('accepts sizing and KV-cache overrides', () {
      final params = ContextParams.mobile(
        nCtx: 2048,
        nBatch: 256,
        nUbatch: 64,
        typeK: KvCacheType.q8_0,
        typeV: KvCacheType.q8_0,
      );

      expect(params.nCtx, 2048);
      expect(params.nBatch, 256);
      expect(params.nUbatch, 64);
      expect(params.typeK, KvCacheType.q8_0);
      expect(params.typeV, KvCacheType.q8_0);
      expect(ContextParams.fromJson(params.toJson()).toJson(), params.toJson());
    });
  });

  group('GenerateChatCommand context shift', () {
    test('defaults to no shifting', () {
      const command = GenerateChatCommand(
        1,
        sessionId: 2,
        messages: <ChatMessage>[ChatMessage.user('hello')],
        sampler: SamplerParams(),
        maxTokens: 8,
      );

      expect(command.shiftPolicy, ContextShiftPolicy.off);
      expect(command.shift, same(ContextShift.defaults));
    });

    test('carries auto-shift configuration to the worker', () {
      const shift = ContextShift(
        nKeep: 32,
        nDiscard: 16,
        keepBos: false,
      );
      const command = GenerateChatCommand(
        1,
        sessionId: 2,
        messages: <ChatMessage>[ChatMessage.user('hello')],
        sampler: SamplerParams(),
        maxTokens: 8,
        shiftPolicy: ContextShiftPolicy.auto,
        shift: shift,
      );

      expect(command.shiftPolicy, ContextShiftPolicy.auto);
      expect(command.shift.nKeep, 32);
      expect(command.shift.nDiscard, 16);
      expect(command.shift.keepBos, isFalse);
    });
  });
}
