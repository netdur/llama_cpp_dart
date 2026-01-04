import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_dart/src/model_params.dart';
import 'package:llama_cpp_dart/src/context_params.dart';
import 'package:llama_cpp_dart/src/sampler_params.dart';

void main() {
  group('ModelParams', () {
    test('defaults are correct', () {
      final params = ModelParams();
      expect(params.nGpuLayers, 99);
      expect(params.splitMode, LlamaSplitMode.none);
      expect(params.mainGpu, 0);
      expect(params.vocabOnly, false);
      expect(params.useMemorymap, true);
    });

    test('serialization works', () {
      final params = ModelParams();
      params.nGpuLayers = 50;
      final json = params.toJson();
      final reconstructed = ModelParams.fromJson(json);
      expect(reconstructed.nGpuLayers, 50);
    });

    test('dispose does not crash', () {
      final params = ModelParams();
      // We can't easily test native free dispatch without mocking or running in valid env,
      // but we can call it.
      // _tensorSplitPtr is only allocated if tensorSplit is not empty in get().
      // get() requires native lib loaded effectively? No, it calls Llama.lib.
      // Llama.lib might be null or throw if not initialized.
      // The dispose method only frees _tensorSplitPtr.
      params.dispose();
    });
  });

  group('ContextParams', () {
    test('defaults are correct', () {
      final params = ContextParams();
      expect(params.nCtx, 512);
      expect(params.nBatch, 512);
      expect(params.ropeScalingType, LlamaRopeScalingType.unspecified);
    });

    test('serialization works', () {
      final params = ContextParams();
      params.nCtx = 1024;
      params.ropeScalingType = LlamaRopeScalingType.yarn;
      final json = params.toJson();
      final reconstructed = ContextParams.fromJson(json);
      expect(reconstructed.nCtx, 1024);
      expect(reconstructed.ropeScalingType, LlamaRopeScalingType.yarn);
    });
  });

  group('SamplerParams', () {
    test('defaults are correct', () {
      final params = SamplerParams();
      expect(params.temp, 0.80);
      expect(params.topK, 40);
      expect(params.topP, 0.95);
    });

    test('serialization works', () {
      final params = SamplerParams();
      params.temp = 0.5;
      params.topK = 20;
      final json = params.toJson();
      final reconstructed = SamplerParams.fromJson(json);
      expect(reconstructed.temp, 0.5);
      expect(reconstructed.topK, 20);
    });
  });
}
