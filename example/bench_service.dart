import 'dart:async';
import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

// Configurable parameters
// Configurable parameters
const kConcurrency = 10; // Number of parallel requests
const kParamPrompt = "count to 20";

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    print('Usage: dart example/bench_service.dart <path_to_model>');
    exit(1);
  }

  final modelPath = args[0];

  if (args.length > 1) {
    Llama.libraryPath = args[1];
  } else if (Platform.isMacOS) {
    // Default fallback for this workspace
    Llama.libraryPath =
        "/Users/adel/Workspace/llama_cpp_dart/bin/MAC_ARM64/libllama.dylib";
  }

  final contextParams = ContextParams();
  contextParams.nSeqMax = kConcurrency; // Allow multiple concurrent sequences
  contextParams.nCtx =
      8192; // Shared context size (must be large enough for all)

  final samplerParams = SamplerParams();
  samplerParams.temp = 0.7;
  samplerParams.topK = 50;
  samplerParams.topP = 0.90;
  samplerParams.penaltyRepeat = 1.1;

  print("Loading model from $modelPath...");
  print("Using library: ${Llama.libraryPath ?? 'system default'}");
  final service =
      LlamaService(modelPath, contextParams: contextParams, verbose: false);

  // Create sessions
  print("Creating $kConcurrency sessions...");
  for (int i = 0; i < kConcurrency; i++) {
    service.createSession("session_$i", samplerParams: samplerParams);
  }

  // Warmup round (sequential, on session 0)
  print("Warmup request (prompt: '$kParamPrompt')...");
  final gemmaPrompt =
      "<start_of_turn>user\n$kParamPrompt<end_of_turn>\n<start_of_turn>model\n";
  await _runRequest(service, "session_0", gemmaPrompt, true, silent: true);
  print("Warmup done.\n");

  print("Starting parallel benchmark ($kConcurrency concurrent requests)...");
  print("-" * 80);
  print(
      "${'SeqID'.padRight(8)} | ${'TTFT (ms)'.padRight(12)} | ${'Total (ms)'.padRight(12)} | ${'Tokens'.padRight(8)} | ${'Integrity Check'}");
  print("-" * 80);

  final futures = <Future<BenchResult>>[];
  for (int i = 0; i < kConcurrency; i++) {
    futures.add(_runRequest(service, "session_$i", gemmaPrompt, true));
  }

  final results = await Future.wait(futures);

  print("-" * 80);

  // Calculate Averages
  final avgTtft =
      results.map((r) => r.ttft).reduce((a, b) => a + b) / kConcurrency;
  final avgTotal =
      results.map((r) => r.totalTime).reduce((a, b) => a + b) / kConcurrency;
  final maxTotalTime =
      results.map((r) => r.totalTime).reduce((a, b) => a > b ? a : b);
  final totalTokens = results.map((r) => r.tokenCount).reduce((a, b) => a + b);
  final tps = (totalTokens / (maxTotalTime / 1000.0));

  print("Results:");
  print("  Concurrency:        $kConcurrency");
  print("  Average TTFT:       ${avgTtft.toStringAsFixed(2)} ms");
  print("  Average Total Time: ${avgTotal.toStringAsFixed(2)} ms");
  print("  Throughput:         ${tps.toStringAsFixed(2)} tokens/sec (agg)");

  await service.dispose();
}

class BenchResult {
  final int ttft;
  final int totalTime;
  final int tokenCount;
  final String output;
  BenchResult(this.ttft, this.totalTime, this.tokenCount, this.output);
}

Future<BenchResult> _runRequest(
    LlamaService service, String sessionId, String prompt, bool clearHistory,
    {bool silent = false}) async {
  final stopwatch = Stopwatch()..start();
  int? firstTokenTime;
  int tokenCount = 0;
  final outputBuffer = StringBuffer();

  // Subscribe first
  final sub = service.generateText(sessionId).listen((chunk) {
    if (firstTokenTime == null) {
      firstTokenTime = stopwatch.elapsedMilliseconds;
    }

    // Buffer output instead of printing immediately to avoid mixing
    outputBuffer.write(chunk);

    if (!silent) {
      // Print dot to show activity without messing up format
      stdout.write(".");
    }

    if (chunk.isNotEmpty) tokenCount++;
  });

  // Start generation
  await service.setPrompt(sessionId, prompt, clearHistory: clearHistory);

  // Poll for completion
  while (service.status(sessionId) == LlamaStatus.generating) {
    await Future.delayed(const Duration(milliseconds: 5));
  }

  stopwatch.stop();
  await sub.cancel();

  final fullOutput = outputBuffer.toString();
  final integrityCheck = fullOutput.contains("20") ? "PASS" : "FAIL";

  if (!silent) {
    // Print stats row immediately upon completion
    stdout.writeln(); // Clear the dots line
    print(
        "${sessionId.padRight(8)} | ${(firstTokenTime ?? 0).toString().padRight(12)} | ${stopwatch.elapsedMilliseconds.toString().padRight(12)} | ${tokenCount.toString().padRight(8)} | $integrityCheck");
  }

  return BenchResult(firstTokenTime ?? stopwatch.elapsedMilliseconds,
      stopwatch.elapsedMilliseconds, tokenCount, fullOutput);
}
