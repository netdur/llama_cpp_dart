import 'dart:async';
import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

// --- CONFIGURATION ---
const kModelPath = "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf";
const kMmprojPath = "/Users/adel/Workspace/gguf/mmproj-model-f16-4B.gguf";
const kImagePath = "/Users/adel/Downloads/Untitlehjhfjd.jpg";
const kPrompt = "Describe this image in one sentence.";
const kConcurrency = 2; // Number of concurrent requests
// ---------------------

Future<void> main() async {
  // 1. Initialize LlamaService with Vision support
  print("Loading model: $kModelPath");
  print("Loading mmproj: $kMmprojPath");

  // Ensure paths exist
  if (!File(kModelPath).existsSync()) {
    print("Error: Model file not found at $kModelPath");
    return;
  }
  if (!File(kMmprojPath).existsSync()) {
    print("Error: MMProj file not found at $kMmprojPath");
    return;
  }
  if (!File(kImagePath).existsSync()) {
    print("Error: Image file not found at $kImagePath");
    return;
  }

  final contextParams = ContextParams();
  contextParams.nSeqMax = kConcurrency;
  contextParams.nCtx = 8192 * 2; // Ensure enough context for image + text

  // Adjust library path if needed (optional)
  if (Platform.isMacOS) {
    Llama.libraryPath =
        "/Users/adel/Workspace/llama_cpp_dart/bin/MAC_ARM64/libllama.dylib";
  }

  final service = LlamaService(
    kModelPath,
    mmprojPath: kMmprojPath,
    contextParams: contextParams,
    verbose: false, // Set to true for debug output
  );

  print("Service initialized. Initializing sessions...");

  // 2. Create Sessions
  final samplerParams = SamplerParams();
  samplerParams.temp = 0.1;

  for (int i = 0; i < kConcurrency; i++) {
    service.createSession("session_$i", samplerParams: samplerParams);
  }

  print("Starting $kConcurrency concurrent multimodal requests...");
  print("-" * 60);

  // 3. Prepare Input
  final imageBytes = await File(kImagePath).readAsBytes();
  final inputs = [
    LlamaImage.fromBytes(imageBytes), // The image
  ];

  // Note: The prompt must contain <image> for each image provided if the model requires it
  // Most valid multimodal models behave like this.
  // We prepend <image> to the user prompt.
  // Gemma 3 Instruction format
  final fullPrompt =
      "<start_of_turn>user\n<image>\n$kPrompt<end_of_turn>\n<start_of_turn>model\n";

  // 4. Run Requests in Parallel
  final futures = <Future<void>>[];
  for (int i = 0; i < kConcurrency; i++) {
    futures.add(_runSession(service, "session_$i", fullPrompt, inputs));
  }

  await Future.wait(futures);

  print("-" * 60);
  print("All requests completed.");
  await service.dispose();
}

Future<void> _runSession(
  LlamaService service,
  String sessionId,
  String prompt,
  List<LlamaInput> inputs,
) async {
  final buffer = StringBuffer();
  print("[$sessionId] Starting generation...");

  try {
    // We use generateWithMedia for multimodal
    await for (final chunk in service.generateWithMedia(
      sessionId,
      prompt,
      inputs: inputs,
    )) {
      buffer.write(chunk);
      stdout.write(chunk); // Stream to stdout mixed (might be messy)
    }
  } catch (e) {
    print("\n[$sessionId] Error: $e");
  }

  print("\n[$sessionId] DONE. Length: ${buffer.length} chars");
}
