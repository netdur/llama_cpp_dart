import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:system_info2/system_info2.dart';

void main() async {
  final cores = SysInfo.cores;

  Llama.libraryPath = "./libllama.dylib";

  SamplingParams samplingParams = SamplingParams()
    ..penaltyRepeat = 1.1
    ..temp = 0.7;

  ModelParams modelParams = ModelParams();

  ContextParams contextParams = ContextParams()
    ..threads = cores.length
    ..threadsBatch = cores.length
    ..context = 512 * 4
    ..batch = 512 * 4;

  Llama llama = Llama("/Users/adel/Downloads/gemma-7b-it-Q4_K_M.gguf",
      modelParams, contextParams, samplingParams);

  int tokensLimit = 500;

  String filePath = "/Users/adel/Downloads/alice_in_wonderland.txt";
  String text = await File(filePath).readAsString();
  List<String> words = text.split(RegExp(r'\s+'));

  List<String> chunks = [];
  List<int> allTokens = [];
  String processedText = '';

  for (var word in words) {
    List<int> tokens = llama.tokenize(word, false);
    if (allTokens.length + tokens.length > tokensLimit) {
      break;
    }
    allTokens.addAll(tokens);
    processedText += '$word ';
  }

  String prompt = """### Task:
Divide the provided text into semantically coherent text, aiming for chunk text to contain between 250-350 words. Preserve logical and thematic continuity within each chunk, ensuring complete ideas or scenes remain intact.

### Guidelines:
1. Identify natural breaks such as paragraph ends or shifts in scene to start new chunk.
2. Estimate word counts to include content in a chunk. Initiate a new chunk upon reaching roughly 250 words, ending on a natural break near this count, without exceeding 350 words if possible.
3. If text does not neatly fit within word limits, prioritize the integrity of ideas and sentences. Slightly exceeding the word count to maintain coherence is acceptable.
4. Use examples of good segmentation: End chunks at scene conclusions, shifts in narrative focus, or before a new major action begins.
5. Iteratively refine chunk boundaries for semantic coherence and continuity. Consider adding transitional phrases to smooth the narrative flow between chunks.

Your primary goal is to ensure a smooth and logical flow of content across chunks, allowing for slight deviations from the word count range as needed to achieve this.

JSON format {
  "chunk": "",
  "next_chunk_starting_point": "" 
}

### Content: $processedText\n

### Assistant:""";

  print("prompt");
  print(prompt);
  llama.setPrompt(prompt);
  print("result");
  while (true) {
    var (token, done) = llama.getNext();
    stdout.write(token);
    if (done) {
      break;
    }
  }
  print("");

  llama.dispose();
}
