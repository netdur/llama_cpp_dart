// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:math';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// Simple RAG (Retrieval-Augmented Generation) implementation
/// This sample demonstrates:
/// 1. Loading and chunking a document
/// 2. Creating embeddings for each chunk
/// 3. Finding the most relevant chunks for a query
/// 4. Generating an answer using the context from relevant chunks

// Configuration with essential settings only
class Config {
  // Paths
  static const String llamaLibraryPath = "bin/MAC_ARM64/libllama.dylib";
  static const String embeddingModelPath =
      "/Users/adel/Workspace/gguf/multilingual-e5-large-instruct-q8_0.gguf";
  static const String llmModelPath =
      "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf";
  static const String documentPath = "example/pie.txt";

  // Default query if none provided
  static const String defaultQuery = "how long to bake?";

  // Chunking settings
  static const int chunkSize = 150;
  static const int chunkOverlap = 2;

  // Embedding and generation settings
  static const int contextSize = 2048;
  static const int maxTokens = 500;

  // RAG settings
  static const int topChunksToUse = 3;
}

void main() async {
  print("Starting simple RAG demo...");
  Llama.libraryPath = Config.llamaLibraryPath;

  // Step 1: Load and chunk the document
  final document = loadDocument(Config.documentPath);
  final chunks = chunkDocument(document);
  print("Document loaded and split into ${chunks.length} chunks");

  // Step 2: Create embeddings for all chunks
  final embeddings = await createEmbeddings(chunks);
  print("Created embeddings for all chunks");

  // Step 3: Process the user query
  final query = Config.defaultQuery;
  print("\nQuery: \"$query\"");

  // Step 4: Find most relevant chunks
  final relevantChunks = findRelevantChunks(query, chunks, embeddings);
  print("\nTop ${relevantChunks.length} most relevant chunks:");
  for (var i = 0; i < relevantChunks.length; i++) {
    print("${i + 1}. ${relevantChunks[i]}\n");
  }

  // Step 5: Generate an answer using retrieved context
  await generateAnswer(query, relevantChunks);
}

/// Load document from file
String loadDocument(String path) {
  try {
    return File(path).readAsStringSync();
  } catch (e) {
    print("Error loading document: $e");
    return ""; // Return empty string for sample
  }
}

/// Split document into chunks with overlap
List<String> chunkDocument(String text) {
  final chunker = TextChunker(
    maxChunkSize: Config.chunkSize,
    overlapSentences: Config.chunkOverlap,
  );
  return chunker.chunk(text);
}

/// Create embeddings for all chunks
Future<List<List<double>>> createEmbeddings(List<String> chunks) async {
  // Initialize embedding model
  final modelParams = ModelParams();
  final contextParams = ContextParams()
    ..embeddings = true
    ..nCtx = Config.contextSize;
  final embedModel = Llama(
    Config.embeddingModelPath,
    modelParams,
    contextParams,
    SamplerParams(),
  );

  print("Creating embeddings for ${chunks.length} chunks...");
  final embeddings = <List<double>>[];

  // Process each chunk
  for (var i = 0; i < chunks.length; i++) {
    try {
      // Print progress every 10 chunks
      if (i % 10 == 0) {
        print("Processing chunk ${i + 1}/${chunks.length}");
      }

      final embedding = embedModel.getEmbeddings(chunks[i]);
      embeddings.add(embedding);
    } catch (e) {
      print("Error embedding chunk ${i + 1}: $e");
      // Add a default embedding to maintain indices
      final defaultSize = embeddings.isNotEmpty ? embeddings.first.length : 384;
      embeddings.add(List.filled(defaultSize, 0.0));
    }
  }

  // Cleanup
  embedModel.dispose();
  return embeddings;
}

/// Find the most relevant chunks for a query
List<String> findRelevantChunks(
  String query,
  List<String> chunks,
  List<List<double>> chunkEmbeddings,
) {
  // Initialize embedding model for the query
  final modelParams = ModelParams();
  final contextParams = ContextParams()
    ..embeddings = true
    ..nCtx = Config.contextSize;
  final model = Llama(
    Config.embeddingModelPath,
    modelParams,
    contextParams,
    SamplerParams(),
  );

  // Get query embedding
  final queryEmbedding = model.getEmbeddings(query);
  model.dispose();

  // Calculate similarity scores
  final similarities = <double>[];
  for (final embedding in chunkEmbeddings) {
    similarities.add(cosineSimilarity(queryEmbedding, embedding));
  }

  // Find top chunks
  final indices = List.generate(chunks.length, (i) => i);
  indices.sort((a, b) => similarities[b].compareTo(similarities[a]));

  // Return top chunks
  final topIndices = indices.take(Config.topChunksToUse).toList();
  return topIndices.map((i) => chunks[i]).toList();
}

/// Generate an answer based on query and relevant chunks
Future<void> generateAnswer(String query, List<String> relevantChunks) async {
  // Combine chunks into context
  final context = relevantChunks.join("\n\n");

  // Initialize LLM
  final modelParams = ModelParams();
  final contextParams = ContextParams()
    ..nPredict = Config.maxTokens
    ..nCtx = Config.contextSize;
  final samplerParams = SamplerParams()
    ..temp = 0.7
    ..topP = 0.95;

  final llm = Llama(
    Config.llmModelPath,
    modelParams,
    contextParams,
    samplerParams,
  );

  // Create prompt
  final prompt = """
<start_of_turn>user
Context information:
$context

Please answer this question based on the context provided: $query
<end_of_turn>

<start_of_turn>model
""";

  print("Generating answer...");
  llm.setPrompt(prompt);

  // Generate response
  try {
    while (true) {
      var (token, done) = llm.getNext();
      stdout.write(token);
      if (done) break;
    }
  } catch (e) {
    print("Error during generation: $e");
  }
  stdout.write("\n");

  // Clean up
  llm.dispose();
}

/// Calculate cosine similarity between two vectors
double cosineSimilarity(List<double> a, List<double> b) {
  if (a.isEmpty || b.isEmpty || a.length != b.length) {
    return 0.0;
  }

  double dotProduct = 0.0;
  double normA = 0.0;
  double normB = 0.0;

  for (var i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = sqrt(normA);
  normB = sqrt(normB);

  if (normA < 1e-10 || normB < 1e-10) return 0.0;
  return dotProduct / (normA * normB);
}

/// Utility class for splitting text into chunks
class TextChunker {
  final int maxChunkSize;
  final int overlapSentences;

  TextChunker({
    required this.maxChunkSize,
    this.overlapSentences = 1,
  });

  List<String> chunk(String text) {
    // Split text into sentences
    final sentences = text
        .replaceAll('\n', ' ')
        .split(RegExp(r'(?<=[.!?])\s+'))
        .where((s) => s.trim().isNotEmpty)
        .toList();

    final chunks = <String>[];
    var currentChunk = <String>[];
    var currentLength = 0;

    for (final sentence in sentences) {
      // If adding this sentence would exceed max size, finalize current chunk
      if (currentLength + sentence.length > maxChunkSize &&
          currentChunk.isNotEmpty) {
        chunks.add(currentChunk.join(' '));

        // Keep overlap sentences for context continuity
        if (overlapSentences > 0 && currentChunk.length >= overlapSentences) {
          currentChunk =
              currentChunk.sublist(currentChunk.length - overlapSentences);
          currentLength = currentChunk.join(' ').length;
        } else {
          currentChunk = [];
          currentLength = 0;
        }
      }

      currentChunk.add(sentence);
      currentLength += sentence.length + 1; // +1 for space
    }

    // Add the last chunk if not empty
    if (currentChunk.isNotEmpty) {
      chunks.add(currentChunk.join(' '));
    }

    return chunks;
  }
}
