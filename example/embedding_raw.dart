// ignore_for_file: avoid_print

import 'dart:ffi';
import 'dart:io';
import 'dart:math';

import 'package:ffi/ffi.dart';
import 'package:llama_cpp_dart/llama_cpp_dart_ffi.dart';

// Custom implementation of batch clearing
void llamaBatchClear(llama_batch batch) {
  // Free any allocated sequence IDs for existing tokens
  for (int i = 0; i < batch.n_tokens; i++) {
    if (batch.seq_id[i] != nullptr) {
      calloc.free(batch.seq_id[i]);
      batch.seq_id[i] = nullptr;
    }
  }

  // Reset the token count
  batch.n_tokens = 0;
}

// Split text into lines using a separator
List<String> splitLines(String s, {String separator = '\n'}) {
  List<String> lines = [];
  int start = 0;
  int end = s.indexOf(separator);

  while (end != -1) {
    lines.add(s.substring(start, end));
    start = end + separator.length;
    end = s.indexOf(separator, start);
  }

  lines.add(s.substring(start)); // Add the last part
  return lines;
}

// Add a sequence of tokens to a batch
void batchAddSeq(llama_batch batch, List<int> tokens, int seqId) {
  int nTokens = tokens.length;
  for (int i = 0; i < nTokens; i++) {
    batch.token[batch.n_tokens + i] = tokens[i];
    batch.pos[batch.n_tokens + i] =
        i; // Use position within sequence, not global position
    batch.n_seq_id[batch.n_tokens + i] = 1;
    batch.seq_id[batch.n_tokens + i] = calloc<llama_seq_id>()..value = seqId;
    batch.logits[batch.n_tokens + i] = i == nTokens - 1 ? 1 : 0;
  }
  batch.n_tokens += nTokens;
}

// Process a batch to generate embeddings
void batchDecode(llama_cpp lib, Pointer<llama_context> ctx, llama_batch batch,
    Pointer<Float> output, int nSeq, int nEmbd, int embdNormalize) {
  final model = lib.llama_get_model(ctx);
  llama_pooling_type poolingType;

  try {
    // Get pooling type using the correct function from bindings
    poolingType = lib.llama_pooling_type1(
        ctx); // Use llama_pooling_type1 instead of llama_pooling_type
    print("Using model pooling type: $poolingType");
  } catch (e) {
    // Default to MEAN pooling for BERT-like models if there's an error
    print("Could not determine pooling type, using MEAN as default: $e");
    poolingType = llama_pooling_type.LLAMA_POOLING_TYPE_MEAN;
  }

  // Run model
  print("batch_decode: n_tokens = ${batch.n_tokens}, n_seq = $nSeq");

  // Determine model type and run the appropriate function
  if (lib.llama_model_has_encoder(model) == true &&
      lib.llama_model_has_decoder(model) == false) {
    // Encoder-only model (like BERT, RoBERTa, etc.)
    print("Using encoder-only model path");
    if (lib.llama_encode(ctx, batch) < 0) {
      print("batch_decode: failed to encode");
      throw Exception("Failed to encode batch");
    }
  } else if (lib.llama_model_has_encoder(model) == false &&
      lib.llama_model_has_decoder(model) == true) {
    // Decoder-only model (like GPT, LLaMA, etc.)
    print("Using decoder-only model path");
    if (lib.llama_decode(ctx, batch) < 0) {
      print("batch_decode: failed to decode");
      throw Exception("Failed to decode batch");
    }
  } else {
    print("Model type not recognized or is encoder-decoder");
    if (lib.llama_decode(ctx, batch) < 0) {
      print("batch_decode: failed to decode");
      throw Exception("Failed to process batch");
    }
  }

  // For sequence-level embeddings (most BERT models), we only need to get one embedding per sequence
  if (poolingType != llama_pooling_type.LLAMA_POOLING_TYPE_NONE) {
    print("Using sequence-level embeddings (pooling type: $poolingType)");

    for (int i = 0; i < nSeq; i++) {
      Pointer<Float> embd;
      try {
        // Get embeddings for the sequence
        embd = lib.llama_get_embeddings_seq(ctx, i);

        if (embd == nullptr) {
          throw Exception("Failed to get sequence embeddings");
        }
      } catch (e) {
        print(
            "Warning: Failed to get sequence embeddings for sequence $i, trying fallback: $e");

        // Fallback: Try to get embeddings for the first token of the sequence
        embd = nullptr;
        for (int j = 0; j < batch.n_tokens; j++) {
          if (batch.seq_id[j][0] == i && batch.logits[j] != 0) {
            try {
              embd = lib.llama_get_embeddings_ith(ctx, j);
              if (embd != nullptr) {
                print("  Found embeddings using token $j");
                break;
              }
            } catch (e) {
              print("  Failed to get embeddings for token $j: $e");
            }
          }
        }

        // Second fallback: Try to get all embeddings
        if (embd == nullptr) {
          try {
            embd = lib.llama_get_embeddings(ctx);
            if (embd != nullptr) {
              print(
                  "  Using default embeddings (this may not be sequence-specific)");
            }
          } catch (e) {
            print("  Failed to get default embeddings: $e");
          }
        }

        if (embd == nullptr) {
          throw Exception(
              "All attempts to get embeddings for sequence $i failed");
        }
      }

      // Copy to output buffer with normalization if needed
      Pointer<Float> out = Pointer<Float>.fromAddress(
          output.address + i * nEmbd * sizeOf<Float>());

      if (embdNormalize != 0) {
        double sum = 0.0;
        for (int j = 0; j < nEmbd; j++) {
          sum += embd[j] * embd[j];
        }
        sum = sqrt(sum);
        for (int j = 0; j < nEmbd; j++) {
          out[j] = sum > 0.0 ? embd[j] / sum : 0.0;
        }
      } else {
        for (int j = 0; j < nEmbd; j++) {
          out[j] = embd[j];
        }
      }
    }
  } else {
    // For token-level embeddings (uncommon for BERT models, common for decoder-only)
    print("Using token-level embeddings");

    int outputIdx = 0;
    for (int i = 0; i < batch.n_tokens; i++) {
      if (batch.logits[i] == 0) {
        continue; // Skip tokens that don't have logits
      }

      Pointer<Float> embd;
      try {
        embd = lib.llama_get_embeddings_ith(ctx, i);
      } catch (e) {
        print("Warning: Failed to get embeddings for token $i: $e");
        continue;
      }

      if (embd == nullptr) {
        print("Warning: No embeddings for token $i, skipping");
        continue;
      }

      Pointer<Float> out = Pointer<Float>.fromAddress(
          output.address + outputIdx * nEmbd * sizeOf<Float>());
      outputIdx++;

      if (embdNormalize != 0) {
        double sum = 0.0;
        for (int j = 0; j < nEmbd; j++) {
          sum += embd[j] * embd[j];
        }
        sum = sqrt(sum);
        for (int j = 0; j < nEmbd; j++) {
          out[j] = sum > 0.0 ? embd[j] / sum : 0.0;
        }
      } else {
        for (int j = 0; j < nEmbd; j++) {
          out[j] = embd[j];
        }
      }
    }
  }
}

// Calculate cosine similarity between two embedding vectors
double embdSimilarityCos(Pointer<Float> a, Pointer<Float> b, int n) {
  double dot = 0.0;
  double normA = 0.0;
  double normB = 0.0;

  for (int i = 0; i < n; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  // Avoid division by zero
  if (normA <= 0.0 || normB <= 0.0) {
    return 0.0;
  }

  return dot / (sqrt(normA) * sqrt(normB));
}

void main(List<String> args) {
  // Common params
  String modelPath =
      "/Users/adel/Workspace/gguf/multilingual-e5-large-instruct-q8_0.gguf";
  String prompt = "hello, world!";
  String embdOut = ""; // Default is empty, could be "json", "json+", "array"
  int embdNormalize = 1; // Default to normalize
  String embdSep = "\n"; // Default separator for multiple prompts
  int nBatch = 512; // Default batch size

  /*
  // Parse command-line arguments
  if (args.isEmpty) {
    print("Usage: dart embedding.dart <model_path> [options]");
    print("Options:");
    print("  --prompt <text>        Input prompt (default: empty)");
    print(
        "  --embd-out <format>    Output format: json, json+, array (default: text)");
    print("  --embd-normalize       Normalize embeddings (default)");
    print("  --embd-no-normalize    Don't normalize embeddings");
    print(
        "  --embd-sep <separator> Separator for multiple prompts (default: newline)");
    print("  -b <n>                 Batch size (default: 512)");
    print("  --verbose-prompt       Show prompt tokenization details");
    exit(1);
  }

  modelPath = args[0];

  // Process other arguments
  for (int i = 1; i < args.length; i++) {
    if (args[i] == "--prompt" && i + 1 < args.length) {
      prompt = args[++i];
    } else if (args[i] == "--embd-out" && i + 1 < args.length) {
      embdOut = args[++i];
    } else if (args[i] == "--embd-normalize") {
      embdNormalize = 1;
    } else if (args[i] == "--embd-no-normalize") {
      embdNormalize = 0;
    } else if (args[i] == "--embd-sep" && i + 1 < args.length) {
      embdSep = args[++i];
    } else if (args[i] == "-b" && i + 1 < args.length) {
      nBatch = int.parse(args[++i]);
    } else if (args[i] == "--verbose-prompt") {
      verbosePrompt = true;
    }
  }
  */

  // Initialize llama
  llama_cpp lib;

  // Determine library path
  String? libraryPath =
      "/Users/adel/Workspace/llama_cpp_dart/bin/MAC_ARM64/libllama.dylib";

  try {
    print("Loading llama.cpp library from: $libraryPath");
    lib = llama_cpp(DynamicLibrary.open(libraryPath));
  } catch (e) {
    print("Failed to load llama.cpp library: $e");
    print("Trying process library as fallback");
    try {
      lib = llama_cpp(DynamicLibrary.process());
    } catch (e) {
      print("Failed to load library from process: $e");
      exit(1);
    }
  }

  lib.llama_backend_init();

  // Load the model
  print("Loading model: $modelPath");

  // Create model params
  final modelParams = calloc<llama_model_params>();
  modelParams.ref = lib.llama_model_default_params();

  final modelPathPtr = modelPath.toNativeUtf8().cast<Char>();
  final model = lib.llama_load_model_from_file(modelPathPtr, modelParams.ref);
  malloc.free(modelPathPtr);

  if (model == nullptr) {
    print("Unable to load model");
    lib.llama_backend_free();
    calloc.free(modelParams);
    exit(1);
  }

  // Create context params with embedding flag
  final contextParams = calloc<llama_context_params>();
  contextParams.ref = lib.llama_context_default_params();
  contextParams.ref.embeddings = true; // Enable embeddings
  contextParams.ref.n_batch = nBatch;
  contextParams.ref.n_ubatch =
      nBatch; // For non-causal models, batch size must equal ubatch size

  final ctx = lib.llama_new_context_with_model(model, contextParams.ref);
  if (ctx == nullptr) {
    print("Unable to create context");
    lib.llama_free_model(model);
    lib.llama_backend_free();
    calloc.free(modelParams);
    calloc.free(contextParams);
    exit(1);
  }

  final vocab = lib.llama_model_get_vocab(model);
  final nCtxTrain = lib.llama_model_n_ctx_train(model);
  final nCtx = lib.llama_n_ctx(ctx);

  // Determine pooling type
  llama_pooling_type poolingType;
  try {
    poolingType = lib.llama_pooling_type1(ctx); // Use the correct function name
    print("Model pooling type: $poolingType");
  } catch (e) {
    // This would happen if the function is not available or fails
    // For BERT models, MEAN (1) is often a good default
    print("Could not determine pooling type, using MEAN as default: $e");
    poolingType = llama_pooling_type.LLAMA_POOLING_TYPE_MEAN;
  }

  // Check model compatibility
  if (lib.llama_model_has_encoder(model) == true &&
      lib.llama_model_has_decoder(model) == true) {
    print(
        "Warning: Computing embeddings in encoder-decoder models may not be fully supported");
  }

  if (nCtx > nCtxTrain) {
    print(
        "Warning: model was trained on only $nCtxTrain context tokens ($nCtx specified)");
  }

  // Print system information
  print("\nSystem information: Dart llama embedding");

  // Split the prompt into lines
  List<String> prompts = splitLines(prompt, separator: embdSep);

  // Tokenize the prompts and trim
  List<List<int>> inputs = [];
  for (String prompt in prompts) {
    if (prompt.isEmpty) {
      print("Warning: Empty prompt detected, skipping");
      continue;
    }

    final promptPtr = prompt.toNativeUtf8().cast<Char>();

    // First get the token count
    int tokenCount = -lib.llama_tokenize(
        vocab, promptPtr, prompt.length, nullptr, 0, true, true);

    if (tokenCount <= 0) {
      print("Warning: Failed to tokenize prompt: '$prompt'");
      malloc.free(promptPtr);
      continue;
    }

    if (tokenCount > nBatch) {
      print(
          "Number of tokens in input line ($tokenCount) exceeds batch size ($nBatch), increase batch size and re-run");
      malloc.free(promptPtr);
      lib.llama_free(ctx);
      lib.llama_free_model(model);
      lib.llama_backend_free();
      calloc.free(modelParams);
      calloc.free(contextParams);
      exit(1);
    }

    final tokensPtr = calloc<llama_token>(tokenCount);
    int actualTokens = lib.llama_tokenize(
        vocab, promptPtr, prompt.length, tokensPtr, tokenCount, true, true);

    if (actualTokens <= 0) {
      print(
          "Warning: Tokenization returned $actualTokens tokens for prompt: '$prompt'");
      malloc.free(promptPtr);
      calloc.free(tokensPtr);
      continue;
    }

    // Convert to Dart list
    List<int> tokens = List<int>.generate(actualTokens, (i) => tokensPtr[i]);
    inputs.add(tokens);

    malloc.free(promptPtr);
    calloc.free(tokensPtr);
  }

  if (inputs.isEmpty) {
    print("Error: No valid inputs after tokenization");
    lib.llama_free(ctx);
    lib.llama_free_model(model);
    lib.llama_backend_free();
    calloc.free(modelParams);
    calloc.free(contextParams);
    exit(1);
  }

  // Check if the last token is SEP
  for (List<int> inp in inputs) {
    if (inp.isEmpty) {
      continue;
    }

    int sepToken = lib.llama_vocab_sep(vocab);
    if (inp.last != sepToken) {
      print(
          "Warning: last token in the prompt is not SEP (${inp.last} != $sepToken)");
      print(
          "Warning: 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header");
    }
  }

  // Initialize batch
  final nPrompts = inputs.length;
  final batch = lib.llama_batch_init(nBatch, 0, 1);

  // Count number of embeddings
  int nEmbdCount = 0;
  if (poolingType == llama_pooling_type.LLAMA_POOLING_TYPE_NONE) {
    for (int k = 0; k < nPrompts; k++) {
      nEmbdCount += inputs[k].length;
    }
  } else {
    nEmbdCount = nPrompts;
  }

  // Allocate output
  final nEmbd = lib.llama_model_n_embd(model);
  print("Embedding dimension: $nEmbd");
  print("Total embedding count: $nEmbdCount");

  final embeddings = calloc<Float>(nEmbdCount * nEmbd);

  // Break into batches
  int e = 0; // number of embeddings already stored
  int s = 0; // number of prompts in current batch

  // Clear the batch to start - using our custom implementation instead of lib.llama_batch_clear()
  llamaBatchClear(batch);

  for (int k = 0; k < nPrompts; k++) {
    List<int> inp = inputs[k];
    int nToks = inp.length;

    // Encode if at capacity
    if (batch.n_tokens + nToks > nBatch) {
      print(
          "Processing batch with ${batch.n_tokens} tokens from $s prompts...");
      Pointer<Float> out = Pointer<Float>.fromAddress(
          embeddings.address + e * nEmbd * sizeOf<Float>());
      batchDecode(lib, ctx, batch, out, s, nEmbd, embdNormalize);
      e += poolingType == llama_pooling_type.LLAMA_POOLING_TYPE_NONE
          ? batch.n_tokens
          : s;
      s = 0;
      // Use our custom function instead of lib.llama_batch_clear()
      llamaBatchClear(batch);
    }

    // Add to batch
    batchAddSeq(batch, inp, s);
    s += 1;
  }

  // Final batch
  if (batch.n_tokens > 0) {
    print(
        "Processing final batch with ${batch.n_tokens} tokens from $s prompts...");
    Pointer<Float> out = Pointer<Float>.fromAddress(
        embeddings.address + e * nEmbd * sizeOf<Float>());
    batchDecode(lib, ctx, batch, out, s, nEmbd, embdNormalize);
  }

  // Output embeddings based on format
  if (embdOut.isEmpty) {
    print("");

    if (poolingType == llama_pooling_type.LLAMA_POOLING_TYPE_NONE) {
      for (int j = 0; j < nEmbdCount; j++) {
        StringBuffer sb = StringBuffer("embedding $j: ");
        for (int i = 0; i < min(3, nEmbd); i++) {
          if (embdNormalize == 0) {
            sb.write("${embeddings[j * nEmbd + i].round()} ");
          } else {
            sb.write("${embeddings[j * nEmbd + i].toStringAsFixed(6)} ");
          }
        }
        sb.write(" ... ");
        for (int i = nEmbd - 3; i < nEmbd; i++) {
          if (embdNormalize == 0) {
            sb.write("${embeddings[j * nEmbd + i].round()} ");
          } else {
            sb.write("${embeddings[j * nEmbd + i].toStringAsFixed(6)} ");
          }
        }
        print(sb.toString());
      }
    } else if (poolingType == llama_pooling_type.LLAMA_POOLING_TYPE_RANK) {
      for (int j = 0; j < nEmbdCount; j++) {
        print("rerank score $j: ${embeddings[j * nEmbd].toStringAsFixed(3)}");
      }
    } else {
      // Print the first part of the embeddings or for a single prompt, the full embedding
      for (int j = 0; j < nPrompts; j++) {
        StringBuffer sb = StringBuffer("embedding $j: ");
        for (int i = 0; i < (nPrompts > 1 ? min(16, nEmbd) : nEmbd); i++) {
          if (embdNormalize == 0) {
            sb.write("${embeddings[j * nEmbd + i].round()} ");
          } else {
            sb.write("${embeddings[j * nEmbd + i].toStringAsFixed(6)} ");
          }
        }
        print(sb.toString());
      }

      // Print cosine similarity matrix
      if (nPrompts > 1) {
        print("\nCosine similarity matrix:\n");

        // Header row
        StringBuffer header = StringBuffer("      ");
        for (int i = 0; i < nPrompts; i++) {
          String shortPrompt = prompts[i].length > 6
              ? prompts[i].substring(0, 6)
              : prompts[i].padRight(6);
          header.write("$shortPrompt ");
        }
        print(header.toString());

        // Matrix
        for (int i = 0; i < nPrompts; i++) {
          StringBuffer row = StringBuffer();
          for (int j = 0; j < nPrompts; j++) {
            double sim = embdSimilarityCos(
                Pointer<Float>.fromAddress(
                    embeddings.address + i * nEmbd * sizeOf<Float>()),
                Pointer<Float>.fromAddress(
                    embeddings.address + j * nEmbd * sizeOf<Float>()),
                nEmbd);
            row.write("${sim.toStringAsFixed(2)} ");
          }

          String shortPrompt =
              prompts[i].length > 10 ? prompts[i].substring(0, 10) : prompts[i];
          row.write(shortPrompt);
          print(row.toString());
        }
      }
    }
  } else if (embdOut == "json" || embdOut == "json+" || embdOut == "array") {
    final bool notArray = embdOut != "array";

    print(notArray ? '{\n  "object": "list",\n  "data": [' : "[");

    for (int j = 0; j < nEmbdCount; j++) {
      if (notArray) {
        print(
            '    {\n      "object": "embedding",\n      "index": $j,\n      "embedding": ');
      }

      StringBuffer embeddingValues = StringBuffer("[");
      for (int i = 0; i < nEmbd; i++) {
        embeddingValues.write(embdNormalize != 0
            ? embeddings[j * nEmbd + i].toStringAsFixed(7)
            : embeddings[j * nEmbd + i].toStringAsFixed(0));

        if (i < nEmbd - 1) {
          embeddingValues.write(",");
        }
      }
      embeddingValues.write("]");

      print(notArray
          ? '${embeddingValues.toString()}\n    }'
          : embeddingValues.toString());

      if (j < nEmbdCount - 1) {
        print(notArray ? ",\n" : ",");
      }
    }

    print(notArray ? "\n  ]" : "]");

    if (embdOut == "json+" && nPrompts > 1) {
      print(',\n  "cosineSimilarity": [');

      for (int i = 0; i < nEmbdCount; i++) {
        StringBuffer row = StringBuffer("    [");
        for (int j = 0; j < nEmbdCount; j++) {
          double sim = embdSimilarityCos(
              Pointer<Float>.fromAddress(
                  embeddings.address + i * nEmbd * sizeOf<Float>()),
              Pointer<Float>.fromAddress(
                  embeddings.address + j * nEmbd * sizeOf<Float>()),
              nEmbd);
          row.write(sim.toStringAsFixed(2));
          if (j < nEmbdCount - 1) {
            row.write(", ");
          }
        }
        row.write(" ]");
        if (i < nEmbdCount - 1) {
          print("${row.toString()},");
        } else {
          print(row.toString());
        }
      }

      print("\n  ]");
    }

    if (notArray) {
      print("\n}");
    }
  }

  print("\nPerformance stats:");
  lib.llama_perf_context_print(ctx);

  // Clean up
  for (int i = 0; i < batch.n_tokens; i++) {
    if (batch.seq_id[i] != nullptr) {
      calloc.free(batch.seq_id[i]);
    }
  }
  lib.llama_batch_free(batch);
  lib.llama_free(ctx);
  lib.llama_free_model(model);
  lib.llama_backend_free();
  calloc.free(embeddings);
  calloc.free(contextParams);
  calloc.free(modelParams);
}
