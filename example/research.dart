// ignore_for_file: avoid_print

import 'dart:core';
import 'dart:io';
import 'dart:math';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

import 'notes.dart';

String generateLlmResponseSync(
    Llama llama, String userPrompt, ChatFormat format) {
  final history = ChatHistory()
    ..addMessage(role: Role.user, content: userPrompt)
    ..addMessage(role: Role.assistant, content: "");

  final formattedPrompt =
      history.exportFormat(format, leaveLastAssistantOpen: true);

  llama.setPrompt(formattedPrompt);
  final buffer = StringBuffer();
  while (true) {
    var (token, done) = llama.getNext();
    buffer.write(token);
    if (done) break;
  }

  return buffer.toString().trim();
}

List<String> chunkNoteContent(String content, int maxLength) {
  if (content.length <= maxLength) {
    return [content];
  }

  List<String> chunks = [];
  int start = 0;
  while (start < content.length) {
    int end = min(start + maxLength, content.length);
    int sentenceEnd = content.lastIndexOf(RegExp(r'[.!?]\s'), end);
    if (sentenceEnd > start && sentenceEnd < end) {
      end = sentenceEnd + 1;
    } else {
      int spaceEnd = content.lastIndexOf(' ', end);
      if (spaceEnd > start && spaceEnd < end) {
        end = spaceEnd + 1;
      }
    }
    chunks.add(content.substring(start, end).trim());
    start = end;
  }
  return chunks;
}

String cleanLlmArtifacts(String text) {
  return text
      .replaceAll(
          RegExp(r'^(e>|e|elling>|_label>|eller>|\*+)\s*',
              caseSensitive: false),
          '')
      .trim();
}

void main() {
  List<String> researchTopics = [
    "Information about the Quadratic Formula",
    "Details on the function of Mitochondria",
    "Time management techniques involving focused work and breaks",
    "The historical context and details of the 1944 Allied invasion of Normandy"
  ];

  Llama? llama;
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf";

    final modelParams = ModelParams();
    final contextParams = ContextParams()
      ..nPredict = -1
      ..nCtx = 1024 * 4;
    final samplerParams = SamplerParams()
      ..temp = 0.3
      ..topK = 40
      ..topP = 0.9;

    const chatFormat = ChatFormat.gemini;
    const int maxChunkLength = 500;

    llama = Llama(modelPath, modelParams, contextParams, samplerParams);

    for (String topic in researchTopics) {
      print("\n--------------------------------------------------");
      print("Topic: \"$topic\"");

      List<String> findings = [];
      stdout.write("Scanning:\n");

      // 1. Iterate through Notes for Information Gathering
      for (int i = 0; i < notes.length; i++) {
        Note currentNote = notes[i];
        bool foundInNote = false;

        List<String> chunks =
            chunkNoteContent(currentNote.content, maxChunkLength);

        // 3. Extract Related Information (per Chunk)
        for (int chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
          String chunk = chunks[chunkIndex];
          // print("   -> Analyzing chunk ${chunkIndex + 1}/${chunks.length}"); // Removed

          // Clear context BEFORE each extraction call
          llama.clear();

          String extractionPrompt = """
Context Text:
\"\"\"
$chunk
\"\"\"

Task: Carefully read the Context Text above. Extract any specific facts, statements, or details that are directly relevant to the topic: "$topic".
List only the relevant information concisely.
If no part of the text is relevant to the topic, respond with the single word: NONE
""";

          // Run extraction prompt through LLM (now quiet)
          String extractionResult =
              generateLlmResponseSync(llama, extractionPrompt, chatFormat);

          String cleanedResult = extractionResult.trim().toUpperCase();
          bool isNone = cleanedResult == "NONE" ||
              (cleanedResult.contains("NONE") &&
                  cleanedResult.endsWith("NONE"));

          // --- Removed detailed extraction result prints ---
          // print("   Extraction Result Raw: \"$extractionResult\"");
          // print("   Extraction Result Cleaned Upper: \"$cleanedResult\"");
          // print("   Is considered NONE? $isNone");

          if (!isNone && extractionResult.trim().isNotEmpty) {
            String actualFinding = cleanLlmArtifacts(extractionResult);
            if (actualFinding.isNotEmpty) {
              String finding =
                  "Source: \"${currentNote.title}\" - Finding: $actualFinding";
              findings.add(finding);
              foundInNote = true; // Mark that this note contributed
              // print("   >>> Relevant Finding Added! (Cleaned: \"$actualFinding\")"); // Removed
            } // else {
            // print("   (Finding was empty after cleaning artifacts)"); // Removed
            // }
          } // else {
          // print("   (No relevant information found or identified as NONE)"); // Removed
          // }
        } // End chunk loop

        // Print progress indicator for the note AFTER processing all its chunks
        stdout.write("\"${currentNote.title}\" ${foundInNote ? '✅' : '❌'}\n");
      } // End notes loop

      print(''); // Finish the "Scanning..." line with a newline

      // 5. Generate and Print Synthesis Report
      // print("\n--------------------------------------------------"); // Removed
      // print("Finished Scanning Notes for Topic: \"$topic\""); // Removed
      // print("--------------------------------------------------"); // Removed

      if (findings.isEmpty) {
        print(
            "Report: No relevant information found in the notes for this topic.");
      } else {
        // print("\nCollected Findings (${findings.length}):"); // Removed
        // for (var f in findings) { print("- $f"); } // Removed

        // Clear context BEFORE the final synthesis call
        llama.clear();

        String synthesisPrompt = """
Research Topic: "$topic"

Collected Findings from Notes:
\"\"\"
${findings.join('\n\n')}
\"\"\"

Task: Based *only* on the 'Collected Findings' provided above, write a brief research report summarizing the information related to the 'Research Topic'.
Synthesize the findings into a coherent paragraph or two. Do not add any information not present in the findings. Start the report directly without conversational filler.
""";

        // print("\n--- Generating Final Synthesis Report ---"); // Removed
        String finalReportRaw =
            generateLlmResponseSync(llama, synthesisPrompt, chatFormat);
        String finalReportClean = cleanLlmArtifacts(finalReportRaw);

        // Print the final report
        print(finalReportClean);
      }
      print(
          "--------------------------------------------------\n"); // Separator after report
    } // End topics loop
  } catch (e, s) {
    print("\n################# ERROR #################");
    print("An error occurred: ${e.toString()}");
    print("Stack trace:\n$s");
    print("#########################################");
  } finally {
    if (llama != null) {
      llama.dispose();
    } else {
      print("LLM was not initialized, no resources to dispose.");
    }
  }
}
