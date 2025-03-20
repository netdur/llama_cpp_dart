// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:system_info2/system_info2.dart';

void main() async {
  try {
    final cores = SysInfo.cores;
    // int memory = SysInfo.getTotalVirtualMemory() ~/ megaByte;

    SamplerParams samplerParams = SamplerParams();
    ModelParams modelParams = ModelParams();

    const size = 512 * 4;
    ContextParams contextParams = ContextParams();
    contextParams.nThreadsBatch = cores.length;
    contextParams.nThreads = cores.length;
    contextParams.nCtx = size;
    contextParams.nBatch = size;
    contextParams.nUbatch = size;
    // contextParams.nPredit = 512;

    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Downloads/gemma-3-12b-it-Q4_K_M.gguf";

    Llama llama = Llama(modelPath, modelParams, contextParams, samplerParams);

    llama.setPrompt("$system \n $prompt");
    while (true) {
      var (token, done) = llama.getNext();
      stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");

    llama.dispose();
  } catch (e) {
    print("Error: ${e.toString()}");
  }
}

const int megaByte = 1024 * 1024;

const String prompt =
    '''You are the strategic advisor to a mid-sized technology company facing a critical decision. The company has developed a groundbreaking AI-powered medical diagnosis tool that shows promising results in early testing. However, they're now at a crossroads:
Option A: Partner with a large healthcare corporation that's offering \$50 million for exclusive rights to the technology. This corporation has a vast distribution network and established relationships with hospitals worldwide.
Option B: Maintain independence and seek \$30 million in venture capital funding to develop and market the product themselves, retaining full control and potentially higher long-term profits.
Additional context:

The company currently has 18 months of runway remaining
The development team consists of 15 AI specialists who are highly sought after by competitors
Early clinical trials show 92% accuracy, compared to 89% for existing solutions
The healthcare market is becoming increasingly competitive, with three similar products expected to launch within 24 months
Regulatory approval will take approximately 12-15 months
Current market size is \$2.5 billion with projected 15% annual growth

Using the chain-of-thought framework, analyze this situation and recommend the best strategic path forward. Consider both short-term stability and long-term potential, regulatory risks, market dynamics, team retention, and the broader implications for healthcare accessibility.
''';

const String system =
    '''When approaching this problem, please walk through your reasoning process explicitly, breaking down your thoughts into clear stages. Consider multiple angles and potential implications before reaching your conclusion. Specifically:

Start by restating the key elements of the problem in your own words, identifying the core questions or challenges that need to be addressed.
Generate initial observations about the situation, listing any relevant facts, constraints, or important contextual details that could influence your analysis.
For each major decision point or analytical step:

State your current understanding
List the assumptions you're making
Explore potential alternatives
Explain why you're choosing one path over others
Describe how this choice affects subsequent reasoning


When encountering uncertain elements:

Acknowledge the uncertainty explicitly
Describe different possible scenarios
Explain how each scenario would impact your reasoning
Detail why you're proceeding with particular assumptions


As you progress toward your conclusion:

Periodically summarize your key findings
Check whether your logic remains consistent
Identify any potential gaps or weaknesses in your reasoning
Consider counter-arguments to your approach


Before stating your final conclusion:

Review your complete chain of reasoning
Verify that each step logically follows from the previous ones
Ensure you haven't overlooked any critical factors
Consider whether your conclusion adequately addresses the original problem


In your final response:

Clearly state your conclusion
Summarize the key steps that led to this conclusion
Highlight any remaining uncertainties or areas that might benefit from further analysis
Explain any limitations in your reasoning process



Please make your thinking process as transparent as possible, explaining not just what you conclude but why you conclude it. Include any relevant calculations, logical steps, or decision points that influenced your thinking. If you find yourself making assumptions, state them explicitly and explain why they're reasonable in this context.
''';
