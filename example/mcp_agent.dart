// ignore_for_file: avoid_print

import 'dart:convert';
import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// AI agent with MCP tool support
class SimpleAgent {
  late final Llama llama;
  late final MCPClient mcp;
  final ChatHistory history = ChatHistory();
  bool debugMode;
  bool streamingMode;
  
  SimpleAgent({
    required String modelPath, 
    String? libraryPath,
    this.debugMode = false,
    this.streamingMode = true,  // Enable streaming by default
  }) {
    if (libraryPath != null) Llama.libraryPath = libraryPath;
    
    llama = Llama(
      modelPath,
      modelParams: ModelParams()..nGpuLayers = 99,
      contextParams: ContextParams()
        ..nPredict = -1
        ..nCtx = 8192
        ..nBatch = 8192,
      samplerParams: SamplerParams()
        ..temp = 0.7
        ..topK = 40
        ..topP = 0.95
        ..penaltyRepeat = 1.1,
      verbose: false,
    );
    
    mcp = MCPClient();
  }
  
  Future<void> connect() async {
    await mcp.connect();
    print('✅ Ready with ${mcp.tools.length} tools');
  }
  
  /// Main chat method
  Future<String> chat(String message) async {
    _debugPrint('\n╔══════════ DEBUG ══════════');
    _debugPrint('║ Input: "$message"');
    
    // Router decides if tools are needed
    final toolDecision = await _routeDecision(message);
    
    String response;
    
    if (toolDecision != null && toolDecision['tool'] != null) {
      // Tool path
      response = await _handleToolPath(message, toolDecision);
    } else {
      // Direct response path
      _debugPrint('║ Router Decision: DIRECT RESPONSE');
      response = await _generateDirectResponse(message);
    }
    
    _debugPrint('╚═══════════════════════════\n');
    return response;
  }
  
  /// Handle tool-based responses
  Future<String> _handleToolPath(String message, Map<String, dynamic> toolDecision) async {
    final toolName = toolDecision['tool'] as String;
    final toolArgs = (toolDecision['args'] as Map?)?.cast<String, dynamic>() ?? {};
    
    _debugPrint('║ Router Decision: USE TOOL');
    _debugPrint('║ Tool: $toolName');
    _debugPrint('║ Args: ${jsonEncode(toolArgs)}');
    
    // Call tool
    String toolResult;
    try {
      toolResult = await mcp.callTool(toolName, toolArgs);
      _debugPrint('║ Tool Result: "$toolResult"');
    } catch (e) {
      toolResult = 'Error calling tool: $e';
      _debugPrint('║ Tool Error: $e');
    }
    
    // Generate response with tool result
    return await _generateToolResponse(message, toolResult);
  }
  
  /// Router: Decides if tools are needed
  Future<Map<String, dynamic>?> _routeDecision(String message) async {
    final toolsDesc = mcp.tools.map((t) {
      final params = (t.inputSchema['properties'] as Map?)?.entries
          .map((e) => '${e.key}: ${e.value['type']}')
          .join(', ') ?? '';
      return '- ${t.name}($params): ${t.description}';
    }).join('\n');
    
    final routerPrompt = '''You are a ROUTER. Your ONLY job is to decide if a tool is needed.

Available tools:
$toolsDesc

User message: "$message"

OUTPUT ONLY JSON:
- If tool needed: {"tool": "toolName", "args": {"param": "value"}}
- If no tool needed: {"tool": null}

JSON:''';
    
    _debugPrint('║ Router analyzing...');
    
    // Setup prompt for router
    final tempHistory = ChatHistory()
      ..addMessage(role: Role.user, content: routerPrompt)
      ..addMessage(role: Role.assistant, content: "");
    
    llama.setPrompt(
      tempHistory.exportFormat(ChatFormat.gemini, leaveLastAssistantOpen: true)
    );
    
    // Generate decision
    String decision = '';
    int tokenCount = 0;
    const maxTokens = 100;
    
    await for (final token in llama.generateText()) {
      decision += token;
      tokenCount++;
      if (tokenCount >= maxTokens) break;
    }
    
    _debugPrint('║ Router Output: ${decision.replaceAll('\n', ' ').trim()}');
    
    // Parse JSON decision
    return _parseJson(decision);
  }
  
  /// Parse JSON from potentially messy LLM output
  Map<String, dynamic>? _parseJson(String text) {
    try {
      String jsonStr = text.trim();
      
      // Handle markdown code blocks
      if (jsonStr.contains('```')) {
        final match = RegExp(r'```(?:json)?\s*([^`]+)\s*```').firstMatch(jsonStr);
        if (match != null) jsonStr = match.group(1)!.trim();
      }
      
      // Extract JSON object
      final jsonMatch = RegExp(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}').firstMatch(jsonStr);
      if (jsonMatch != null) jsonStr = jsonMatch.group(0)!;
      
      // Clean and parse
      jsonStr = jsonStr
          .replaceAll(RegExp(r'[\r\n]+'), ' ')
          .replaceAll(RegExp(r'\s+'), ' ')
          .trim();
      
      final parsed = jsonDecode(jsonStr) as Map<String, dynamic>;
      return parsed.containsKey('tool') ? parsed : null;
      
    } catch (e) {
      _debugPrint('║ JSON Parse Error: $e');
      return {'tool': null};
    }
  }
  
  /// Generate response for tool-based queries
  Future<String> _generateToolResponse(String message, String toolResult) async {
    history.addMessage(role: Role.user, content: message);
    
    final prompt = '''User asked: "$message"

You called a tool and got this result: $toolResult

Provide a helpful, natural response:''';
    
    _debugPrint('║ Generating response with tool result...');
    _debugPrint('║ Generation prompt:\n║   ${prompt.replaceAll('\n', '\n║   ')}');
    
    final tempHistory = ChatHistory()
      ..addMessage(role: Role.user, content: prompt)
      ..addMessage(role: Role.assistant, content: "");
    
    llama.setPrompt(
      tempHistory.exportFormat(ChatFormat.gemini, leaveLastAssistantOpen: true)
    );
    
    final response = await _generateTextWithStreaming();
    history.addMessage(role: Role.assistant, content: response);
    return response;
  }
  
  /// Generate direct response (no tools needed)
  Future<String> _generateDirectResponse(String message) async {
    history.addMessage(role: Role.user, content: message);
    
    final prompt = '''You are a helpful AI assistant. Have a natural conversation with the user.

User: $message
Assistant:''';
    
    _debugPrint('║ Generating direct response...');
    
    final tempHistory = ChatHistory()
      ..addMessage(role: Role.user, content: prompt)
      ..addMessage(role: Role.assistant, content: "");
    
    llama.setPrompt(
      tempHistory.exportFormat(ChatFormat.gemini, leaveLastAssistantOpen: true)
    );
    
    final response = await _generateTextWithStreaming();
    history.addMessage(role: Role.assistant, content: response);
    return response;
  }
  
  /// Generate text from LLM with optional streaming
  Future<String> _generateTextWithStreaming() async {
    String response = '';
    bool firstToken = true;
    
    await for (final token in llama.generateText()) {
      response += token;

      if (streamingMode && !debugMode) {
        if (firstToken) {
          stdout.write('AI: ');
          firstToken = false;
        }
        stdout.write(token);
      }
    }
    
    // Add newline after streaming
    if (streamingMode && !debugMode) {
      print('');
    }
    
    return response.trim();
  }
  
  /// Debug print helper
  void _debugPrint(String message) {
    if (debugMode) print(message);
  }
  
  void dispose() {
    llama.dispose();
    mcp.dispose();
  }
}
