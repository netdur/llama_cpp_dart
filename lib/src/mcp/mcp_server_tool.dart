import 'dart:async';

/// An MCP tool. Implementations must return **content parts** per MCP:
/// e.g. [{'type':'text','text':'hello'}]
abstract class MCPServerTool {
  String get name;
  String get description;

  /// JSON Schema for the tool's input
  Map<String, dynamic> get inputSchema;

  /// Execute the tool and return a list of MCP content parts.
  Future<List<Map<String, dynamic>>> call(Map<String, dynamic> args);
}