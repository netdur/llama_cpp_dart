import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'mcp_server_tool.dart';

void _log(String msg) => stderr.writeln(msg);

Map<String, dynamic> _ok(dynamic id, Map<String, dynamic> result) => {
      'jsonrpc': '2.0',
      'id': id,
      'result': result,
    };

Map<String, dynamic> _err(dynamic id, int code, String message,
        [Object? data]) =>
    {
      'jsonrpc': '2.0',
      'id': id,
      'error': {
        'code': code,
        'message': message,
        if (data != null) 'data': data,
      },
    };

class MCPServer {
  final Map<String, MCPServerTool> _tools = {};

  void add(MCPServerTool tool) {
    _tools[tool.name] = tool;
  }

  /// STDIO server (optional)
  Future<void> listenStdio() async {
    _log('[Server] MCP (stdio) starting…');

    await stdin
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .forEach((line) async {
      final raw = line.trim();
      if (raw.isEmpty) return;

      Map<String, dynamic>? req;
      try {
        final decoded = jsonDecode(raw);
        if (decoded is Map<String, dynamic>) {
          req = decoded;
        } else {
          _send(_err(null, -32600, 'Invalid Request: not an object'));
          return;
        }
      } catch (_) {
        _send(_err(null, -32600, 'Invalid Request: malformed JSON'));
        return;
      }

      try {
        final resp = await _handle(req);
        _send(resp);
      } catch (e) {
        _send(_err(req['id'], -32603, 'Internal error'));
      }
    });

    _log('[Server] STDIN closed; exiting.');
  }

  void _send(Map<String, dynamic> msg) {
    stdout.writeln(jsonEncode(msg));
  }

  /// Public handler you can call from any transport (TCP, stdio, etc.)
  /// This is what your TCP server should call.
  Future<Map<String, dynamic>> handleForTcp(Map<String, dynamic> req) async {
    final resp = await _handle(req);
    if (resp['jsonrpc'] != '2.0') resp['jsonrpc'] = '2.0';
    return resp;
  }

  /// Core JSON-RPC/MCP router.
  Future<Map<String, dynamic>> _handle(Map<String, dynamic> req) async {
    final method = req['method'];
    final id = req['id'];
    final isNotification = !req.containsKey('id');

    if (method is! String) {
      if (isNotification) return {};
      return _err(id, -32600, 'Invalid Request: "method" must be a string');
    }

    switch (method) {
      case 'initialize':
        if (id == null) {
          return _err(null, -32600, 'Invalid Request: id required');
        }
        return _ok(id, {
          'protocolVersion': '2025-03-26',
          'serverInfo': {'name': 'dart-mcp-server', 'version': '0.2.0'},
          'capabilities': {
            'tools': {'listChanged': false},
          },
        });

      case 'notifications/initialized':
        return {};

      case 'tools/list':
        if (id == null) {
          return _err(null, -32600, 'Invalid Request: id required');
        }
        return _ok(id, {
          'tools': _tools.values
              .map((t) => {
                    'name': t.name,
                    'description': t.description,
                    'inputSchema': t.inputSchema,
                  })
              .toList(),
        });

      case 'tools/call':
        if (id == null) {
          return _err(null, -32600, 'Invalid Request: id required');
        }

        if (req['params'] != null && req['params'] is! Map) {
          return _err(id, -32602, 'Invalid params: "params" must be an object');
        }
        final params =
            (req['params'] as Map?)?.cast<String, dynamic>() ?? const {};
        final toolName = params['name'];
        if (toolName is! String || toolName.isEmpty) {
          return _err(
              id, -32602, 'Invalid params: "name" (string) is required');
        }

        final args = (params['arguments'] is Map)
            ? (params['arguments'] as Map).cast<String, dynamic>()
            : const <String, dynamic>{};

        final tool = _tools[toolName];
        if (tool == null) {
          return _err(id, -32602, 'Unknown tool: $toolName');
        }

        try {
          final contentParts = await tool.call(args);
          return _ok(id, {
            'content': contentParts,
            'isError': false,
          });
        } catch (e) {
          return _ok(id, {
            'content': [
              {'type': 'text', 'text': 'Tool "$toolName" failed: $e'}
            ],
            'isError': true,
          });
        }

      default:
        if (isNotification) return {};
        return _err(id, -32601, 'Method not found: $method');
    }
  }
}
