// ignore_for_file: avoid_print

import 'dart:convert';
import 'dart:io';
import 'dart:async';
import 'dart:typed_data';

/// Clean MCP client that handles all the protocol stuff
class MCPClient {
  late final Socket _socket;
  final StringBuffer _buffer = StringBuffer();
  late StreamSubscription<Uint8List> _subscription;
  final _completers = <int, Completer<Map<String, dynamic>>>{};
  int _requestId = 0;
  
  final List<Tool> tools = [];
  
  /// Connect and initialize
  Future<void> connect({
    String host = '127.0.0.1',
    int port = 4242,
  }) async {
    _socket = await Socket.connect(host, port);
    
    _subscription = _socket.listen(
      _handleData,
      onError: (e) => print('Socket error: $e'),
      onDone: () => print('Socket closed'),
    );
    
    // Initialize protocol
    await _call('initialize', {
      'protocolVersion': '2024-11-05',
      'clientInfo': {'name': 'dart-mcp-client', 'version': '1.0.0'},
    });
    
    await _notify('notifications/initialized', {});
    
    // Load tools
    final resp = await _call('tools/list', {});
    final list = resp['result']['tools'] as List;
    
    tools.clear();
    for (final t in list) {
      tools.add(Tool(
        name: t['name'],
        description: t['description'],
        inputSchema: t['inputSchema'],
      ));
    }
  }
  
  /// Call a tool by name
  Future<String> callTool(String name, [Map<String, dynamic>? args]) async {
    final result = await _call('tools/call', {
      'name': name,
      'arguments': args ?? {},
    });
    
    final content = result['result']['content'] as List;
    return content
        .where((c) => c['type'] == 'text')
        .map((c) => c['text'])
        .join(' ');
  }
  
  /// Find tool by keyword
  Tool? findTool(String keyword) {
    final lower = keyword.toLowerCase();
    return tools.firstWhere(
      (t) => t.name.toLowerCase().contains(lower) ||
             t.description.toLowerCase().contains(lower),
      orElse: () => Tool(name: '', description: '', inputSchema: {}),
    ).name.isNotEmpty ? tools.firstWhere((t) => 
      t.name.toLowerCase().contains(lower) ||
      t.description.toLowerCase().contains(lower)) : null;
  }
  
  void dispose() {
    _subscription.cancel();
    _socket.close();
  }
  
  // --- Private boring stuff ---
  
  void _handleData(Uint8List data) {
    _buffer.write(utf8.decode(data));
    
    String content = _buffer.toString();
    while (true) {
      final nl = content.indexOf('\n');
      if (nl < 0) break;
      
      final line = content.substring(0, nl).trim();
      content = content.substring(nl + 1);
      
      if (line.isEmpty) continue;
      
      try {
        final resp = jsonDecode(line) as Map<String, dynamic>;
        final id = resp['id'];
        if (id != null && _completers.containsKey(id)) {
          _completers[id]!.complete(resp);
          _completers.remove(id);
        }
      } catch (_) {}
    }
    
    _buffer.clear();
    _buffer.write(content);
  }
  
  Future<Map<String, dynamic>> _call(String method, Map<String, dynamic> params) async {
    final id = ++_requestId;
    final completer = Completer<Map<String, dynamic>>();
    _completers[id] = completer;
    
    _socket.add(utf8.encode('${jsonEncode({
      'jsonrpc': '2.0',
      'id': id,
      'method': method,
      'params': params,
    })}\n'));
    
    return await completer.future.timeout(const Duration(seconds: 10));
  }
  
  Future<void> _notify(String method, Map<String, dynamic> params) async {
    _socket.add(utf8.encode('${jsonEncode({
      'jsonrpc': '2.0',
      'method': method,
      'params': params,
    })}\n'));
  }
}

class Tool {
  final String name;
  final String description;
  final Map<String, dynamic> inputSchema;
  
  Tool({required this.name, required this.description, required this.inputSchema});
}