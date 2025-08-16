// ignore_for_file: avoid_print

import 'dart:convert';
import 'dart:io';
import 'package:intl/intl.dart';
import 'package:http/http.dart' as http;
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// --- Example tools (same as before) ---

class CurrentTimeTool implements MCPServerTool {
  @override
  String get name => 'getCurrentTime';
  @override
  String get description => 'Get the current local time of the server';
  @override
  Map<String, dynamic> get inputSchema => {
        'type': 'object',
        'properties': {},
        'additionalProperties': false,
      };
  @override
  Future<List<Map<String, dynamic>>> call(Map<String, dynamic> args) async {
    final time = DateFormat('HH:mm').format(DateTime.now());
    return [
      {'type': 'text', 'text': 'The current time is $time.'}
    ];
  }
}

class WeatherTool implements MCPServerTool {
  WeatherTool({http.Client? client}) : _client = client ?? http.Client();
  final http.Client _client;
  @override
  String get name => 'getWeather';
  @override
  String get description => 'Get the weather for a specific city';
  @override
  Map<String, dynamic> get inputSchema => {
        'type': 'object',
        'properties': {
          'city': {'type': 'string', 'description': 'Name of the city'}
        },
        'required': ['city'],
        'additionalProperties': false,
      };
  @override
  Future<List<Map<String, dynamic>>> call(Map<String, dynamic> args) async {
    final city = (args['city'] ?? '').toString().trim();
    if (city.isEmpty) throw ArgumentError('Missing required argument: city');
    try {
      final url = Uri.parse('https://wttr.in/$city?format=%C,+%t');
      final res = await _client.get(url).timeout(const Duration(seconds: 5));
      if (res.statusCode == 200) {
        final body = res.body.trim();
        return [
          {'type': 'text', 'text': 'The weather in $city is $body.'}
        ];
      }
      throw Exception('HTTP ${res.statusCode}');
    } catch (_) {
      return [
        {
          'type': 'text',
          'text': 'The weather in $city is sunny, 22°C. (fallback)'
        }
      ];
    }
  }
}

/// --- TCP accept loop that reuses MCPServer._handle() ---

Future<void> main() async {
  final serverCore = MCPServer()
    ..add(CurrentTimeTool())
    ..add(WeatherTool());

  final srv = await ServerSocket.bind(InternetAddress.loopbackIPv4, 4242);
  stderr.writeln(
      '[Server] TCP MCP listening on ${srv.address.address}:${srv.port}');

  await for (final client in srv) {
    stderr.writeln(
        '[Server] ✅ Client ${client.remoteAddress.address}:${client.remotePort}');
    final buffer = StringBuffer();

    client.listen((data) async {
      try {
        final text = utf8.decode(data);
        buffer.write(text);

        var content = buffer.toString();
        while (true) {
          final nl = content.indexOf('\n');
          if (nl < 0) break;

          final line = content.substring(0, nl).trim();
          content = content.substring(nl + 1);

          if (line.isEmpty) continue;
          Map<String, dynamic> resp;
          try {
            final req = jsonDecode(line) as Map<String, dynamic>;
            // Ensure jsonrpc presence sanity (not strictly required to respond)
            if (req['jsonrpc'] != '2.0' && req['id'] != null) {
              resp = {
                'jsonrpc': '2.0',
                'id': req['id'],
                'error': {
                  'code': -32600,
                  'message': 'Invalid Request: jsonrpc must be "2.0"'
                }
              };
            } else {
              resp = await serverCore.handleForTcp(req);
            }
          } catch (e) {
            // No id known → generic error
            resp = {
              'jsonrpc': '2.0',
              'id': null,
              'error': {'code': -32600, 'message': 'Invalid JSON'}
            };
          }

          final out = '${jsonEncode(resp)}\n';
          client.add(utf8.encode(out));
          await client.flush();
        }

        // keep leftover partial
        buffer
          ..clear()
          ..write(content);
      } catch (e) {
        stderr.writeln('[Server] read error: $e');
      }
    }, onDone: () {
      stderr.writeln('[Server] client disconnected');
      client.close();
    }, onError: (e) {
      stderr.writeln('[Server] client error: $e');
    });
  }
}
