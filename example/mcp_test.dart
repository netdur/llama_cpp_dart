// ignore_for_file: avoid_print

import 'dart:io';
import 'mcp_agent.dart';

void main() async {
  // Setup agent
  final agent = SimpleAgent(
    modelPath: '/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf',
    libraryPath: 'bin/MAC_ARM64/libllama.dylib',
    debugMode: false,
    streamingMode: true,  // Enable streaming by default
  );
  
  await agent.connect();
  
  // Enable line mode for better input handling
  stdin.echoMode = true;
  stdin.lineMode = true;
  
  print('💬 Chat ready! (Commands: "exit", "debug on/off", "stream on/off")\n');
  
  while (true) {
    stdout.write('You: ');
    
    // Read input with proper terminal support
    final input = stdin.readLineSync();
    if (input == null || input.isEmpty) continue;
    
    // Handle commands
    final command = input.trim().toLowerCase();
    
    if (command == 'exit') break;
    
    if (command == 'debug on') {
      agent.debugMode = true;
      agent.streamingMode = false;  // Auto-disable streaming in debug mode
      print('Debug mode enabled (streaming auto-disabled)\n');
      continue;
    }
    
    if (command == 'debug off') {
      agent.debugMode = false;
      agent.streamingMode = true;  // Auto-enable streaming when debug is off
      print('Debug mode disabled (streaming auto-enabled)\n');
      continue;
    }
    
    if (command == 'stream on') {
      agent.streamingMode = true;
      print('Streaming enabled\n');
      continue;
    }
    
    if (command == 'stream off') {
      agent.streamingMode = false;
      print('Streaming disabled\n');
      continue;
    }
    
    // Process chat
    // Only print "AI: " prefix if not streaming (streaming handles its own prefix)
    if (!agent.streamingMode) {
      stdout.write('AI: ');
    }
    
    final response = await agent.chat(input);
    
    // Only print response if not streaming (streaming already printed it)
    if (!agent.streamingMode) {
      print('$response\n');
    } else {
      print('');  // Extra line for spacing after streamed response
    }
  }
  
  agent.dispose();
  print('\nGoodbye! 👋');
}