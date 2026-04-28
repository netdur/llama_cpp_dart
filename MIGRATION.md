# Migrating from 0.2.x to 0.9.0

The 0.9.0 line is a clean rewrite. There is no shim layer; every entry
point has a different name and most have a different shape. This guide
maps the common patterns so you can port code by skimming.

## Mental model

| 0.2.x | 0.9.0 |
|---|---|
| `Llama` god-class | `LlamaModel` + `LlamaContext` + `LlamaSession` + `Sampler` (split into focused modules; rarely used directly) |
| `LlamaParent` / `LlamaChild` | `LlamaEngine` (one isolate, async stream API) |
| `LlamaService` | dropped (single-session); spawn multiple `LlamaEngine`s if you really need parallel models |
| `IsolateScope` | dropped (subscription-cancel does the same job via stream cancellation) |
| `ChatHistory` + format classes | `ChatMessage` + `EngineChat` + `KnownChatTemplates` |
| Auto-trim flag on context | `ContextShiftPolicy.auto` per generation request |
| `LlamaImage` | `LlamaMedia` (image **and** audio) |
| Manual `loadModel` / `setPrompt` / `getNext` loop | `await for (final ev in session.generate(...))` |
| MCP server / client | dropped — application layer |
| `TextChunker` | dropped — application layer |

## Loading a model

Before:

```dart
Llama.libraryPath = '/path/to/libllama.dylib';
final llama = Llama(
  '/path/to/model.gguf',
  modelParams: ModelParams()..nGpuLayers = 99,
  contextParams: ContextParams()..nCtx = 4096,
  samplerParams: SamplerParams()..temp = 0.7,
);
```

After:

```dart
final engine = await LlamaEngine.spawn(
  libraryPath: '/path/to/libllama.dylib',
  modelParams: ModelParams(path: '/path/to/model.gguf', gpuLayers: 99),
  contextParams: const ContextParams(nCtx: 4096),
);
final session = await engine.createSession();
```

Note: `SamplerParams` has moved to per-`generate` argument — the same
engine can run different sampling configs across calls.

## Generating tokens

Before:

```dart
llama.setPrompt('Hello');
String full = '';
while (true) {
  final (text, done) = llama.getNext();
  full += text;
  if (done) break;
}
```

After:

```dart
final buf = StringBuffer();
await for (final event in session.generate(
  prompt: 'Hello',
  addSpecial: true,
  sampler: const SamplerParams(temperature: 0.7),
  maxTokens: 256,
)) {
  if (event is TokenEvent) buf.write(event.text);
}
```

`generate` is a stream — break out, `take(N)`, listen-with-cancel, etc.
Stream cancellation propagates to the worker (via `StreamController.onCancel`
sending a `CancelCommand`).

## Chat

Before:

```dart
final history = ChatHistory();
history.addMessage(role: Role.system, content: 'You are concise.');
history.addMessage(role: Role.user, content: 'Hi');
final formatted = history.exportFormat(ChatFormat.chatml);
llama.setPrompt(formatted);
// drain getNext loop
```

After:

```dart
final chat = await engine.createChat();
chat.addSystem('You are concise.');
chat.addUser('Hi');
await for (final ev in chat.generate(maxTokens: 64)) {
  if (ev is TokenEvent) stdout.write(ev.text);
}
// chat.messages now holds the full history including the assistant reply
```

The chat template comes from the GGUF metadata via
`llama_chat_apply_template`. For models with custom Jinja the C API can't
parse, pass `templateOverride: KnownChatTemplates.gemma` (or `chatml`,
`llama3`, ...) — these are sentinel substrings llama.cpp's matcher uses
to pick a built-in renderer. If even that fails, render the prompt
yourself and use `EngineSession.generate(prompt:)` directly.

## Vision

Before:

```dart
final llama = Llama(..., mmprojPath: 'mmproj.gguf');
final image = LlamaImage.fromFile('cat.jpg');
llama.setVisionPrompt(image, 'Describe.');
```

After:

```dart
final engine = await LlamaEngine.spawn(
  libraryPath: '/path/to/libllama.dylib',
  modelParams: ModelParams(path: 'llm.gguf', gpuLayers: 99),
  contextParams: const ContextParams(nCtx: 4096),
  multimodalParams: const MultimodalParams(mmprojPath: 'mmproj.gguf'),
);
final chat = await engine.createChat();
chat.addUser('Describe.', media: [LlamaMedia.imageFile('cat.jpg')]);
await for (final ev in chat.generate(maxTokens: 128)) {
  if (ev is TokenEvent) stdout.write(ev.text);
}
```

Audio works the same way — `LlamaMedia.audioFile('clip.mp3')` — when the
projector reports `engine.supportsAudio == true`. Internally mtmd
auto-detects the format.

## Persistence

Before:

```dart
final blob = llama.saveState();
File('state.bin').writeAsBytesSync(blob);
// ... later
llama.loadState(File('state.bin').readAsBytesSync());
```

After:

```dart
await session.saveState('/path/to/state.lcdc');
// ... later, possibly after an engine restart:
await otherSession.loadState('/path/to/state.lcdc');
```

The new format includes metadata (model identity, context params,
mmproj fingerprint, token checksum). Loading into an incompatible
engine throws `LlamaStateException` with a discriminator
(`modelMismatch`, `contextTooSmall`, `multimodalMismatch`, ...).

## Auto-trim / context shift

Before (0.2.2+):

```dart
llama.setContextParams(ContextParams()..autoTrim = true);
```

After:

```dart
session.generate(
  prompt: '...',
  shiftPolicy: ContextShiftPolicy.auto,
  shift: const ContextShift(nKeep: -1, keepBos: true),
);
```

Per-generate now, opt-in. Check `engine.canShift` first — Qwen3 SWA and
similar caches don't support shifting.

## Multi-session

`LlamaService` is gone. If you need multiple conversations, create
multiple `EngineSession`s — each has its own `seqId` in the same
context. Bump `ContextParams.nSeqMax` to allow > 1 concurrent sequences.

```dart
final s1 = await engine.createSession(seqId: 0);
final s2 = await engine.createSession(seqId: 1);
```

Generate calls serialize on the worker — one in-flight at a time. If
you need genuine parallel decode, spawn a second `LlamaEngine` (own
isolate, own model load).

## Things that no longer exist

- `LlamaProcessor`, `LlamaSlot`, `IsolateScope` — replaced by `EngineSession`.
- `ChatFormat` enum + per-format classes — handled by the GGUF template.
- `MCPClient`, `MCPServer`, `MCPServerTool` — agent / tool wiring is your
  app's job, not the binding's.
- `TextChunker` — RAG helper, use a separate package or inline code.
- `Llama.libraryPath` static — replaced by per-engine `libraryPath` arg
  on `LlamaEngine.spawn` (or `loadFromProcess` on iOS / app builds).
- `Llama.verbose` flag — log control comes via `LlamaLog.silence()` and
  is off-by-default in the worker isolate (see CHANGELOG).
