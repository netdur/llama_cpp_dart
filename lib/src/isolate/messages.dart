import 'dart:isolate';

import '../chat/chat_message.dart';
import '../context/context_params.dart';
import '../generation/context_shift.dart';
import '../generation/event.dart';
import '../model/model_params.dart';
import '../multimodal/media.dart';
import '../multimodal/multimodal_params.dart';
import '../sampling/sampler_params.dart';

/// Bootstrap payload sent to a worker isolate at spawn.
///
/// Carries everything the worker needs to load the library, model, and
/// context, plus the [SendPort] it uses to talk back to the main isolate.
final class EngineBootstrap {
  final SendPort replyPort;

  /// Path to the llama dylib. Ignored when [useProcessSymbols] is true.
  final String libraryPath;

  /// When true, the worker resolves symbols from the running process
  /// (iOS / macOS apps that statically linked the xcframework) instead
  /// of opening [libraryPath].
  final bool useProcessSymbols;

  final String? backendDirectory;
  final ModelParams modelParams;
  final ContextParams contextParams;
  final MultimodalParams? multimodalParams;

  const EngineBootstrap({
    required this.replyPort,
    required this.libraryPath,
    required this.modelParams,
    required this.contextParams,
    this.useProcessSymbols = false,
    this.backendDirectory,
    this.multimodalParams,
  });
}

/// Commands sent from the main isolate to the worker.
sealed class EngineCommand {
  /// Request id used to correlate responses.
  final int requestId;
  const EngineCommand(this.requestId);
}

final class CreateSessionCommand extends EngineCommand {
  final int sessionId;
  final int seqId;
  const CreateSessionCommand(super.requestId,
      {required this.sessionId, required this.seqId});
}

final class DisposeSessionCommand extends EngineCommand {
  final int sessionId;
  const DisposeSessionCommand(super.requestId, {required this.sessionId});
}

final class AppendTextCommand extends EngineCommand {
  final int sessionId;
  final String text;
  final bool addSpecial;
  final bool parseSpecial;
  const AppendTextCommand(
    super.requestId, {
    required this.sessionId,
    required this.text,
    required this.addSpecial,
    required this.parseSpecial,
  });
}

final class ClearSessionCommand extends EngineCommand {
  final int sessionId;
  const ClearSessionCommand(super.requestId, {required this.sessionId});
}

final class GenerateCommand extends EngineCommand {
  final int sessionId;
  final String? prompt;
  final bool addSpecial;
  final bool parseSpecial;
  final SamplerParams sampler;
  final int maxTokens;

  /// Optional media attached to this request. When non-empty the worker
  /// routes through `mtmd_helper_eval_chunks` and the prompt must contain
  /// one media marker per item.
  final List<LlamaMedia> media;

  /// Context-shift policy for this request. See [ContextShiftPolicy].
  final ContextShiftPolicy shiftPolicy;

  /// Tunables for [ContextShiftPolicy.auto].
  final ContextShift shift;

  const GenerateCommand(
    super.requestId, {
    required this.sessionId,
    required this.prompt,
    required this.addSpecial,
    required this.parseSpecial,
    required this.sampler,
    required this.maxTokens,
    this.media = const <LlamaMedia>[],
    this.shiftPolicy = ContextShiftPolicy.off,
    this.shift = ContextShift.defaults,
  });
}

/// Render [messages] via the model's chat template (or [templateOverride])
/// and run the resulting prompt through the session's generator.
///
/// The session's KV cache is cleared before rendering so the full
/// conversation is re-prefilled from scratch each turn. Incremental KV
/// reuse across turns is a future optimization.
final class GenerateChatCommand extends EngineCommand {
  final int sessionId;
  final List<ChatMessage> messages;
  final SamplerParams sampler;
  final int maxTokens;
  final String? templateOverride;
  const GenerateChatCommand(
    super.requestId, {
    required this.sessionId,
    required this.messages,
    required this.sampler,
    required this.maxTokens,
    this.templateOverride,
  });
}

final class CancelCommand extends EngineCommand {
  /// Id of the in-flight generate request to cancel.
  final int targetRequestId;
  const CancelCommand(super.requestId, {required this.targetRequestId});
}

final class ShutdownCommand extends EngineCommand {
  const ShutdownCommand(super.requestId);
}

/// Save the session's KV state and token history to [path].
///
/// The worker writes a single self-describing file containing wrapper
/// metadata (model identity, mmproj identity, codec version, etc.),
/// the token history, and the raw `llama_state_seq_get_data` blob.
final class SaveSessionStateCommand extends EngineCommand {
  final int sessionId;
  final String path;
  final Map<String, Object?> extra;
  const SaveSessionStateCommand(
    super.requestId, {
    required this.sessionId,
    required this.path,
    required this.extra,
  });
}

/// Load a previously saved state file into the session. The worker
/// validates that the metadata is compatible with the current engine.
final class LoadSessionStateCommand extends EngineCommand {
  final int sessionId;
  final String path;
  const LoadSessionStateCommand(
    super.requestId, {
    required this.sessionId,
    required this.path,
  });
}

/// Returned for [LoadSessionStateCommand]; carries the wrapper-supplied
/// `extra` map (e.g. chat history) so the caller can rehydrate it.
final class SessionStateLoadedResponse extends EngineResponse {
  final Map<String, Object?> extra;
  final int tokensCount;
  final int kvHead;
  const SessionStateLoadedResponse(
    super.requestId, {
    required this.extra,
    required this.tokensCount,
    required this.kvHead,
  });
}

/// Responses sent from the worker back to the main isolate.
sealed class EngineResponse {
  /// Matches [EngineCommand.requestId] of the originating command.
  final int requestId;
  const EngineResponse(this.requestId);
}

/// Sent once after spawn when model + context are loaded successfully.
final class EngineReadyResponse extends EngineResponse {
  /// Send-port the main isolate uses to issue commands.
  final SendPort commandPort;

  /// Default chat template embedded in the model's GGUF metadata, or `null`
  /// if the model has no template. When `null`, [EngineChat] requires an
  /// explicit `templateOverride`.
  final String? modelChatTemplate;

  /// True if a multimodal projector was loaded successfully.
  final bool multimodalLoaded;

  /// True if the loaded projector supports image input.
  final bool supportsVision;

  /// True if the loaded projector supports audio input.
  final bool supportsAudio;

  /// Audio sample rate the model expects (e.g. 16000), or -1 if audio is
  /// not supported.
  final int audioSampleRate;

  /// Mirrors `llama_memory_can_shift`. False on recurrent / iSWA caches
  /// (e.g. some Qwen3, Gemma 3 with sliding-window attention). When false,
  /// `ContextShiftPolicy.auto` is rejected at generate-time.
  final bool canShift;

  const EngineReadyResponse(
    super.requestId,
    this.commandPort, {
    this.modelChatTemplate,
    this.multimodalLoaded = false,
    this.supportsVision = false,
    this.supportsAudio = false,
    this.audioSampleRate = -1,
    this.canShift = true,
  });
}

final class EngineErrorResponse extends EngineResponse {
  final String message;
  const EngineErrorResponse(super.requestId, this.message);
}

final class EngineAckResponse extends EngineResponse {
  const EngineAckResponse(super.requestId);
}

final class EngineGenerationEvent extends EngineResponse {
  final GenerationEvent event;
  const EngineGenerationEvent(super.requestId, this.event);
}

final class EngineGenerationCancelled extends EngineResponse {
  const EngineGenerationCancelled(super.requestId);
}

/// Final response of a successful generate stream — the user-facing stream
/// is closed after this is received.
final class EngineGenerationFinished extends EngineResponse {
  const EngineGenerationFinished(super.requestId);
}

final class EngineShutdownComplete extends EngineResponse {
  const EngineShutdownComplete(super.requestId);
}
