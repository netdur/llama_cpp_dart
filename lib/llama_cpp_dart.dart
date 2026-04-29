/// Dart FFI binding for llama.cpp targeting iOS, Android, and macOS.
///
/// Public API surface for the v1.0 rewrite. See `plan.md` for the full
/// architecture.
library;

export 'src/batch/batch.dart' show LlamaBatch;
export 'src/chat/chat_message.dart' show ChatMessage;
export 'src/chat/chat_template.dart' show ChatTemplate, ChatTemplateException;
export 'src/chat/known_templates.dart' show KnownChatTemplates;
export 'src/context/context.dart' show LlamaContext;
export 'src/context/context_params.dart' show ContextParams, FlashAttention;
export 'src/ffi/backends.dart'
    show BackendDevice, BackendDeviceType, LlamaBackends;
export 'src/ffi/library_loader.dart' show LlamaLibrary;
export 'src/ffi/log.dart' show LlamaLog;
export 'src/generation/context_shift.dart'
    show ContextShift, ContextShiftPolicy;
export 'src/generation/event.dart'
    show DoneEvent, GenerationEvent, ShiftEvent, TokenEvent;
export 'src/generation/generator.dart' show Generator;
export 'src/generation/request.dart' show Request;
export 'src/generation/stop.dart'
    show StopReason, StopEog, StopMaxTokens, StopUserAbort;
export 'src/isolate/engine.dart' show EngineChat, EngineSession, LlamaEngine;
export 'src/model/model.dart' show LlamaModel;
export 'src/model/model_params.dart' show ModelParams;
export 'src/model/vocab.dart' show LlamaVocab;
export 'src/multimodal/media.dart' show LlamaMedia, MediaKind;
export 'src/multimodal/multimodal_context.dart'
    show MultimodalContext, MultimodalException;
export 'src/multimodal/multimodal_params.dart' show MultimodalParams;
export 'src/sampling/sampler.dart' show Sampler;
export 'src/sampling/sampler_factory.dart' show SamplerFactory;
export 'src/sampling/sampler_params.dart' show SamplerParams, defaultSeed;
export 'src/session/session.dart' show LlamaSession;
export 'src/session/state_codec.dart'
    show LlamaStateError, LlamaStateException, StateMetadata, stateCodecVersion;
export 'src/tokenizer/tokenizer.dart' show Tokenizer;
export 'src/tokenizer/utf8_accumulator.dart' show Utf8Accumulator;
export 'src/types/exceptions.dart';
