/// Magic-substring chat templates that llama.cpp's
/// `llama_chat_apply_template` pattern-matcher picks up.
///
/// llama.cpp's matcher is not a Jinja parser — it scans the template string
/// for distinctive substrings and routes to a built-in renderer for that
/// family. Any string containing the right substring is enough to select
/// the family. Models with custom Jinja that the matcher can't classify
/// (e.g. Unsloth Gemma quants) ship with a Jinja blob the matcher fails on;
/// pass one of these constants as `templateOverride` to bypass that.
///
/// All constants here are valid input to
/// `llama_chat_apply_template(template_str, messages, n, add_ass, ...)`.
/// llama.cpp will render messages in the family's canonical format.
final class KnownChatTemplates {
  KnownChatTemplates._();

  /// Google Gemma family (gemma-1, gemma-2, gemma-3, gemma-4).
  /// Renders `<start_of_turn>{role}\n{content}<end_of_turn>` with `system`
  /// folded into the next user turn (Gemma has no native system role).
  static const String gemma = '<start_of_turn>';

  /// ChatML — used by Qwen, OpenChat, many fine-tunes.
  /// Renders `<|im_start|>{role}\n{content}<|im_end|>`.
  static const String chatml = '<|im_start|>';

  /// Meta Llama 3 / 3.1 / 3.2.
  /// Renders `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>`.
  static const String llama3 = '<|start_header_id|>';

  /// Mistral instruct (v1).
  /// Renders `[INST] {content} [/INST]`.
  static const String mistral = '[INST]';

  /// Microsoft Phi-3 family.
  /// Renders `<|user|>\n{content}<|end|>\n<|assistant|>`.
  static const String phi3 = '<|user|>';

  /// DeepSeek instruct family.
  /// Renders `### Instruction:` / `### Response:` blocks.
  static const String deepseek = '### Instruction:\n<|EOT|>';

  /// Cohere Command-R.
  /// Renders `<|START_OF_TURN_TOKEN|><|USER_TOKEN|>...`.
  static const String commandR = '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>';

  /// Vicuna v1.1 (without explicit system prompt slot).
  static const String vicuna = 'USER: \nASSISTANT: ';

  /// Falcon-3.
  static const String falcon3 = '<|im_start|>assistant';
}
