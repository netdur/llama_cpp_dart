# Using llama_cpp_dart in flutter

## Classes

### Generation Options
The purpose of this class is to combine the various parameters available to the the user into a single class 
so it can be passed to the generation class more easily. 

Alternatively, you could opt to forgo using a `GenerationOptions` class and instead use the `ModelParams`, 
`ContextParams` and `SamplingParams` classes directly. The only difference in this case would be that you 
would have to pass 3 different parameter objects into your prompt class along with the prompt and 
potentially a callback function. This would ammount to 5 parameters for 1 function which is considered 
bad practice. 

For my own app (Maid) I chose to use a `GenerationOptions` class for the above reasons, but also because 
Maid has to support multiple API's each with their own parameter classes. 

Below is the `GenerationOptions` class for maid:

```dart
class GenerationOptions {
  late List<Map<String, dynamic>> _messages;
  late String? _remoteUrl;
  late PromptFormatType _promptFormat;
  late ApiType _apiType;
  late String? _apiKey;
  late String? _remoteModel;
  late String? _path;
  late String _description;
  late String _personality;
  late String _scenario;
  late String _system;
  late int _nKeep;
  late int _seed;
  late int _nPredict;
  late int _topK;
  late double _topP;
  late double _minP;
  late double _tfsZ;
  late double _typicalP;
  late int _penaltyLastN;
  late double _temperature;
  late double _penaltyRepeat;
  late double _penaltyPresent;
  late double _penaltyFreq;
  late int _mirostat;
  late double _mirostatTau;
  late double _mirostatEta;
  late bool _penalizeNewline;
  late int _nCtx;
  late int _nBatch;
  late int _nThread;

  List<Map<String, dynamic>> get messages => _messages;
  String? get remoteUrl => _remoteUrl;
  PromptFormatType get promptFormat => _promptFormat;
  ApiType get apiType => _apiType;
  String? get apiKey => _apiKey;
  String? get remoteModel => _remoteModel;
  String? get path => _path;
  String get description => _description;
  String get personality => _personality;
  String get scenario => _scenario;
  String get system => _system;
  int get nKeep => _nKeep;
  int get seed => _seed;
  int get nPredict => _nPredict;
  int get topK => _topK;
  double get topP => _topP;
  double get minP => _minP;
  double get tfsZ => _tfsZ;
  double get typicalP => _typicalP;
  int get penaltyLastN => _penaltyLastN;
  double get temperature => _temperature;
  double get penaltyRepeat => _penaltyRepeat;
  double get penaltyPresent => _penaltyPresent;
  double get penaltyFreq => _penaltyFreq;
  int get mirostat => _mirostat;
  double get mirostatTau => _mirostatTau;
  double get mirostatEta => _mirostatEta;
  bool get penalizeNewline => _penalizeNewline;
  int get nCtx => _nCtx;
  int get nBatch => _nBatch;
  int get nThread => _nThread;

  Map<String, dynamic> toMap() {
    Map<String, dynamic> map = {};
    map["messages"] = _messages;
    map["remote_url"] = _remoteUrl;
    map["prompt_format"] = _promptFormat.index;
    map["api_type"] = _apiType.index;
    map["api_key"] = _apiKey;
    map["remote_model"] = _remoteModel;
    map["path"] = _path;
    map["description"] = _description;
    map["personality"] = _personality;
    map["scenario"] = _scenario;
    map["system"] = _system;
    map["n_keep"] = _nKeep;
    map["seed"] = _seed;
    map["n_predict"] = _nPredict;
    map["top_k"] = _topK;
    map["top_p"] = _topP;
    map["min_p"] = _minP;
    map["tfs_z"] = _tfsZ;
    map["typical_p"] = _typicalP;
    map["penalty_last_n"] = _penaltyLastN;
    map["temperature"] = _temperature;
    map["penalty_repeat"] = _penaltyRepeat;
    map["penalty_present"] = _penaltyPresent;
    map["penalty_freq"] = _penaltyFreq;
    map["mirostat"] = _mirostat;
    map["mirostat_tau"] = _mirostatTau;
    map["mirostat_eta"] = _mirostatEta;
    map["penalize_nl"] = _penalizeNewline;
    map["n_ctx"] = _nCtx;
    map["n_batch"] = _nBatch;
    map["n_threads"] = _nThread;
    return map;
  }

  GenerationOptions({
    required Model model,
    required Character character,
    required Session session,
  }) {
    try {
      Logger.log(model.toMap().toString());
      Logger.log(character.toMap().toString());
      Logger.log(session.toMap().toString());

      _messages = [];
      if (character.useExamples) {
        _messages.addAll(character.examples);
        _messages.addAll(session.getMessages());
      }

      _remoteUrl = model.parameters["remote_url"];
      _promptFormat = model.format;
      _apiType = model.apiType;
      _apiKey = model.parameters["api_key"];
      _remoteModel = model.parameters["remote_model"];
      _path = model.parameters["path"];
      _description = character.description;
      _personality = character.personality;
      _scenario = character.scenario;
      _system = character.system;
      _nKeep = model.parameters["n_keep"];
      _seed = model.parameters["random_seed"] ? Random().nextInt(1000000) : model.parameters["seed"];
      _nPredict = model.parameters["n_predict"];
      _topK = model.parameters["top_k"];
      _topP = model.parameters["top_p"];
      _minP = model.parameters["min_p"];
      _tfsZ = model.parameters["tfs_z"];
      _typicalP = model.parameters["typical_p"];
      _penaltyLastN = model.parameters["penalty_last_n"];
      _temperature = model.parameters["temperature"];
      _penaltyRepeat = model.parameters["penalty_repeat"];
      _penaltyPresent = model.parameters["penalty_present"];
      _penaltyFreq = model.parameters["penalty_freq"];
      _mirostat = model.parameters["mirostat"];
      _mirostatTau = model.parameters["mirostat_tau"];
      _mirostatEta = model.parameters["mirostat_eta"];
      _penalizeNewline = model.parameters["penalize_nl"];
      _nCtx = model.parameters["n_ctx"];
      _nBatch = model.parameters["n_batch"];
      _nThread = model.parameters["n_threads"];
    } catch (e) {
      Logger.log(e.toString());
    }
  }
}

```

As you can see this class is pretty comprehensive, but as previously stated, in the context of Maid it has to 
support multiple API's. If you want your app to only support llama_cpp_dart your GenerationOptions class could 
be much more simple, the general principle stays the same.

For example:

```dart
class GenerationOptions {
  late ModelParams modelParams;
  late ContextParams contextParams;
  late SamplingParams samplingParams;

  GenrationOptions({
    required this.modelParams, 
    required this.contextParams, 
    required this.samplingParams
  });
}
```

### Generation Class
The purpose of this class is to actually call the llama_cpp_dart functions to generate text. This class can be static 
and only requires 2 public functions (`prompt` and `stop`). The general process for the `prompt` function is to process 
the data from your `GenerationOptions` class into the `ModelParams`, `ContextParams` and `SamplingParams` classes.

You can then use these classes along with the path to your chosen model to create a `LlamaProcessor` class. At this time if 
you have any previous messages of the conversation you should structure and pass them into the `LlamaProcessor`. Finally 
you can start listening to the `LlamaProcessor` stream and then call the prompt function.

The stop function should simply be a proxie for the `LlamaProcessor` stop function.

Below is the Class used for LocalGeneration in Maid

```dart
class LocalGeneration {
  static LlamaProcessor? _llamaProcessor;
  static Completer? _completer;
  static Timer? _timer;
  static DateTime? _startTime;

  static void prompt(
    String input,
    GenerationOptions options,
    void Function(String?) callback
  ) async {
    _timer = null;
    _startTime = null;
    _completer = Completer();

    ModelParams modelParams = ModelParams();
    modelParams.format = options.promptFormat;
    ContextParams contextParams = ContextParams();
    contextParams.batch = options.nBatch;
    contextParams.context = options.nCtx;
    contextParams.seed = options.seed;
    contextParams.threads = options.nThread;
    contextParams.threadsBatch = options.nThread;
    SamplingParams samplingParams = SamplingParams();
    samplingParams.temp = options.temperature;
    samplingParams.topK = options.topK;
    samplingParams.topP = options.topP;
    samplingParams.tfsZ = options.tfsZ;
    samplingParams.typicalP = options.typicalP;
    samplingParams.penaltyLastN = options.penaltyLastN;
    samplingParams.penaltyRepeat = options.penaltyRepeat;
    samplingParams.penaltyFreq = options.penaltyFreq;
    samplingParams.penaltyPresent = options.penaltyPresent;
    samplingParams.mirostat = options.mirostat;
    samplingParams.mirostatTau = options.mirostatTau;
    samplingParams.mirostatEta = options.mirostatEta;
    samplingParams.penalizeNl = options.penalizeNewline;
    
    _llamaProcessor = LlamaProcessor(
      options.path!, 
      modelParams, 
      contextParams,
      samplingParams
    );

    List<Map<String, dynamic>> messages = [
      {
        'role': 'system',
        'content': '''
          ${options.description}\n\n
          ${options.personality}\n\n
          ${options.scenario}\n\n
          ${options.system}\n\n
        '''
      }
    ];

    for (var message in options.messages) {
      switch (message['role']) {
        case "user":
          messages.add(message);
          break;
        case "assistant":
          messages.add(message);
          break;
        case "system": // Under normal circumstances, this should never be called
          messages.add(message);
          break;
        default:
          break;
      }

      messages.add({
        'role': 'system',
        'content': options.system
      });
    }

    _llamaProcessor!.messages = options.messages;

    _llamaProcessor!.stream.listen((data) {
      _resetTimer();
      callback.call(data);
    });

    _llamaProcessor?.prompt(input);
    await _completer?.future;
    callback.call(null);
    _llamaProcessor?.unloadModel();
    _llamaProcessor = null;
    Logger.log('Local generation completed');
  }

  static void _resetTimer() {
    _timer?.cancel();
    if (_startTime != null) {
      final elapsed = DateTime.now().difference(_startTime!);
      _startTime = DateTime.now();
      _timer = Timer(elapsed * 10, stop);
    } else {
      _startTime = DateTime.now();
      _timer = Timer(const Duration(seconds: 5), stop);
    }
  }

  static void stop() {
    _timer?.cancel();
    _llamaProcessor?.stop();
    _completer?.complete();
    Logger.log('Local generation stopped');
  }
}
```