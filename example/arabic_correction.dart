// ignore_for_file: avoid_print

import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

String prompt = """
in the object 7, the sentence should end with gender of newborn
however duo to OCR erros, the gender could be misspelled
so check the the next field "والد"
then check whatever the value start with "ه" or "ها" as continued of label
thefor the label would be "والده" or "والدها"
print your guess
then in field 7 you have to print the value with both the mispelled geneder and name of newborn removed and only keep the place name
""";

String input = """
{
  "fields": [
    {
      "id": 1,
      "key": "الصفحة",
      "value": "4"
    },
    {
      "id": 2,
      "key": "فى يوم",
      "value": "ثالث عشر ربيع الثاني"
    },
    {
      "id": 3,
      "key": "من سنة",
      "value": "ثلاثة عشر\nواربعمائة والف"
    },
    {
      "id": 4,
      "key": "(هجرية) موافق",
      "value": "عاشر اكتوبرائن\nوتنفيس نهائة والف"
    },
    {
      "id": 5,
      "key": "على الساعة",
      "value": ""
    },
    {
      "id": 6,
      "key": "والدقيقة",
      "value": ""
    },
    {
      "id": 7,
      "key": "ولد ب",
      "value": "ـ دوار لوطي\nLaicen - لحسن وكر"
    },
    {
      "id": 8,
      "key": "والد",
      "value": "ه الزموري\nمحمد"
    },
    {
      "id": 9,
      "key": "الذي اتخذ الاسم العائلى",
      "value": "بند قاق\nBENDAKAK"
    },
    {
      "id": 10,
      "key": "المغربي الجنسية المولود بـ",
      "value": "جماعة الشر الجديد\nاقليم الجديدة"
    },
    {
      "id": 11,
      "key": "في",
      "value": "عام ثامن مشى رمضان ست وثمانين وثلاثمائة\nوالف -"
    },
    {
      "id": 12,
      "key": "موافق",
      "value": "عشرين د جنى سبعة\nويتين تمائة والف"
    },
    {
      "id": 13,
      "key": "حرفته",
      "value": "عامل نك حبي"
    },
    {
      "id": 14,
      "key": "ووالدته",
      "value": "مريم بنت محمد"
    },
    {
      "id": 15,
      "key": "جنسيتها",
      "value": "مغر بم"
    },
    {
      "id": 16,
      "key": "ولدت بـ",
      "value": "جماعة\nالشر الجدية - اقليم الجد بدة"
    },
    {
      "id": 17,
      "key": "في",
      "value": "عام تابع على"
    },
    {
      "id": 18,
      "key": "موافق",
      "value": "ثامناً من يونية تسعة وستين\nربيع الأول عمان وثمانين وثل امانت و\nتسعمائة والا"
    },
    {
      "id": 19,
      "key": "حرفتها",
      "value": "ربة بيت"
    },
    {
      "id": 20,
      "key": "الساكنان بـ",
      "value": "دوار لوطى لعديرة"
    },
    {
      "id": 21,
      "key": "وحرر في يوم",
      "value": "ثاني جمادى الاولى ثلاثة عشر واربعمائة\nوالف"
    },
    {
      "id": 22,
      "key": "(هجرية) موافق",
      "value": "ثامناً وعشرين اكتوبر اتنين وتعين\nتمائة والف"
    },
    {
      "id": 23,
      "key": "حسبما صرح به السيد",
      "value": "بند تا قان موري\nوالده تحت عدد 07"
    },
    {
      "id": 24,
      "key": "عمره",
      "value": "خمس وعشريس"
    },
    {
      "id": 25,
      "key": "سنة الساكن ب",
      "value": "ـ دوار لوطى\nلعد سيرة"
    },
    {
      "id": 26,
      "key": "الذي بعد الاطلاع عليه أمضاه معنا أو أمضيناه وحدنا نحن",
      "value": "الطاهي رهيب رئيس المجلس\nالقروب بالغدر"
    },
    {
      "id": 27,
      "key": "ضابط الحالة المدنية",
      "value": ""
    }
  ]
}
""";

void main() async {
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    // String modelPath = "/Users/adel/Downloads/gemma-3-finetune.Q8_0.gguf";
    // String modelPath = "/Users/adel/Workspace/gguf/gemma-3-12b-it-q4_0.gguf";
    String modelPath = "/Users/adel/Workspace/gguf/gemma-3-4b-it-q4_0.gguf";
    // String modelPath = "/Users/adel/Workspace/gguf/Qwen3-30B-A3B-Q4_K_M.gguf";

    ChatHistory history = ChatHistory()
      ..addMessage(role: Role.user, content: "$prompt\n$input")
      ..addMessage(role: Role.assistant, content: "");

    final modelParams = ModelParams();

    final contextParams = ContextParams()
      ..nPredict = -1
      ..nBatch = 8192
      ..nCtx = 8192;

    final samplerParams = SamplerParams()
      ..temp = 0.1
      ..topK = 64
      ..topP = 0.95
      ..penaltyRepeat = 1.1;

    Llama llama =
        Llama(modelPath, modelParams, contextParams, samplerParams, false);

    llama.setPrompt(
        history.exportFormat(ChatFormat.gemini, leaveLastAssistantOpen: true));
    while (true) {
      var (token, done) = llama.getNext();
      stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");

    llama.dispose();
  } catch (e) {
    print("\nError: ${e.toString()}");
  }
}
