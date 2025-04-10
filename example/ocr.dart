// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() async {
  try {
    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/qwq-32b-q4_k_m.gguf";

    ContextParams contextParams = ContextParams()..nCtx = 512;
    SamplerParams samplerParams = SamplerParams()
      ..topK = 40
      ..topP = 0.95
      ..penaltyRepeat = 1.1
      ..temp = 0.8;

    Llama llama =
        Llama(modelPath, ModelParams(), contextParams, samplerParams, true);

    const instruction = """
في ثامن محرم الف و ثلاثمائة و تسعة و تسعين
الموافق لثلاثين يوليوز سنة
الف و تسعمائة و سبعة و سبعين
ان باشا مدينة مسقط
قد طلب منا تقييد التصريح الدولي, الآتي رقم: 1597 - ح.م.م.ع.
في تاريخ ألف و ثلاثمائة و تسعة و تسعين هجرية
الموافق لسنة الف وتسعمائة.
على الساعة
ازداد بمدينة ولاد المختار
بنت الحرم خديجة مليكة الترتس قبيلة
من والده الجيلالي عيسى.

المغربي الجنسية المولود في تاريخ
الموافق
ب-
حرفته
ELBACHARY
الذي اتخذ الاسم العائلي البشاري.
ومن والدته عائشة بنت علي.
جنسيتهما المسلمة.
المولودة في تاريخ الف وثلاثمائة
الموافق لسنة الف وتسعمائة وسبعة وسبعين
حرفتها لا شيئ.
الساكنين بمسقط
قبيل رقم 447.
وقع هذا التصريح في ثامن يوم الف وثلاثمائة و تسعة.
الموافق لثلاثين يوليوز سنة
الف وتسعمائة و سبعة وسبعين
من صاحبة حرفة.
عمره
الساكن بالمنزل المذكور
لدينا نحن العربي محمد.
ضابط الحالة المدنية فامضينا عليه مع المصرح.
ضابط الحالة المدنية (السكتي)
طابع مكتب الحالة المدنية (السكتي)
امضاء:
قيدناه في تاريخ 18 محرم الف وثلاثمائة و تسعة وسبعين.
الموافق لثالث غشت و يوليوز سنة
الف وتسعمائة وسبعة.
نحن الكاتب العام لبلدية جماعة
التوالت.
ضابط الحالة المدنية,
(ارح)
""";
    const text = """
please perform language correction on this text from OCR function
""";

    llama.setPrompt("""
<start_of_turn>user\n$instruction\n<end_of_turn>
<start_of_turn>user\n$text\n<end_of_turn>
<start_of_turn>model\n
""");
    while (true) {
      var (token, done) = llama.getNext();
      stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");

    llama.dispose();
  } catch (e) {
    print("Error: ${e.toString()}");
  }
}
