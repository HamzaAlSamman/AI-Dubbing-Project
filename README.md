# مشروع الدبلجة — تنفيذ مرحلة الترجمة V2 وفق Figma

**الحالة: حزمة برمجية مُختبرة محلياً على بيانات اصطناعية؛ ليست دبلجة نهائية أو موافقة على ترجمة المستخدم.**

هذه الحزمة تنفّذ الأساس البرمجي للقرارات المعمارية في أقسام Figma `02.9–02.27`، خصوصاً:

1. **Data Contract:** نسخة `Master` مشتقة بدون محو الأصل، مع حفظ `source_segments`، و`translation_units`، والمعرّفات، والكلمات المحاذاة والتوقيتات ونسخ الترجمة الحالية.
2. **Translation Unit Builder:** نسخة تجريبية مستقلة تدعم JOIN / SPLIT مع تغطية `source_refs` لكل كلمة وعدم توليد أوقات وهمية. *لا تحل تلقائياً محل الـBuilder V1.1 المراجع بشرياً*.
3. **Context Builder:** مراجع الجوار عبر `previous_unit_ids` و`next_unit_ids`، وتجميع السياق وقت الحاجة فقط.
4. **Dynamic Glossary:** كشف مجال تمهيدي من عنوان الكورس أو أنماط نصه؛ استخراج مرشّحات مصطلحات من المحتوى ذاته، وإعداد خطط `Contextual Validation` دون قاموس خاص بكورس Andrew Ng؛ ترجمات المرشحات تظل فارغة حتى مراجعتها أو تفعيل مزوّد ترجمة منفصل بشكل صريح.
5. **Multi-stage Translation V2:** أربع خطط متميزة `accurate_ar`، `natural_ar`، `timing_candidate_ar`، و`semantic QA`. يمكن ربط Backend قابل للحقن، لكن لا تتصل هذه الحزمة بأي LLM أو API تلقائياً. لا يتم ملء `final_dub_ar`.
6. **Educational QA:** فحوص تحفظ/تنبّه إلى الأعداد، الكود، وبعض مؤشرات النفي، مع حقول `unknown` للفحص الدلالي غير المنفذ. **لا تدّعي** أن الترجمة سليمة دلالياً دون فحص ثنائي اللغة مستقل.
7. **TTS Preview:** خطة معاينة عامة لوحدات الكورس (لا توجد معرفات ثابتة 7/15). توليد Edge TTS عبر الإنترنت **مُعطل** ما لم يُفعّل المستخدم صراحةً علمي `--execute` و`--allow-online-tts`، ويتطلب تثبيت `edge-tts` منفصلاً. لا Google TTS.

## محتويات الحزمة

- `arabic_dubbing_v2/`: كود مستقل يعتمد فقط على مكتبات Python القياسية للتهيئة/التحليل/خطط الترجمة.
- `tests/`: اختبارات تلقائية لا تتصل بالإنترنت ولا تشغل WhisperX ولا TTS.
- `notebooks/ArabicDubbing_CourseTranslation_V2_6_DynamicGlossary_QA_Colab.ipynb`: دفتر Colab مستقل لقراءة أحدث Master متاح من Drive، وتحليل السياق والمصطلحات وتصدير الخطط وحفظ نسخة جديدة.
- `examples/`: ملفات اصطناعية صغيرة من مجالات متعددة لا تُمثل كورسات حقيقية.

## تشغيل محلي من مجلد الحزمة

```bash
python -m unittest discover -s tests -v
python -m arabic_dubbing_v2.cli prepare --master /path/to/translation_master_v2_5_editorial_candidate.json --out /path/to/translation_master_v2_6_derived.json
python -m arabic_dubbing_v2.cli plan-terminology --master /path/to/translation_master_v2_6_derived.json --out /path/to/terminology_plans.json
python -m arabic_dubbing_v2.cli plan-translation --master /path/to/translation_master_v2_6_derived.json --out /path/to/translation_pass_plans.json
python -m arabic_dubbing_v2.cli preview --master /path/to/translation_master_v2_6_derived.json --out-dir /path/to/previews
```

الأمر الأخير **يعرض خطة فقط** ولن يولّد صوتاً. لو أراد المستخدم الاختبار الصوتي صراحة بعد مراجعة النصوص، يتم تثبيت المكتبة ثم استخدام `--execute --allow-online-tts`.

## قواعد السلامة والنسخ

- يرفض النظام الكتابة فوق ملف موجود.
- لا تستعمل فهارس وحدات مصمّمة لدرس معين، ولا تفترض مجال ذكاء اصطناعي.
- لا تعيد ASR/WhisperX أو الترجمة أو توليد الصوت أثناء `prepare`.
- الملفات الأصلية تُقرأ فقط وتُتحقق بصمتها (`SHA256`).
- لا ينقل النظام مقابلات عربية من قاموس قديم إلى ذاكرة مصطلحات معتمدة إلا إذا كان السجل موسوماً `approved` ومقيّداً بمعرّف **الكورس نفسه**.
- `term candidates` ليست ترجمة، وتحليل المجال الاستخراجي ليس تصنيفاً متخصصاً معتمداً.
- `final_dub_ar` والمزامنة والدبلجة النهائية لا يُستنتجون أو يُعتمدون تلقائياً.

## موضع التكامل مع بيانات المشروع الفعلية

تم لاحقاً التحقق من وجود `translation_master_v2_5_editorial_candidate.json` و`editorial_bilingual_qa_v2_5.json` في Google Drive ضمن `ArabicDubbingV2/outputs/lesson/gemini_translation_v1/`. هذه الملفات محفوظة على Drive وليست مرفوعة للمستودع العام. تحتاج نسخة V2.5 إلى فحص تكامل محلي على البيانات الفعلية؛ لا تُعد نهائية أو معتمدة للصوت.

أما وحدات `TU Builder V1.1` المراجعة، فهي محفوظة من حيث المفهوم في المعمارية، لكن شيفرة ZIP الأصلية للإصدار V1.1 غير مرفقة. الكود هنا يحتوي Builder تجريبياً مستقلاً يحتاج تقييم الحدود على الصوت الحقيقي قبل أن يحل محله.

## حدود الصلاحية الحالية

التنفيذ الحالي قابل للتشغيل والفحص محلياً، لكن **لا يحقق بعد المنتج النهائي الكامل**: يحتاج ملفات Master الأصلية للتحقق الحقيقي، وBackend ترجمة يختاره المستخدم ويأذن بتشغيله، ومراجعة دلالية متخصصة، وتجربة TTS فعلية، ثم المزامنة والمكساج. لا يصح وصف هذه المراحل بأنها اختُبرت على كورس حقيقي.

## GitHub و Colab

- المستودع: `HamzaAlSamman/AI-Dubbing-Project`، الفرع `main`.
- دفتر Colab الرئيسي: `notebooks/ArabicDubbing_CourseTranslation_V2_6_DynamicGlossary_QA_Colab.ipynb`.
- هذا المستودع عام؛ لا ترفع أرشيف المحادثات أو مفاتيح API أو الملفات الصوتية أو نصوص المصدر المحمية.
- لا توجد أي خطوة تشغيل تلقائي للترجمة أو TTS؛ يلزم موافقة ومراجعة بشرية.
