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
- `notebooks/ArabicDubbing_CourseTranslation_V2_6_DynamicGlossary_QA_Colab.ipynb`: دفتر Colab V2.6 يسحب الكود من GitHub، ويقرأ أحدث Master متاح من Drive، ثم يحفظ نسخاً مشتقة فقط.
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

تأكدنا من وجود `translation_master_v2_5_editorial_candidate.json` في Google Drive داخل `ArabicDubbingV2/outputs/lesson/gemini_translation_v1/`، ومعه `editorial_bilingual_qa_v2_5.json`. هذه الملفات ليست جزءاً من المستودع العام؛ يقرأها Colab من Drive ولا ينسخها إلى GitHub. V2.5 مسودة تحريرية وليست اعتماداً نهائياً للصوت. لم نُجرِ اختبار تكامل مباشر على جلسة Colab.

أما وحدات `TU Builder V1.1` المراجعة، فهي محفوظة من حيث المفهوم في المعمارية، لكن شيفرة ZIP الأصلية للإصدار V1.1 غير مرفقة. الكود هنا يحتوي Builder تجريبياً مستقلاً يحتاج تقييم الحدود على الصوت الحقيقي قبل أن يحل محله.

## حدود الصلاحية الحالية

التنفيذ الحالي قابل للتشغيل والفحص محلياً، لكن **لا يحقق بعد المنتج النهائي الكامل**: يحتاج ملفات Master الأصلية للتحقق الحقيقي، وBackend ترجمة يختاره المستخدم ويأذن بتشغيله، ومراجعة دلالية متخصصة، وتجربة TTS فعلية، ثم المزامنة والمكساج. لا يصح وصف هذه المراحل بأنها اختُبرت على كورس حقيقي.


## الاتصال بـ GitHub و Colab

- المستودع العام: `HamzaAlSamman/AI-Dubbing-Project`، الفرع `main`.
- دفتر Colab الرئيسي: `notebooks/ArabicDubbing_CourseTranslation_V2_6_DynamicGlossary_QA_Colab.ipynb`.
- خلية الإعداد تستنسخ المستودع العام قراءة فقط، ولا تستبدل مجلد checkout موجوداً أو تغيّر الملفات الأصلية.
- لا ترفع أصول الفيديو، الصوت، تفريغات الكورس المحمية، أرشيف المحادثات، أو مفاتيح API إلى المستودع العام.

## إضافات تشغيل V2.6.1: الترجمة والصوت المجاني (اختياري)

تم تحديث **الدفتر نفسه** `notebooks/ArabicDubbing_CourseTranslation_V2_6_DynamicGlossary_QA_Colab.ipynb` دون إنشاء دفتر بديل. لا تنفذ هذه الإعدادات تلقائياً خدمات مدفوعة أو أي اتصال خارجي:

### النصوص / الترجمة

| `TRANSLATION_MODE` | الوظيفة | اتصالات خارجية افتراضياً |
| --- | --- | --- |
| `reuse_existing` | استخدام الـ`natural_ar` الموجود بدون إعادة ترجمة | لا |
| `review_existing` | تدقيق ثنائي اللغة بوحدة واحدة أو أكثر عبر `GeminiJSONBackend`، مع تقرير مستقل | يتطلب `RUN_TEXT_API=True` و`ALLOW_ONLINE_TEXT=True` |
| `regenerate_selected` | أربع مراحل ترجمة جديدة للوحدات المحددة فقط | يتطلب الإذن نفسه، أربع طلبات لكل وحدة |

يستخدم خيار Gemini النصّي `gemini-2.5-flash` نموذجاً تتوافر له فئة API مجانية عند أهلية المشروع وحدود الخطة. **قد تنشأ رسوم إذا فعّلت Billing، ولا يوجد تحقق برمجي من حالة اشتراكك.** يُحفظ السر في `google.colab.userdata` باسم `GEMINI_API_KEY`؛ لا ترفعه إلى GitHub. قد تُستخدم بيانات الطبقة المجانية لتحسين خدمات المزود؛ تأكد من حقوق مشاركة المحتوى. لا يتم ملء `final_dub_ar`.

### معاينة الكلام العربي

| `TTS_ENGINE` | وصف وقيود |
| --- | --- |
| `plan_only` | تخطيط فقط، لا صوت ولا اتصال |
| `edge_online` | أصوات Edge Neural عبر `edge-tts`: خدمة عبر الإنترنت غير رسمية وغير مضمونة للإنتاج؛ يتطلب `RUN_TTS_PREVIEW=True` و`ALLOW_ONLINE_TTS=True` |
| `mms_local` | `facebook/mms-tts-ara` يعمل محلياً على Colab بعد تنزيل النموذج؛ **ترخيص غير تجاري CC-BY-NC-4.0**؛ يتطلب `RUN_TTS_PREVIEW=True` و`ALLOW_LOCAL_TTS=True` |
| `gemini_flash_lite` | `gemini-3.8-flash-lite-tts` يدعم العربية وFree Tier عند الأهلية؛ يتطلب مفتاح Gemini والإذن عبر الإنترنت، وقد تنشأ رسوم لمشروع مدفوع |

`MAX_TTS_UNITS = 1` يحول دون إجراء معاينة لجميع الوحدات بالخطأ. تحفظ المعاينات وبيانات مدتها وبصماتها تحت `translation_v2_figma_implementation/temporary_audio_previews_v2_6/` من دون تعديل الملف الأصلي. لا تُنشأ أصوات معتمدة آلياً، ويظل القرار النهائي للمراجع.

### الاختبار المحلي

- `python -m unittest discover -s tests -v`: 21 اختباراً محلياً بما فيها أذونات التنفيذ وحماية المخرجات من الاستبدال.
- اختُبرت تهيئة Master V2.5 الحقيقي وقراءة وحداته الـ17 وتخطيط الصوت محلياً بلا أي اتصال خارجي، **ولم تُختبر جودة صوت Gemini أو Edge أو MMS فعلياً على جلسة Colab بعد**.

مصادر معلومات المزوّدين: https://ai.google.dev/gemini-api/docs/pricing ، https://ai.google.dev/gemini-api/docs/speech-generation ، https://github.com/rany2/edge-tts ، https://huggingface.co/facebook/mms-tts-ara .
