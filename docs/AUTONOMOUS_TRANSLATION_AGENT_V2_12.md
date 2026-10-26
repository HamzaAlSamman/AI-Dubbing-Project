# V2.12 — Autonomous Translation Agent

المسار الجديد في **القسم 6D** من دفتر Colab لا يتطلب مرحلة اعتماد بشري إلزامية.

```text
Source-only course → WhisperX aligned Master
→ translate one TU with bounded context
→ separate model QA for every TU
→ automatic repair and re-review (limited attempts)
→ machine gate:
   PASS → candidate for TTS preview
   BLOCK → quarantine; do not skip or fabricate
```

يعمل على كورسات جديدة دون ترجمة عربية، ولا يرسل الكورس دفعة واحدة. يحتفظ بملفات Checkpoint مستقلة، مع بصمات معرّفات الوحدات والمصدر والتوقيتات. لا يعدّل نصوص WhisperX أو التوقيتات، ولا يكتب في `final_dub_ar`.

**الحدود:** اجتياز الفحص الآلي لا يثبت صحة المعنى بصورة قطعية ولا يمثل اعتماداً نهائياً لجودة الصوت. لا يرسل الكود أي API أو يشغّل صوتاً بالافتراض. يلزم تفعيل صريح للاتصال والتأكد من الحصة والفوترة. الوحدات المحجوبة لا تنتقل إلى الصوت تلقائياً.

المكتبة: `arabic_dubbing_v2/translation_agent_v212.py`، والاختبارات: `tests/test_translation_agent_v212.py`.
