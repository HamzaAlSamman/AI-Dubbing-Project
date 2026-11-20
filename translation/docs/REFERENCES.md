# المراجع والدراسات

**مفتاح الحالة:**
- **قُرئ**: قرأنا الملخص أو النص أو صفحة النموذج أثناء العمل، والأرقام المذكورة منقولة منه.
- **خلفية**: مرجع معروف استُند إليه في التصميم دون إعادة قراءته في هذا العمل.

BibTeX في [references.bib](references.bib).

## الترجمة الواعية بالمدة والدبلجة الآلية

| # | المرجع | ما أخذناه | الحالة |
|---|---|---|---|
| 1 | Lakew et al. *Isometric MT: Neural Machine Translation for Automatic Dubbing.* ICASSP 2022. [arXiv:2112.08682](https://arxiv.org/abs/2112.08682) | تعريف المشكلة. الحل المعتاد هو توليد عدة ترجمات ثم ترتيبها حسب الطول والجودة | قُرئ (الملخص) |
| 2 | *Prompting LLMs: Length Control for Isometric Machine Translation.* 2025. [arXiv:2506.04855](https://arxiv.org/abs/2506.04855) | تحديد الطول صراحة في الـ prompt، و±10% من حروف الأصل، و10 مخرجات + فلتر طول + ترتيب بـ QE. نسبة الطول المناسب حوالي 65–72% | قُرئ (النص) |
| 3 | Subramanian et al. *Length Aware Speech Translation for Video Dubbing.* Interspeech 2025. [arXiv:2506.00740](https://arxiv.org/abs/2506.00740) | نسخ قصيرة/عادية/طويلة في تمرير واحد، واختيار حسب مدة متوقعة دون توليد صوت. تحسّن التزامن 16–20% | قُرئ (النص) |
| 4 | Won et al. *End-to-End Multilingual Automatic Dubbing via Duration-based Translation with LLMs.* EMNLP 2025 (Demos). [ACL Anthology](https://aclanthology.org/2025.emnlp-demos.37/) | تحويل مدة المقطع إلى هدف طول للنموذج، ثم تعديل متكرر | قُرئ (الملخص) |
| 5 | Cui et al. *Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization.* ACL 2025. [ACL Anthology](https://aclanthology.org/2025.acl-long.227/) | ضبط المدة بالتفضيل (preference optimization)، كبديل تدريبي مستقبلي | قُرئ (الملخص) |
| 6 | *Length-Aware NMT and Adaptive Duration for Automatic Dubbing.* IWSLT 2023. [ACL Anthology](https://aclanthology.org/2023.iwslt-1.9.pdf) | وسوم نسبة الطول، وإعادة الترتيب بمدة متوقعة | قُرئ (الملخص) |
| 7 | *Isometric NMT using Phoneme Count Ratio Reward-based RL.* 2024. [arXiv:2403.15469](https://arxiv.org/abs/2403.15469) | مطابقة عدد الصوتيات بالتعلم المعزز. تذكر الورقة أنه يضر بالجودة، ولذلك لم نتبعه | قُرئ (الملخص) |

## نماذج الترجمة

| # | المرجع | ما أخذناه | الحالة |
|---|---|---|---|
| 8 | *TranslateGemma: A new suite of open translation models.* 2026. [arXiv:2601.09012](https://arxiv.org/abs/2601.09012) · [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/) | مدرَّب بمكافأة MetricX-QE، ولذلك استُبعد من التقييم تحت بوابة MetricX. الأحجام 4B و12B و27B، ورخصة Gemma مع موافقة مسبقة | قُرئ (الملخص وصفحة النموذج) |
| 9 | Tencent Hunyuan. *Hy-MT2 Technical Report.* 2026. [arXiv:2605.22064](https://arxiv.org/abs/2605.22064) · [Hy-MT2-7B](https://hf.co/tencent/Hy-MT2-7B) | الرخصة Apache-2.0. XCOMET على FLORES-200: الـ 7B ‏86.9 مقابل حوالي 88.7 لـ Gemini 3.1 Pro. يتفوق على HY-MT1.5 في WMT25. قالب الـ prompt من صفحة النموذج | قُرئ |
| — | Tencent Hunyuan. *HY-MT1.5* ([tencent/HY-MT1.5-7B](https://hf.co/tencent/HY-MT1.5-7B)) | المولّد الأساسي. القالب بلا سياق (القالب السياقي ترجم السياق أيضًا في تجاربنا) | قُرئ (صفحة النموذج) |
| — | WMT 2026 Proceedings. [ACL Anthology](https://aclanthology.org/volumes/2026.wmt-main/) | الإنكليزية→العربية من أصعب الأزواج، وأنظمة الترجمة المتخصصة قوية فيها | قُرئ (ملخص البحث) |

## التقييم الآلي

| # | المرجع | ما أخذناه | الحالة |
|---|---|---|---|
| 10 | Juraska et al. *MetricX-24: The Google Submission to the WMT 2024 Metrics Shared Task.* WMT 2024. [arXiv:2410.03983](https://arxiv.org/abs/2410.03983) | مقياس الجودة بلا مرجع (QE) في البوابة (`metricx-24-hybrid-large-v2p6`) | خلفية |
| 11 | Kocmi & Federmann. *Large Language Models Are State-of-the-Art Evaluators of Translation Quality* (GEMBA). EAMT 2023. [arXiv:2302.14520](https://arxiv.org/abs/2302.14520) | تقييم بدرجات 0–100 بنموذج لغوي، للمرتِّب وللحَكَم المستقل | خلفية |
| 12 | Fernandes et al. *Quality-Aware Decoding for Neural Machine Translation.* NAACL 2022. [arXiv:2205.00978](https://arxiv.org/abs/2205.00978) | الترتيب بمقاييس الجودة يحسّن الترجمة، لكن التقييم بالمقياس نفسه يضخّم النتيجة. لذلك استعملنا حَكَمًا مستقلًا | خلفية |
| 13 | Panickssery et al. *LLM Evaluators Recognize and Favor Their Own Generations.* 2024. [arXiv:2404.13076](https://arxiv.org/abs/2404.13076) | انحياز النموذج لمخرجاته. لذلك نخفي المصدر، ونرتّب عشوائيًا، ونختار الحَكَم من عائلة ثالثة | خلفية |

## البيانات والـ fine-tuning (للمرحلة القادمة)

| # | المرجع | ما أخذناه | الحالة |
|---|---|---|---|
| 14 | Abdelali et al. *The AMARA Corpus: Building Parallel Language Resources for the Educational Domain* (QED). LREC 2014. [ACL Anthology](https://aclanthology.org/L14-1675/) · [OPUS](https://opus.nlpl.eu/QED/corpus/version/QED) | ترجمات عربية لفيديوهات تعليمية (حوالي 2.6 مليون كلمة عربية)، رخصة بحثية، للـ fine-tuning مستقبلًا | قُرئ (الصفحات) |
| 15 | Guzmán et al. *The AMARA Corpus…* IWSLT 2013. [PDF](https://aclanthology.org/www.mt-archive.info/10/IWSLT-2013-Guzman.pdf) | حجم الجزء العربي–الإنكليزي | قُرئ (الملخص) |
| 16 | *Bridging Scientific Heritage: An Arabic–Russian Parallel Corpus and LLM Benchmark.* 2026. [arXiv:2606.30943](https://arxiv.org/abs/2606.30943) | QLoRA على Qwen2.5-7B: ‏BLEU من 18.8 إلى 23.2، وCOMET من 0.707 إلى 0.758. دليل على جدوى الـ fine-tuning | قُرئ (ملخص البحث) |

## الصوت (مرحلة لاحقة تستعمل مخرج هذه الخطوة)

| # | المرجع | الاستخدام |
|---|---|---|
| 17 | Radford et al. *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). 2022. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) | قياس CER للمقاطع المولّدة (large-v3) |
| — | Habibi-TTS ([SWivid/Habibi-TTS](https://github.com/SWivid/Habibi-TTS)) | صوت الفصحى، ومنه سرعة 10.58 حرف/ث المستعملة في تقدير المدة |
