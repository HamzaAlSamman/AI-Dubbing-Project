"""Offline tests for Translation V2 language features (policy, LLM-built glossary, gate, polish).
Run from the repo root:  python -m unittest discover -s translation/tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubbing_translation.gate import evaluate                                           # noqa: E402
from dubbing_translation.generators import HFChatMT, HyMT                              # noqa: E402
from dubbing_translation.io import V2_CONFIG, load_config                             # noqa: E402
from dubbing_translation.language import (Glossary, Language, Policy, build_glossary,  # noqa: E402
                                          normalize_ar, require_from_forms)
from dubbing_translation.pipeline import run                                          # noqa: E402
from dubbing_translation.text import clean_translation                                # noqa: E402
from dubbing_translation.timing import Window                                         # noqa: E402
from test_translation_step import FakeQE, FakeSpeller, FixedGen, arabic_of            # noqa: E402

CFG2 = load_config(V2_CONFIG)
RATE = CFG2['voice']['letters_per_second']
POLICY = Policy.load('language/msa_simple_v1.toml')
TERMS = [{'en': ['label', 'labels'], 'ar': 'تسمية', 'ar_forms': ['التسميات'], 'avoid': ['الوسم']},
         {'en': ['feature vector'], 'ar': 'متجه الخصائص', 'avoid': ['متجه الميزات']},
         {'en': ['re:\\b(?:capital|upper ?case)\\s+[A-Za-z]\\b'], 'ar': 'Y الكبيرة', 'ar_forms': ['حرف كبير'],
          'avoid': ['رأس المال']}]


class GlossaryClient:
    """Fake LLM: answers the glossary request; counts calls."""
    provider, model = 'fake', 'terminology'

    def __init__(self, terms=TERMS):
        self.terms, self.calls = terms, 0

    def chat_json(self, system, user):
        self.calls += 1
        return {'subject': 'logistic regression notation', 'terms': self.terms}


class TextAndPolicyTests(unittest.TestCase):
    def test_normalize_and_stems(self):
        self.assertEqual(normalize_ar('أَمْثِلَة'), 'امثله')
        self.assertEqual(require_from_forms(['متجه الخصائص']), [['متج'], ['خصائص']])
        self.assertEqual(require_from_forms(['مثال تدريب', 'أمثلة التدريب']), [['امثل', 'مثال'], ['تدريب']])

    def test_dialect_words_and_clitics(self):
        self.assertEqual(sorted(POLICY.dialect_words('شو بدك؟ وعشان هيك')), ['بدك', 'شو', 'عشان', 'هيك'])
        self.assertEqual(POLICY.dialect_words('ماذا تريد؟ لذلك نبدأ'), [])

    def test_calques(self):
        self.assertEqual(POLICY.calque_ids('لتخزين صورة، يقوم الكمبيوتر بتخزين ثلاث مصفوفات'), ['yaqum_bi'])
        self.assertEqual(POLICY.calque_ids('يتضح أنه عند تنفيذ شبكة'), ['yatadih_annahu'])
        self.assertEqual(POLICY.calque_ids('يخزّن الحاسوب ثلاث مصفوفات'), [])

    def test_full_width_punctuation_is_normalised(self):
        self.assertEqual(clean_translation('واحد، فاصلة، m，وهو'), 'واحد، فاصلة، m،وهو')


class GlossaryTests(unittest.TestCase):
    def setUp(self):
        self.g = Glossary.from_terms(TERMS, 'test')

    def test_missing_terms(self):
        src = 'where x is a feature vector and y, the label, is 0 or 1'
        self.assertEqual([t['id'] for t in self.g.missing(src, 'x متجه ميزات و y هو الوسم')], ['label', 'feature_vector'])
        self.assertEqual(self.g.missing(src, 'x هو متجه الخصائص و y هي التسمية'), [])

    def test_regex_terms_and_hymt_pairs(self):
        src = 'we define capital Y to be y1 up to ym'
        self.assertEqual([t['id'] for t in self.g.missing(src, 'سنعرّف رأس المال Y')], ['re_b_capital_upper_case_s_a_za_z_b'])
        self.assertEqual(self.g.missing(src, 'سنعرّف Y الكبيرة'), [])
        self.assertEqual(self.g.hymt_pairs('the labels of the training set'), [('labels', 'تسمية')])

    def test_manual_terms_win_on_merge(self):
        manual = Glossary.from_terms([{'en': ['label'], 'ar': 'وسم'}], 'manual')
        merged = manual.merged(self.g)
        label = [t for t in merged.terms if t['id'] == 'label']
        self.assertEqual(len(label), 1)
        self.assertEqual(label[0]['ar'], 'وسم')
        self.assertEqual(len(merged.terms), 3)

    def test_build_glossary_is_cached_write_once(self):
        tasks = [{'unit_id': 'u0', 'source_text': 'y, the label, is 0 or 1'}]
        client = GlossaryClient()
        with tempfile.TemporaryDirectory() as tmp:
            g1, path = build_glossary(tasks, client, tmp, log=lambda *_: None)
            g2, _ = build_glossary(tasks, client, tmp, log=lambda *_: None)
            self.assertEqual(client.calls, 1)
            self.assertEqual(len(g1.terms), 3)
            self.assertEqual([t['ar'] for t in g1.terms], [t['ar'] for t in g2.terms])
            self.assertTrue(path.exists())


class GateTests(unittest.TestCase):
    def test_language_reasons(self):
        lang = Language(POLICY, Glossary.from_terms(TERMS, 'test'))
        w = Window('u', 3.0, 3.0)
        kw = dict(qe=1.0, misspelled_words=set(), cfg=CFG2, language=lang)
        bad = evaluate('y هو الوسم يا صاحبي شو', 'y, the label', w, **kw)
        self.assertIn('term:label', bad['reasons'])
        self.assertIn('dialect:شو', bad['reasons'])
        stiff = evaluate('يقوم الحاسوب بتخزين التسمية', 'the computer stores the label', w, **kw)
        self.assertNotIn('term:label', stiff['reasons'])
        self.assertEqual(stiff['language']['calques'], ['yaqum_bi'])
        self.assertFalse(any(r.startswith('style') for r in stiff['reasons']))      # flagged, not rejected

    def test_v1_config_has_no_language(self):
        self.assertIsNone(Language.from_config(load_config()))


class GeneratorPromptTests(unittest.TestCase):
    def test_hymt_terminology_template(self):
        task = {'source_text': 'y, the label', 'term_pairs': [('label', 'تسمية')]}
        self.assertIn('label 翻译成 تسمية', HyMT(language_aware=True).prompt(task))
        self.assertTrue(HyMT().prompt(task).startswith('Translate the following segment into Arabic'))
        self.assertTrue(HyMT(language_aware=True).name.endswith('__terms_v1'))

    def test_hy_mt2_terms_template_is_language_aware(self):
        self.assertTrue(HFChatMT('tencent/Hy-MT2-7B', 'hy_mt2_terms').language_aware)
        self.assertFalse(HFChatMT('tencent/Hy-MT2-7B', 'hy_mt2').language_aware)


class V2PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.tasks = [{'unit_id': 'u0', 'source_text': 'y is the label of this image', 'context_before': [],
                       'context_after': [], 'source_duration_s': 3.0}]
        self.windows = {'u0': Window('u0', 3.0, 3.0)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_glossary_rejects_wrong_term_and_polish_removes_stiff_pattern(self):
        n = int(3.0 * RATE)
        wrong = 'الوسم ' + arabic_of(n - 5)                                   # misses the approved term
        stiff = 'يقوم بتحديد التسمية ' + arabic_of(n - 16)                 # approved term, stiff pattern
        polished = 'يحدد التسمية ' + arabic_of(n - 11)
        qe = FakeQE({wrong: 0.5, stiff: 1.5, polished: 1.6})

        class Client(GlossaryClient):
            def chat_json(self, system, user):
                if 'terminology lead' in system:
                    return super().chat_json(system, user)
                self.calls += 1
                return {'natural_ar': polished}

        client = Client()
        gens = [FixedGen('mt', {'u0': wrong}), FixedGen('llm', {'u0': stiff})]
        run_dir, final, metrics = run(self.tasks, self.windows, CFG2, self.work, generators=gens, qe=qe,
                                      speller=FakeSpeller(), refine_client=client, log=lambda *_: None)
        f = final[0]
        self.assertEqual(f['status'], 'accepted')
        self.assertEqual(f['natural_ar'], polished)                  # wrong term rejected, stiff line polished
        self.assertEqual(f['polished_from'], 'llm:v')
        self.assertEqual(metrics['polished_units'], 1)
        self.assertEqual(metrics['units_term_missing'], 0)
        self.assertEqual(f['accurate_ar'], wrong)                    # plain MT line kept as the accurate version
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['language']['glossary_terms'], 4)     # 3 of the video + 1 generic (lowercase)
        self.assertTrue((run_dir / 'glossary.json').exists())


    def test_compare_v1_v2_reads_both_runs(self):
        from dubbing_translation.colab import compare_v1_v2
        n = int(3.0 * RATE)
        line = 'التسمية ' + arabic_of(n - 7)
        gens = [FixedGen('mt', {'u0': line})]
        work = self.work / 'translation_step'
        kw = dict(generators=gens, qe=FakeQE(), speller=FakeSpeller(), refine_client=GlossaryClient(),
                  log=lambda *_: None)
        run(self.tasks, self.windows, CFG2, work, run_label='final_v1', **kw)
        v2_dir, _, _ = run(self.tasks, self.windows, CFG2, work, run_label='v2', **kw)
        table = compare_v1_v2(self.work, v2_dir)
        self.assertEqual(set(table), {'V1 final_v1', 'V2'})
        self.assertEqual(table['V2']['accepted_pct'], 100.0)
        self.assertEqual(table['V2']['term_missing_units'], 0)



class SegmentTests(unittest.TestCase):
    def test_merge_fragments_keeps_ids_and_sums_windows(self):
        from dubbing_translation.segment import merge_fragments
        tasks = [{'unit_id': 'a', 'source_text': 'So if you take X1 and'},
                 {'unit_id': 'b', 'source_text': 'you know, put that as the first column.'},
                 {'unit_id': 'c', 'source_text': 'Next idea.'}]
        windows = {'a': Window('a', 1.0, 1.2), 'b': Window('b', 3.0, 3.5), 'c': Window('c', 1.0, 1.0)}
        t, w, merges = merge_fragments(tasks, windows)
        self.assertEqual([x['unit_id'] for x in t], ['a+b', 'c'])
        self.assertEqual(t[0]['merged_from'], ['a', 'b'])
        self.assertEqual((w['a+b'].slot_s, w['a+b'].window_s), (4.0, 4.7))
        self.assertEqual(len(merges), 1)

    def test_spelled_digits(self):
        from dubbing_translation.segment import fix_spelled_digits
        self.assertEqual(fix_spelled_digits('to be 1, 2, 2, 8, 8. That')[0], 'to be 12288. That')
        self.assertEqual(fix_spelled_digits('values 255, 231, and 1, 2, 3')[1], [])

    def test_simplicity_findings(self):
        long_line = ' '.join(['كلمة'] * 30) + '.'
        self.assertEqual(POLICY.simplicity_ids(long_line), ['long_sentence:30'])
        self.assertIn('rare:ثمه', POLICY.simplicity_ids('ثمة فكرة أخرى.'))
        self.assertTrue(any(x.startswith('address') for x in POLICY.simplicity_ids('وحتى إن كنتم تعلمونه')))
        self.assertEqual(POLICY.simplicity_ids('أهلًا بكم، سترى الفكرة الآن.'), [])
        self.assertEqual(POLICY.calque_ids('يتم تمثيل المثال بزوج'), ['yatimm'])


if __name__ == '__main__':
    unittest.main()
