"""Offline tests: no model, no network. Run from the repo root:
    python -m unittest discover -s translation/tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubbing_translation import text as T                                          # noqa: E402
from dubbing_translation.gate import evaluate, mechanical_issues, rank_key        # noqa: E402
from dubbing_translation.generators import LLMClient, parse_json_object           # noqa: E402
from dubbing_translation.io import load_config                                    # noqa: E402
from dubbing_translation.pipeline import consensus_allowlist, run                 # noqa: E402
from dubbing_translation.timing import Window, duration_verdict, letter_budget, windows_from_timing  # noqa: E402

CFG = load_config()
RATE = CFG['voice']['letters_per_second']


class FakeQE:
    name = 'fake-qe'

    def __init__(self, table=None, default=2.0):
        self.table, self.default, self.calls = table or {}, default, 0

    def score(self, pairs):
        self.calls += 1
        return [self.table.get(h, self.default) for _, h in pairs]


class FakeSpeller:
    name = 'fake-speller'

    def __init__(self, bad=()):
        self.bad = set(bad)

    def misspelled(self, words):
        return {w for w in words if w in self.bad}


class FixedGen:
    uses_budget = False

    def __init__(self, name, texts):
        self.name, self.texts, self.calls = name, texts, 0

    def generate(self, task, budget):
        self.calls += 1
        t = self.texts.get(task['unit_id'])
        return [{'text': t, 'variant': 'v'}] if t else []


class FakeClient:
    provider, model = 'fake', 'refiner'

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def chat_json(self, system, user):
        self.calls += 1
        return {'natural_ar': self.replies.pop(0)} if self.replies else None


def arabic_of(n_letters):
    """Arabic text with exactly n letters (words of 5)."""
    words, left = [], n_letters
    while left > 0:
        k = min(5, left)
        words.append('كتبنا'[:k])
        left -= k
    return ' '.join(words)


class TextTests(unittest.TestCase):
    def test_letters_ignore_diacritics_spaces_punctuation_latin(self):
        self.assertEqual(T.n_letters('مَرْحَبًا، يا x عالم!'), len('مرحبا') + len('يا') + len('عالم'))

    def test_clean_removes_backticks_quotes_and_double_opener(self):
        self.assertEqual(T.clean_translation('إذًا، إذا أخذت “X1”'), 'إذا أخذت X1')
        self.assertEqual(T.clean_translation('المتغير `x` يمثل مصفوفة `nx × m`'), 'المتغير x يمثل مصفوفة nx × m')
        self.assertEqual(T.clean_translation('```json\n"نص"\n```'), 'نص')

    def test_tts_text_spells_variables_and_symbols(self):
        out = T.tts_text('المصفوفة x ذات أبعاد nx × m')
        self.assertIn('إكس', out)
        self.assertIn('إن إكس في إم', out)
        self.assertNotRegex(out, '[A-Za-z×]')

    def test_tts_text_enumeration_and_dimensions(self):
        out = T.tts_text('Y يساوي y1، y2، …، ym')
        self.assertIn('وصولًا إلى', out)
        self.assertNotRegex(out, '[A-Za-z]')
        self.assertIn(' في ', T.tts_text('صورة 64x64'))
        self.assertNotRegex(T.tts_text('العدد ١٢ و 7'), r'[0-9٠-٩]')

    def test_validity(self):
        self.assertEqual(T.validity('هذه ترجمة جيدة للجملة', 'this is a good sentence'), [])
        self.assertIn('not_arabic', T.validity('< < < <', 'this is a sentence'))
        self.assertIn('too_long_for_unit', T.validity(' '.join(['كلمة'] * 30), 'two words'))


class TimingTests(unittest.TestCase):
    def test_windows_use_next_gap_and_lead(self):
        w = windows_from_timing([('a', 0.0, 2.0), ('b', 3.0, 4.0)])
        self.assertAlmostEqual(w['a'].window_s, 2.0 + 0.9)              # gap 1.0 - guard 0.1
        self.assertAlmostEqual(w['b'].window_s, 1.0 + 0.9 + 0.5)        # last unit + lead 0.5

    def test_verdict_zones(self):
        w = Window('u', 4.0, 5.0)
        self.assertEqual(duration_verdict(int(4.0 * RATE), w, RATE)['zone'], 'fits')
        self.assertEqual(duration_verdict(int(5.0 * RATE * 1.2), w, RATE)['zone'], 'speedup')
        self.assertEqual(duration_verdict(int(5.0 * RATE * 1.4), w, RATE)['zone'], 'too_long')
        self.assertEqual(duration_verdict(int(4.0 * RATE * 0.4), w, RATE)['zone'], 'too_short')

    def test_budget(self):
        lo, hi = letter_budget(Window('u', 4.0, 5.0), RATE)
        self.assertLess(lo, hi)
        self.assertEqual(hi, int(5.0 * RATE * 1.05))


class GateTests(unittest.TestCase):
    def test_mechanical(self):
        self.assertEqual(mechanical_issues('take 255 and 231', 'خذ 255 و231'), [])
        self.assertIn('numbers_missing:231', mechanical_issues('take 255 and 231', 'خذ 255'))
        self.assertIn('negation_lost', mechanical_issues("this is not correct", 'هذا صحيح'))
        self.assertEqual(mechanical_issues("this is not correct", 'هذا ليس صحيحًا'), [])

    def test_evaluate_and_allowlist(self):
        w = Window('u', 3.0, 3.0)
        text = arabic_of(int(3.0 * RATE))
        ok = evaluate(text, 'one two three four five six', w, qe=2.0, misspelled_words=set(), cfg=CFG)
        self.assertTrue(ok['accepted'], ok['reasons'])
        bad = evaluate(text, 'one two three four five six', w, qe=6.0, misspelled_words={'كتبنا'}, cfg=CFG)
        self.assertIn('qe', bad['reasons'])
        self.assertIn('spelling', bad['reasons'])
        allowed = evaluate(text, 'one two three four five six', w, qe=2.0, misspelled_words={'كتبنا'}, cfg=CFG,
                           allowlist={'كتبنا'})
        self.assertTrue(allowed['accepted'])

    def test_rank_prefers_qe_then_no_speedup(self):
        a = {'qe': 2.0, 'zone': 'speedup', 'ratio_slot': 1.0}
        b = {'qe': 2.0, 'zone': 'fits', 'ratio_slot': 0.9}
        c = {'qe': 1.5, 'zone': 'speedup', 'ratio_slot': 1.2}
        self.assertEqual(sorted([a, b, c], key=rank_key), [c, b, a])

    def test_consensus_allowlist_needs_two_families(self):
        cands = {'u': [{'generator': 'tencent_HY__plain', 'also_from': [], 'text': 'البكسلات هنا'},
                       {'generator': 'gemini__flash__iso', 'also_from': [], 'text': 'قيم البكسلات'},
                       {'generator': 'gemini__flash__iso', 'also_from': [], 'text': 'كلمةغريبة'}]}
        allow = consensus_allowlist(cands, {'البكسلات', 'كلمةغريبة'})
        self.assertIn('البكسلات', allow)
        self.assertNotIn('كلمةغريبة', allow)


class SegmentedQETests(unittest.TestCase):
    def test_long_units_scored_per_sentence_when_counts_match(self):
        from dubbing_translation.scoring import SegmentedQE

        class LenQE:
            name = 'len'
            def score(self, pairs):
                return [float(len(s.split())) for s, _ in pairs]

        q = SegmentedQE(LenQE(), min_words=6)
        src = 'One two three four. Five six seven eight nine.'
        two = 'جملة أولى. جملة ثانية.'
        one = 'جملة واحدة فقط'
        # per sentence: scores 4 and 5 weighted by 4 and 5 words -> 41/9
        self.assertAlmostEqual(q.score([(src, two)])[0], 41 / 9)
        self.assertEqual(q.score([(src, one)])[0], 9.0)                 # counts differ -> whole unit
        self.assertEqual(q.score([('short one.', two)])[0], 2.0)       # below min_words -> whole unit
        self.assertEqual(q.name, 'len+seg6')


class JudgeTests(unittest.TestCase):
    def test_pairwise_unblinds_order_and_resumes(self):
        from dubbing_translation.judge import pairwise

        class AlwaysA:
            provider, model, calls = 'fake', 'judge', 0
            def chat_json(self, system, user):
                self.calls += 1
                return {'A': {'adequacy': 90, 'fluency': 90}, 'B': {'adequacy': 50, 'fluency': 50}, 'better': 'A'}

        tasks = [{'unit_id': f'u{i}', 'source_text': 'x'} for i in range(10)]
        a = {t['unit_id']: 'نص أ' for t in tasks}
        b = {t['unit_id']: 'نص ب' for t in tasks}
        b['u0'] = 'نص أ'                                   # identical -> no call
        with tempfile.TemporaryDirectory() as tmp:
            j = AlwaysA()
            s = pairwise(tasks, a, b, j, Path(tmp) / 'j.jsonl', log=lambda *_: None)
            self.assertEqual(j.calls, 9)
            self.assertEqual(s['identical'], 1)
            self.assertEqual(s['a_wins'] + s['b_wins'], 9)      # position "A" always wins -> both systems win some
            self.assertGreater(s['a_wins'], 0)
            self.assertGreater(s['b_wins'], 0)
            s2 = pairwise(tasks, a, b, j, Path(tmp) / 'j.jsonl', log=lambda *_: None)
            self.assertEqual(j.calls, 9)                         # resumed from cache
            self.assertEqual(s, s2)


class EvaluationTests(unittest.TestCase):
    def test_automatic_metrics_and_registry(self):
        from dubbing_translation.evaluate import Evaluation
        tasks = [{'unit_id': f'u{i}', 'source_text': 'we take the values and use them now'} for i in range(4)]
        windows = {t['unit_id']: Window(t['unit_id'], 3.0, 3.0) for t in tasks}
        good, long_ = arabic_of(int(3.0 * RATE)), arabic_of(int(3.0 * RATE * 1.6))
        systems = {'sysA': {t['unit_id']: good for t in tasks},
                   'sysB': {'u0': long_, 'u1': good}}
        with tempfile.TemporaryDirectory() as tmp:
            ev = Evaluation(tmp, tasks, CFG, label='t')
            s = ev.run_automatic(windows, systems, FakeQE(default=2.0), FakeSpeller())
            self.assertEqual(s['sysA']['gate_accept_pct'], 100.0)
            self.assertEqual(s['sysB']['coverage_pct'], 50.0)
            self.assertEqual(s['sysB']['gate_accept_pct'], 25.0)
            self.assertEqual(s['sysB']['too_long_pct'], 25.0)
            reg = (Path(tmp) / 'evaluations' / 'registry.jsonl').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(reg), 2)
            self.assertEqual(json.loads(reg[0])['kind'], 'automatic')


class LectureEvaluationTests(unittest.TestCase):
    def test_evaluate_lecture_reads_drive_layout(self):
        from dubbing_translation.colab import evaluate_lecture
        good = arabic_of(int(3.0 * RATE))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            tasks = [{'unit_id': f'u{i}', 'source_text': 'we take the values and use them now',
                      'source_duration_s': 3.0} for i in range(2)]
            (run / 'experiments_v1' / 'outputs' / 'qwen3_4b').mkdir(parents=True)
            (run / 'experiments_v1' / 'tasks.json').write_text(json.dumps({'tasks': tasks}), encoding='utf-8')
            (run / 'experiments_v1' / 'outputs' / 'qwen3_4b' / 'u0.json').write_text(
                json.dumps({'natural_ar': good}, ensure_ascii=False), encoding='utf-8')
            final = run / 'translation_step' / 'runs' / 'tr_x_final_v1'
            final.mkdir(parents=True)
            (final / 'translation_final.json').write_text(json.dumps(
                {'units': [{'unit_id': t['unit_id'], 'natural_ar': good} for t in tasks]}, ensure_ascii=False),
                encoding='utf-8')
            logs = []
            s = evaluate_lecture(run, systems=('qwen3_4b', 'absent'), qe=FakeQE(default=2.0),
                                 speller=FakeSpeller(), log=logs.append)
        self.assertEqual(s['final_v1']['gate_accept_pct'], 100.0)
        self.assertEqual(s['qwen3_4b']['coverage_pct'], 50.0)
        self.assertIn('missing system folder: absent', logs)
        self.assertTrue(any(line.startswith('| final_v1 |') for line in logs[-1].splitlines()))


class BenchmarkTests(unittest.TestCase):
    ROWS = [
        {'domain': 'speech', 'segment_id': 1, 'document_id': 'd', 'is_bad_source': False,
         'source': 'one two three four five six seven eight nine ten', 'target': 'لهجة', 'original_target': 'فصحى'},
        {'domain': 'speech', 'segment_id': 2, 'document_id': 'd', 'is_bad_source': True,
         'source': 'bad source', 'target': 'x', 'original_target': 'y'},
        {'domain': 'news', 'segment_id': 3, 'document_id': 'n', 'is_bad_source': False,
         'source': 'news text', 'target': 'خبر', 'original_target': 'خبر'},
    ]

    def test_loader_uses_msa_reference_and_filters(self):
        from dubbing_translation.benchmark import as_tasks, load_wmt24pp
        items = load_wmt24pp(self.ROWS)
        self.assertEqual([i['unit_id'] for i in items], ['wmt24pp_1'])
        self.assertEqual(items[0]['reference'], 'فصحى')                 # original_target, not the dialect post-edit
        self.assertEqual(len(load_wmt24pp(self.ROWS, domains=None)), 2)
        self.assertAlmostEqual(as_tasks(items)[0]['slot_s'], round(10 / 2.6, 3))

    def test_scores_lengths_zones_and_registry(self):
        from dubbing_translation.benchmark import run_reference
        from dubbing_translation.evaluate import Evaluation

        class FakeRef:
            name = 'fake-ref'

            def score(self, triples):
                return [0.0 if h == r else 5.0 for _, h, r in triples]

        slot = 10 / 2.6
        ref = arabic_of(int(slot * RATE))
        items = [{'unit_id': 'u1', 'source_text': ' '.join(['w'] * 10), 'reference': ref}]
        systems = {'same': {'u1': ref}, 'long': {'u1': arabic_of(int(slot * RATE * 1.6))}, 'none': {}}
        with tempfile.TemporaryDirectory() as tmp:
            ev = Evaluation(tmp, items, CFG, label='b')
            s = run_reference(ev, items, systems, rate=RATE, ref_scorer=FakeRef())
            self.assertEqual(s['same']['lc10_pct'], 100.0)
            self.assertEqual(s['same']['metricx_ref_mean'], 0.0)
            self.assertEqual(s['long']['too_long_pct'], 100.0)
            self.assertEqual(s['long']['lc10_pct'], 0.0)
            self.assertEqual(s['none']['coverage_pct'], 0.0)
            self.assertIn('_human_reference', s)
            reg = [json.loads(x) for x in
                   (Path(tmp) / 'evaluations' / 'registry.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual({r['kind'] for r in reg}, {'reference'})
            self.assertEqual(len(reg), 3)                                # the human reference row is not registered
            self.assertTrue((ev.dir / 'reference.json').exists())


class BenchmarkGenerationTests(unittest.TestCase):
    def test_generate_cached_resumes_and_strips_reasoning(self):
        from dubbing_translation.benchmark import as_tasks, generate_cached
        from dubbing_translation.generators import strip_reasoning
        self.assertEqual(strip_reasoning('<think>plan</think>\n ترجمة'), 'ترجمة')
        self.assertEqual(strip_reasoning('<think>unfinished'), '')

        class Gen:
            name, calls = 'fake_gen', 0

            def generate(self, task, budget):
                Gen.calls += 1
                assert budget['min'] <= budget['max']
                return [{'text': 'نص ' + task['unit_id'], 'variant': 'plain'}]

        items = [{'unit_id': f'u{i}', 'source_text': 'a b c d e', 'reference': 'ر'} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            out = generate_cached(Gen(), as_tasks(items), tmp, rate=RATE, log=lambda *_: None)
            again = generate_cached(Gen(), as_tasks(items), tmp, rate=RATE, log=lambda *_: None)
        self.assertEqual(Gen.calls, 3)                                   # second pass reads the cache only
        self.assertEqual(out, again)
        self.assertEqual(out['u1'], 'نص u1')


class ParsingTests(unittest.TestCase):
    def test_parse_json_object(self):
        self.assertEqual(parse_json_object('```json\n{"fit": "أ"}\n```'), {'fit': 'أ'})
        self.assertEqual(parse_json_object('text {"a": 1} tail'), {'a': 1})
        self.assertIsNone(parse_json_object('nothing'))

    def test_openrouter_paid_model_refused(self):
        with self.assertRaises(ValueError):
            LLMClient('openrouter', 'some/paid-model')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.tasks = [{'unit_id': f'u{i}', 'source_text': 'we take the values and use them now', 'context_before': [],
                       'context_after': [], 'source_duration_s': 3.0} for i in range(3)]
        self.windows = {t['unit_id']: Window(t['unit_id'], 3.0, 3.0) for t in self.tasks}
        self.good = arabic_of(int(3.0 * RATE))
        self.long = arabic_of(int(3.0 * RATE * 1.6))

    def tearDown(self):
        self.tmp.cleanup()

    def test_selects_lowest_qe_accepted_and_writes_artifacts(self):
        g1 = FixedGen('mt', {'u0': self.good, 'u1': self.long, 'u2': self.good})
        g2 = FixedGen('llm', {'u0': self.good + ' جدا', 'u1': self.good, 'u2': self.long})
        qe = FakeQE({self.good: 3.0, self.good + ' جدا': 1.0})
        run_dir, final, metrics = run(self.tasks, self.windows, CFG, self.work, generators=[g1, g2], qe=qe,
                                      speller=FakeSpeller(), log=lambda *_: None)
        by = {f['unit_id']: f for f in final}
        self.assertEqual(by['u0']['selected_from'], 'llm:v')
        self.assertEqual(by['u1']['selected_from'], 'llm:v')
        self.assertEqual(metrics['accepted'], 3)
        for name in ('manifest.json', 'translation_final.json', 'metrics.json', 'candidates_scored.csv'):
            self.assertTrue((run_dir / name).exists(), name)
        self.assertEqual(len((self.work / 'registry.jsonl').read_text(encoding='utf-8').splitlines()), 1)

    def test_resume_uses_cache_not_the_model(self):
        g = FixedGen('mt', {t['unit_id']: self.good for t in self.tasks})
        run(self.tasks, self.windows, CFG, self.work, generators=[g], qe=FakeQE(), speller=FakeSpeller(),
            log=lambda *_: None)
        run(self.tasks, self.windows, CFG, self.work, generators=[g], qe=FakeQE(), speller=FakeSpeller(),
            log=lambda *_: None, run_label='second')
        self.assertEqual(g.calls, 3)
        self.assertEqual(len((self.work / 'registry.jsonl').read_text(encoding='utf-8').splitlines()), 2)

    def test_guarded_refine_rejects_meaning_loss_then_accepts(self):
        g = FixedGen('mt', {t['unit_id']: self.long for t in self.tasks})
        shorter_bad, shorter_ok = arabic_of(int(3.0 * RATE * 0.95)), arabic_of(int(3.0 * RATE)) + ' و'
        qe = FakeQE({self.long: 2.0, shorter_bad: 4.0, shorter_ok: 2.3})
        client = FakeClient([shorter_bad, shorter_ok] * 3)
        _, final, metrics = run(self.tasks[:1], self.windows, CFG, self.work, generators=[g], qe=qe,
                                speller=FakeSpeller(), refine_client=client, log=lambda *_: None)
        self.assertEqual(final[0]['status'], 'accepted_after_refine')
        self.assertEqual(final[0]['natural_ar'], shorter_ok)
        self.assertEqual(client.calls, 2)
        log = json.loads(next((self.work / 'runs').glob('*/refine_log.json')).read_text(encoding='utf-8'))
        self.assertEqual(log['u0'][0]['verdict'], 'meaning_guard')

    def test_llm_ranker_overrides_metricx_choice_among_accepted(self):
        g1 = FixedGen('mt', {'u0': self.good})
        alt = self.good + ' و'
        g2 = FixedGen('llm', {'u0': alt})
        qe = FakeQE({self.good: 1.0, alt: 2.0})                    # MetricX prefers g1

        class PreferAlt:
            provider, model, calls = 'fake', 'ranker', 0
            def chat_json(self, system, user):
                self.calls += 1
                cands = json.loads(user)['candidates']
                return {'scores': [{'id': c['id'], 'adequacy': 99 if c['arabic'] == alt else 60, 'fluency': 90}
                                   for c in cands]}

        r = PreferAlt()
        _, final, metrics = run(self.tasks[:1], self.windows, CFG, self.work, generators=[g1, g2], qe=qe,
                                speller=FakeSpeller(), rank_client=r, log=lambda *_: None)
        self.assertEqual(final[0]['natural_ar'], alt)
        self.assertEqual(metrics['rank_changed_choice'], 1)
        # cached: a second run makes no new ranker call
        run(self.tasks[:1], self.windows, CFG, self.work, generators=[g1, g2], qe=qe, speller=FakeSpeller(),
            rank_client=r, log=lambda *_: None, run_label='again')
        self.assertEqual(r.calls, 1)

    def test_unfixable_unit_is_flagged_for_review(self):
        g = FixedGen('mt', {'u0': self.long})
        _, final, metrics = run(self.tasks[:1], self.windows, CFG, self.work, generators=[g], qe=FakeQE(),
                                speller=FakeSpeller(), refine_client=FakeClient([]), log=lambda *_: None)
        self.assertEqual(final[0]['status'], 'needs_review')
        self.assertEqual(metrics['needs_review_units'], ['u0'])
        self.assertIn('duration_too_long', final[0]['reasons'])


if __name__ == '__main__':
    unittest.main()
