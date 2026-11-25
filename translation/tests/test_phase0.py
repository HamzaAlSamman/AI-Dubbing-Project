"""Phase 0 regression tests (audit P0-1 .. P0-5 and the promoted data-integrity items).
Offline: no model, no network. Run from the repo root:
    python -m unittest discover -s translation/tests -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TR = Path(__file__).resolve().parents[1]


def _notebook_code(name):
    nb = json.loads((TR / 'notebooks' / name).read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


# ---------------------------------------------------------------- P0-1 Colab environment
class ColabEnvironmentTests(unittest.TestCase):
    def test_requirements_file_pins_every_generator_dependency(self):
        req = (TR / 'requirements-colab.txt').read_text(encoding='utf-8')
        names = {line.split('==')[0].split('>=')[0].split('<')[0].strip().lower()
                 for line in req.splitlines() if line.strip() and not line.startswith('#')}
        for pkg in ('transformers', 'accelerate', 'bitsandbytes', 'google-genai', 'sentencepiece', 'sacrebleu'):
            self.assertIn(pkg, names)
        for line in req.splitlines():
            if line.strip() and not line.startswith('#'):
                self.assertTrue(any(op in line for op in ('==', '>=')), f'unpinned: {line}')

    def test_v2_notebook_installs_requirements_and_runs_preflight_before_models(self):
        cells = _notebook_code('02_translation_v2_colab.ipynb')
        setup = next(c for c in cells if 'drive.mount' in c)
        self.assertIn('requirements-colab.txt', setup)
        pre = [i for i, c in enumerate(cells) if 'preflight' in c]
        run = [i for i, c in enumerate(cells) if 'trc.start(' in c]
        self.assertTrue(pre and run and pre[0] < run[0])

    def test_preflight_reports_missing_package_without_loading_models(self):
        from dubbing_translation.preflight import preflight
        rep = preflight(need={'gpu': False, 'packages': ['surely_not_installed_pkg_xyz'], 'env': [], 'binaries': []},
                        raise_on_error=False)
        self.assertFalse(rep['ok'])
        self.assertIn('surely_not_installed_pkg_xyz', ' '.join(rep['errors']))

    def test_preflight_names_needed_packages_from_the_generator_plan(self):
        from dubbing_translation.preflight import requirements_for
        need = requirements_for(use_hymt=True, hf_models=[('tencent/Hy-MT2-7B', 'hy_mt2_terms', 1)], use_llm=True)
        self.assertIn('bitsandbytes', need['packages'])
        self.assertIn('google.genai', need['packages'])
        self.assertIn('GEMINI_API_KEY', need['env'])
        self.assertTrue(need['gpu'])



# ---------------------------------------------------------------- P0-2 API failures, cache, degraded runs
class _NoSleep:
    def __enter__(self):
        import dubbing_translation.generators as g
        self.g, self.orig = g, g.time.sleep
        g.time.sleep = lambda *_: None
        return self

    def __exit__(self, *a):
        self.g.time.sleep = self.orig


def _raise(exc):
    def f(*_a, **_k):
        raise exc
    return f


class ApiFailureTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from dubbing_translation.timing import Window
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.tasks = [{'unit_id': f'u{i}', 'source_text': 'we take the values and use them now', 'context_before': [],
                       'context_after': [], 'source_duration_s': 3.0} for i in range(2)]
        self.windows = {t['unit_id']: Window(t['unit_id'], 3.0, 3.0) for t in self.tasks}

    def tearDown(self):
        self.tmp.cleanup()

    def test_client_raises_transient_error_instead_of_returning_none(self):
        from dubbing_translation.generators import LLMClient, TransientLLMError
        c = LLMClient('gemini', 'm', min_interval_s=0, max_attempts=2)
        c._once = _raise(RuntimeError('503 UNAVAILABLE. high demand'))
        with _NoSleep(), self.assertRaises(TransientLLMError):
            c.chat_json('s', 'u')
        self.assertEqual(c.calls, 2)                                  # bounded retries

    def test_client_unparsable_output_is_invalid_not_an_empty_result(self):
        from dubbing_translation.generators import InvalidLLMOutput, LLMClient
        c = LLMClient('gemini', 'm', min_interval_s=0, max_attempts=5)
        c._once = lambda s, u: 'not json at all'
        with _NoSleep(), self.assertRaises(InvalidLLMOutput):
            c.chat_json('s', 'u')
        self.assertEqual(c.calls, 2)                                  # bounded: max_invalid answers

    def test_client_valid_empty_json_is_a_result(self):
        from dubbing_translation.generators import LLMClient
        c = LLMClient('gemini', 'm', min_interval_s=0)
        c._once = lambda s, u: '{}'
        with _NoSleep():
            self.assertEqual(c.chat_json('s', 'u'), {})

    def test_client_authentication_failure_is_typed_and_not_retried(self):
        from dubbing_translation.generators import AuthenticationError, LLMClient
        c = LLMClient('gemini', 'm', min_interval_s=0, max_attempts=3)
        c._once = _raise(ValueError('400 API_KEY_INVALID: API key not valid'))
        with _NoSleep(), self.assertRaises(AuthenticationError):
            c.chat_json('s', 'u')
        self.assertEqual(c.calls, 1)
        import os
        saved = os.environ.pop('GEMINI_API_KEY', None)
        try:
            with self.assertRaises(AuthenticationError):              # missing key: not a per-unit error
                LLMClient('gemini', 'm')._gemini()
        finally:
            if saved is not None:
                os.environ['GEMINI_API_KEY'] = saved

    def test_client_quota_and_request_cap(self):
        from dubbing_translation.generators import CallBudget, CallLimitReached, LLMClient, QuotaExhausted
        c = LLMClient('gemini', 'm', min_interval_s=0)
        c._once = _raise(RuntimeError('429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel'))
        with _NoSleep(), self.assertRaises(QuotaExhausted):
            c.chat_json('s', 'u')
        budget = CallBudget(2)
        a, b = LLMClient('gemini', 'a', min_interval_s=0, budget=budget), LLMClient('gemini', 'b', min_interval_s=0,
                                                                                    budget=budget)
        a._once = b._once = lambda s, u: '{"x": 1}'
        a.chat_json('s', 'u')
        b.chat_json('s', 'u')
        with self.assertRaises(CallLimitReached):                     # shared by every client of the run
            a.chat_json('s', 'u')
        self.assertEqual((budget.used, a.calls + b.calls), (2, 2))

    def test_client_permanent_error_is_typed(self):
        from dubbing_translation.generators import LLMClient, PermanentLLMError
        c = LLMClient('gemini', 'm', min_interval_s=0, max_attempts=3)
        c._once = _raise(ValueError('400 INVALID_ARGUMENT'))
        with _NoSleep(), self.assertRaises(PermanentLLMError):
            c.chat_json('s', 'u')
        self.assertEqual(c.calls, 1)

    def _gen(self, behaviour):
        from dubbing_translation.generators import TransientLLMError

        class Gen:
            uses_budget, name = False, 'flaky'

            def __init__(self):
                self.calls = 0

            def generate(self, task, budget):
                self.calls += 1
                if behaviour(task['unit_id']) == 'transient':
                    raise TransientLLMError('503 after retries')
                return [{'text': 'كتبنا ' * 6, 'variant': 'v'}]
        return Gen()

    def test_transient_failure_is_not_cached_and_marks_the_run_degraded(self):
        from dubbing_translation.io import load_config
        from dubbing_translation.pipeline import run
        from test_translation_step import FakeQE, FakeSpeller
        cfg = load_config()
        gen = self._gen(lambda u: 'transient' if u == 'u0' else 'ok')
        run_dir, final, metrics = run(self.tasks, self.windows, cfg, self.work, generators=[gen], qe=FakeQE(),
                                      speller=FakeSpeller(), log=lambda *_: None)
        self.assertFalse((self.work / 'cache' / 'candidates' / 'flaky' / 'u0.json').exists())
        self.assertTrue((self.work / 'cache' / 'candidates' / 'flaky' / 'u1.json').exists())
        self.assertTrue(metrics['degraded'])
        self.assertEqual(metrics['deferred_units'], {'flaky': ['u0']})
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(manifest['status']['degraded'])
        # resume: the deferred unit is generated on the next run, the cached one is not regenerated
        gen2 = self._gen(lambda u: 'ok')
        _, _, metrics2 = run(self.tasks, self.windows, cfg, self.work, generators=[gen2], qe=FakeQE(),
                             speller=FakeSpeller(), log=lambda *_: None, run_label='resume')
        self.assertEqual(gen2.calls, 1)
        self.assertFalse(metrics2['degraded'])

    def test_consecutive_transient_failures_stop_the_generator(self):
        from dubbing_translation.io import load_config
        from dubbing_translation.pipeline import run
        from dubbing_translation.timing import Window
        from test_translation_step import FakeQE, FakeSpeller
        tasks = [{'unit_id': f'u{i}', 'source_text': 'we take the values', 'context_before': [], 'context_after': [],
                  'source_duration_s': 3.0} for i in range(6)]
        windows = {t['unit_id']: Window(t['unit_id'], 3.0, 3.0) for t in tasks}
        gen = self._gen(lambda u: 'transient')
        _, _, metrics = run(tasks, windows, load_config(), self.work, generators=[gen], qe=FakeQE(),
                            speller=FakeSpeller(), log=lambda *_: None)
        self.assertEqual(gen.calls, 3)                                # max consecutive failures (default 3)
        self.assertEqual(len(metrics['deferred_units']['flaky']), 6)
        self.assertTrue(metrics['degraded'])

    def test_permanent_error_is_recorded_once_and_marks_the_run_degraded(self):
        from dubbing_translation.generators import PermanentLLMError
        from dubbing_translation.io import load_config
        from dubbing_translation.pipeline import run
        from test_translation_step import FakeQE, FakeSpeller

        class Gen:
            uses_budget, name, calls = False, 'blocked', 0

            def generate(self, task, budget):
                self.calls += 1
                raise PermanentLLMError('400 blocked')

        gen = Gen()
        _, _, metrics = run(self.tasks[:1], self.windows, load_config(), self.work, generators=[gen], qe=FakeQE(),
                            speller=FakeSpeller(), log=lambda *_: None)
        folder = self.work / 'cache' / 'candidates' / 'blocked'
        self.assertFalse((folder / 'u0.json').exists())               # a failure is never a result
        rec = json.loads((folder / 'u0.failure.json').read_text(encoding='utf-8'))
        self.assertEqual((rec['kind'], rec['attempts']), ('permanent', 1))
        self.assertTrue(metrics['degraded'])
        _, _, m2 = run(self.tasks[:1], self.windows, load_config(), self.work, generators=[gen], qe=FakeQE(),
                       speller=FakeSpeller(), log=lambda *_: None, run_label='again')
        self.assertEqual(gen.calls, 1)                                # not retried forever
        self.assertTrue(m2['degraded'])

    def test_quota_in_ranker_marks_the_run_degraded(self):
        from dubbing_translation.generators import QuotaExhausted
        from dubbing_translation.io import load_config
        from dubbing_translation.pipeline import run
        from test_translation_step import FakeQE, FakeSpeller, FixedGen, arabic_of

        class NoQuota:
            provider, model, calls = 'fake', 'ranker', 0

            def chat_json(self, system, user):
                raise QuotaExhausted('per day')

        cfg = load_config()
        good = arabic_of(int(3.0 * cfg['voice']['letters_per_second']))
        gens = [FixedGen('a', {'u0': good}), FixedGen('b', {'u0': good + ' و'})]
        _, _, metrics = run(self.tasks[:1], self.windows, cfg, self.work, generators=gens, qe=FakeQE(),
                            speller=FakeSpeller(), rank_client=NoQuota(), log=lambda *_: None)
        self.assertTrue(metrics['degraded'])
        self.assertIn('rank', ' '.join(metrics['degraded_reasons']))

    def test_comparison_refuses_a_degraded_run(self):
        from dubbing_translation.colab import compare_v1_v2
        from dubbing_translation.io import load_config
        from dubbing_translation.pipeline import run
        from test_translation_step import FakeQE, FakeSpeller
        cfg = load_config()
        work = self.work / 'lesson' / 'translation_step'
        ok = self._gen(lambda u: 'ok')
        run(self.tasks, self.windows, cfg, work, generators=[ok], qe=FakeQE(), speller=FakeSpeller(),
            log=lambda *_: None, run_label='final_v1')
        flaky = self._gen(lambda u: 'transient')
        flaky.name = 'flaky2'
        v2_dir, _, _ = run(self.tasks, self.windows, cfg, work, generators=[flaky], qe=FakeQE(),
                           speller=FakeSpeller(), log=lambda *_: None, run_label='v2')
        with self.assertRaises(ValueError):
            compare_v1_v2(self.work / 'lesson', v2_dir)

    def test_benchmark_cache_skips_transient_failures(self):
        from dubbing_translation.benchmark import generate_cached
        gen = self._gen(lambda u: 'transient' if u == 'u0' else 'ok')
        report = {}
        out = generate_cached(gen, [dict(t, slot_s=3.0) for t in self.tasks], self.work, rate=10.0,
                              log=lambda *_: None, probe=0, report=report)
        lines = (self.work / 'flaky.jsonl').read_text(encoding='utf-8').splitlines()
        self.assertEqual([json.loads(x)['unit_id'] for x in lines], ['u1'])
        self.assertEqual(report['deferred'], ['u0'])
        self.assertNotIn('u0', out)



    def _flaky_gen(self, name, plan):
        """Generator whose answer per call follows `plan` (list of 'ok' | 'invalid' | 'permanent')."""
        from dubbing_translation.generators import InvalidLLMOutput, PermanentLLMError

        class Gen:
            uses_budget = False

            def __init__(self):
                self.name, self.calls = name, 0

            def generate(self, task, budget):
                step = plan[min(self.calls, len(plan) - 1)]
                self.calls += 1
                if step == 'invalid':
                    raise InvalidLLMOutput('no JSON')
                if step == 'permanent':
                    raise PermanentLLMError('400 blocked')
                return [{'text': 'كتبنا ' * 6, 'variant': 'v'}]
        return Gen()

    def _run(self, gens, cfg=None, label=''):
        from dubbing_translation.io import load_config
        from dubbing_translation.pipeline import run
        from test_translation_step import FakeQE, FakeSpeller
        return run(self.tasks[:1], self.windows, cfg or load_config(), self.work, generators=gens, qe=FakeQE(),
                   speller=FakeSpeller(), log=lambda *_: None, run_label=label)

    def test_invalid_output_is_not_cached_and_retried_a_bounded_number_of_times(self):
        folder = self.work / 'cache' / 'candidates' / 'inv'
        gen = self._flaky_gen('inv', ['invalid'])
        for i in range(5):
            _, final, metrics = self._run([gen], label=f'r{i}')
            self.assertFalse((folder / 'u0.json').exists())
            self.assertTrue(metrics['degraded'])
            self.assertIn('invalid_output', ' '.join(metrics['degraded_reasons']))
        self.assertEqual(gen.calls, 3)                                # models.max_unit_attempts (default 3)
        self.assertEqual(json.loads((folder / 'u0.failure.json').read_text(encoding='utf-8'))['attempts'], 3)

    def test_retry_errors_after_the_cause_is_fixed_keeps_good_results(self):
        from dubbing_translation.io import load_config
        good = self._flaky_gen('good', ['ok'])
        bad = self._flaky_gen('bad', ['permanent', 'ok'])
        self._run([good, bad])
        good_rec = (self.work / 'cache' / 'candidates' / 'good' / 'u0.json').read_bytes()
        self._run([good, bad], label='again')
        self.assertEqual(bad.calls, 1)                                # blocked until asked
        cfg = load_config()
        cfg['models']['retry_errors'] = True
        _, final, metrics = self._run([good, bad], cfg, label='retry')
        self.assertEqual((good.calls, bad.calls), (1, 2))             # the good result is not regenerated
        self.assertEqual((self.work / 'cache' / 'candidates' / 'good' / 'u0.json').read_bytes(), good_rec)
        self.assertTrue((self.work / 'cache' / 'candidates' / 'bad' / 'u0.json').exists())
        ledger = json.loads((self.work / 'cache' / 'candidates' / 'bad' / 'u0.failure.json').read_text(encoding='utf-8'))
        self.assertIsNotNone(ledger['resolved_at'])                   # the history of the failure is kept
        self.assertFalse(metrics['degraded'])

    def test_authentication_failure_stops_the_generator_and_caches_nothing(self):
        from dubbing_translation.generators import AuthenticationError

        class NoKey:
            uses_budget, name, calls = False, 'nokey', 0

            def generate(self, task, budget):
                self.calls += 1
                raise AuthenticationError('GEMINI_API_KEY is not loaded')
        gen = NoKey()
        _, _, metrics = self._run([gen])
        folder = self.work / 'cache' / 'candidates' / 'nokey'
        self.assertFalse(folder.exists() and any(folder.iterdir()))
        self.assertIn('authentication', ' '.join(metrics['degraded_reasons']))
        self.assertEqual(metrics['deferred_units'], {'nokey': ['u0']})

    def test_invalid_json_from_an_llm_role_is_not_cached(self):
        from dubbing_translation.generators import InvalidLLMOutput
        from dubbing_translation.io import load_config
        from test_translation_step import FixedGen, arabic_of
        cfg = load_config()
        good = arabic_of(int(3.0 * cfg['voice']['letters_per_second']))

        class BadJSON:
            provider, model, calls = 'fake', 'ranker', 0

            def chat_json(self, system, user):
                self.calls += 1
                raise InvalidLLMOutput('no JSON')
        gens = [FixedGen('a', {'u0': good}), FixedGen('b', {'u0': good + ' و'})]
        from dubbing_translation.pipeline import run
        from test_translation_step import FakeQE, FakeSpeller
        _, final, metrics = run(self.tasks[:1], self.windows, cfg, self.work, generators=gens, qe=FakeQE(),
                                speller=FakeSpeller(), rank_client=BadJSON(), log=lambda *_: None)
        self.assertEqual(final[0]['status'], 'accepted')              # the MetricX choice is kept
        self.assertFalse((self.work / 'cache' / 'rank').exists())
        self.assertTrue(metrics['degraded'])
        self.assertIn('invalid_output:rank', ' '.join(metrics['degraded_reasons']))

    def test_quality_rejection_is_not_infrastructure_degradation(self):
        from test_translation_step import FixedGen
        _, final, metrics = self._run([FixedGen('short', {'u0': 'كتب'})])
        self.assertEqual(final[0]['status'], 'needs_review')          # failed the quality gate
        self.assertFalse(metrics['degraded'])                         # but the run is complete
        self.assertEqual(metrics['infrastructure'], 'complete')
        self.assertEqual(metrics['quality_gate']['failed_gate'], 1)


# ---------------------------------------------------------------- P0-3 source numbers are never rewritten
class SourceNumberTests(unittest.TestCase):
    CASES = [
        'Steps 1, 2, 3, 4 are shown on the slide.',
        'The point (1, 2) and the point (3, 4) are close.',
        'Pi is about 3.14 and e is about 2.718.',
        'We set y = 2x + 1, then x = -5.',
        'Install version 1.2.3 or 10.0.1 of the library.',
        'which in this case turns out to be 1, 2, 2, 8, 8.',
        'The dates 2026-11-25 and 25/11/2026 are the same day.',
        'Use 1,000,000 samples, 50% of them for training.',
    ]

    def _prepare(self, text):
        from dubbing_translation.io import V2_CONFIG, load_config
        from dubbing_translation.segment import prepare_sources
        from dubbing_translation.timing import Window
        tasks = [{'unit_id': 'u0', 'source_text': text}]
        return prepare_sources(tasks, {'u0': Window('u0', 3.0, 3.0)}, load_config(V2_CONFIG))

    def test_source_text_is_never_changed(self):
        for text in self.CASES:
            tasks, _, rep = self._prepare(text)
            self.assertEqual(tasks[0]['source_text'], text)
            self.assertNotIn('source_text_asr', tasks[0])

    def test_suspected_spelled_number_is_only_flagged(self):
        _, _, rep = self._prepare('which in this case turns out to be 1, 2, 2, 8, 8.')
        self.assertEqual(rep['spelled_digit_flags']['u0'], ['1, 2, 2, 8, 8'])
        _, _, rep = self._prepare('Pi is about 3.14 and e is about 2.718.')
        self.assertEqual(rep['spelled_digit_flags'], {})

    def test_auto_fix_option_is_refused(self):
        from dubbing_translation.segment import prepare_sources
        from dubbing_translation.timing import Window
        with self.assertRaises(ValueError):
            prepare_sources([{'unit_id': 'u0', 'source_text': 'x'}], {'u0': Window('u0', 1.0, 1.0)},
                            {'segmentation': {'fix_spelled_digits': True}})



# ---------------------------------------------------------------- promoted D: only safe merges
class SafeMergeTests(unittest.TestCase):
    def _merge(self, a, b, gap=0.2):
        from dubbing_translation.segment import merge_fragments
        from dubbing_translation.timing import Window
        tasks = [{'unit_id': 'a', 'source_text': a}, {'unit_id': 'b', 'source_text': b}]
        windows = {'a': Window('a', 2.0, 2.0 + gap), 'b': Window('b', 2.0, 2.0)}
        timings = {'a': (10.0, 12.0), 'b': (12.0 + gap, 14.0 + gap)}
        return merge_fragments(tasks, windows, timings=timings)

    def test_clear_continuation_is_merged_without_losing_words(self):
        a, b = 'So if you take X1 and', 'you know, put that as the first column.'
        tasks, windows, merges, suspects = self._merge(a, b)
        self.assertEqual([t['unit_id'] for t in tasks], ['a+b'])
        self.assertEqual(tasks[0]['source_text'].split(), a.split() + b.split())
        self.assertEqual(suspects, [])
        tasks, _, _, _ = self._merge('until we list the red pixels 255, 134,', '255, 134, and so on.')
        self.assertEqual([t['unit_id'] for t in tasks], ['a+b'])

    def test_sentence_end_followed_by_lower_case_brand_is_not_merged(self):
        tasks, _, merges, suspects = self._merge('So that is the idea.', 'iPhone apps use this pattern.')
        self.assertEqual([t['unit_id'] for t in tasks], ['a', 'b'])
        self.assertEqual(merges, [])
        self.assertEqual(suspects[0]['units'], ['a', 'b'])

    def test_one_signal_only_is_a_suspect_not_a_merge(self):
        for a, b in (('if this image is a 64 by 64 image,', 'The total dimension of x is 12288.'),
                     ('stack the examples, x1 transpose down to xm.', 'transpose, but it turns out that this helps.')):
            tasks, _, merges, suspects = self._merge(a, b)
            self.assertEqual(len(tasks), 2, a)
            self.assertEqual(len(suspects), 1, a)

    def test_long_pause_blocks_the_merge(self):
        tasks, _, merges, suspects = self._merge('So if you take X1 and', 'you know, put that first.', gap=2.5)
        self.assertEqual(len(tasks), 2)
        self.assertIn('pause', suspects[0]['reason'])



# ---------------------------------------------------------------- P0-4 course terminology memory
class _Agent:
    """Fake terminology agent: returns the terms given for each call, records the prior it received."""
    provider, model = 'fake', 'terminology'

    def __init__(self, *replies):
        self.replies, self.priors, self.calls = list(replies), [], 0

    def chat_json(self, system, user):
        self.calls += 1
        self.priors.append(json.loads(user).get('approved_terms_from_earlier_videos'))
        return {'subject': 'course', 'terms': self.replies.pop(0) if self.replies else []}


class CourseMemoryV2Tests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from dubbing_translation.io import V2_CONFIG, load_config
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mem = self.root / 'course' / 'glossary.json'
        self.cfg = load_config(V2_CONFIG)
        self.cfg['polish']['enabled'] = False

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, video, agent, *, gens=None, text='y is the label of this image', snapshot=None, label=''):
        from dubbing_translation.pipeline import run
        from dubbing_translation.timing import Window
        from test_translation_step import FakeQE, FakeSpeller, FixedGen, arabic_of
        n = int(3.0 * self.cfg['voice']['letters_per_second'])
        tasks = [{'unit_id': 'u0', 'source_text': text, 'context_before': [], 'context_after': [],
                  'source_duration_s': 3.0}]
        gens = gens or [FixedGen('mt', {'u0': 'الوسم ' + arabic_of(n - 5)})]
        return run(tasks, {'u0': Window('u0', 3.0, 3.0)}, self.cfg, self.root / video / 'translation_step',
                   generators=gens, qe=FakeQE(), speller=FakeSpeller(), refine_client=agent, log=lambda *_: None,
                   course_memory=self.mem if snapshot is None else None, course_memory_snapshot=snapshot,
                   video_id=video, run_label=label)

    def _data(self):
        return json.loads(self.mem.read_text(encoding='utf-8'))

    def test_proposed_term_is_a_suggestion_not_a_rejection(self):
        _, final, metrics = self._run('v1', _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        self.assertEqual(final[0]['status'], 'accepted')               # 'الوسم' is not rejected
        self.assertIn('label', final[0]['language']['terms'])           # but reported
        self.assertEqual(self._data()['terms']['label']['status'], 'proposed')

    def test_degraded_or_failed_run_does_not_change_the_memory(self):
        from dubbing_translation.generators import TransientLLMError

        class Down:
            uses_budget, name = False, 'down'

            def generate(self, task, budget):
                raise TransientLLMError('503')

        _, _, metrics = self._run('v1', _Agent([{'en': ['label'], 'ar': 'تسمية'}]), gens=[Down()])
        self.assertTrue(metrics['degraded'])
        self.assertFalse(self.mem.exists())

    def test_rerunning_a_video_is_recorded_once(self):
        self._run('v1', _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        self._run('v1', _Agent([{'en': ['label'], 'ar': 'تسمية'}]), label='again')
        d = self._data()
        self.assertEqual(list(d['videos']), ['v1'])
        self.assertEqual(d['terms']['label']['renderings']['تسمية']['videos'], ['v1'])
        self.assertEqual(d['terms']['label']['status'], 'proposed')     # one video is not a verification

    def test_independent_agreement_verifies_but_is_only_a_strong_preference(self):
        self._run('v1', _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        a2 = _Agent([{'en': ['label'], 'ar': 'تسمية'}])
        self._run('v2', a2)
        self.assertIsNone(a2.priors[0])                                  # proposed terms are not shown: independent
        self.assertEqual(self._data()['terms']['label']['status'], 'auto_verified')
        _, final, _ = self._run('v3', _Agent([]))
        self.assertEqual(final[0]['status'], 'accepted')                # agreement is not proof: no rejection
        self.assertNotIn('term:label', final[0]['reasons'])
        self.assertIn('label', final[0]['language']['terms'])           # but reported as missing

    def test_auto_verified_can_reject_only_when_the_config_asks(self):
        self._run('v1', _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        self._run('v2', _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        self.cfg['language']['reject_statuses'] = ['approved', 'auto_verified']
        _, final, _ = self._run('v3', _Agent([]))
        self.assertIn('term:label', final[0]['reasons'])

    def test_a_rejected_rendering_is_dropped_from_later_proposals(self):
        from dubbing_translation.language import CourseMemory
        self._run('v1', _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        CourseMemory(self.mem).set_status('label', 'rejected', rendering='تسمية', by='reviewer')
        run_dir, final, _ = self._run('v2', _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        glossary = json.loads((run_dir / 'glossary.json').read_text(encoding='utf-8'))
        self.assertNotIn('label', [t['id'] for t in glossary['terms']])
        self.assertEqual(self._data()['terms']['label']['status'], 'rejected')   # a person's decision is kept

    def test_invalid_glossary_reply_is_not_cached(self):
        class Broken(_Agent):
            def chat_json(self, system, user):
                self.calls += 1
                from dubbing_translation.generators import InvalidLLMOutput
                raise InvalidLLMOutput('no JSON')
        _, _, metrics = self._run('v1', Broken())
        self.assertFalse((self.root / 'v1' / 'translation_step' / 'cache' / 'glossary').exists())
        self.assertTrue(metrics['degraded'])
        self.assertFalse(self.mem.exists())                             # a degraded run writes no memory

    def test_conflict_is_recorded_and_never_enforced(self):
        self._run('v1', _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        self._run('v2', _Agent([{'en': ['label'], 'ar': 'وسم'}]))
        d = self._data()
        self.assertEqual(d['terms']['label']['status'], 'conflicting')
        self.assertEqual(sorted(d['terms']['label']['renderings']), ['تسمية', 'وسم'])
        self.assertTrue(any(c['term'] == 'label' for c in d['conflicts']))
        _, final, _ = self._run('v3', _Agent([]))
        self.assertNotIn('term:label', final[0]['reasons'])

    def test_success_never_promotes_to_approved_but_a_person_can(self):
        from dubbing_translation.language import CourseMemory
        for v in ('v1', 'v2', 'v3'):
            self._run(v, _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        self.assertEqual(self._data()['terms']['label']['status'], 'auto_verified')
        CourseMemory(self.mem).set_status('label', 'approved', rendering='تسمية', by='reviewer')
        self._run('v4', _Agent([{'en': ['label'], 'ar': 'وسم'}]))
        d = self._data()
        self.assertEqual(d['terms']['label']['status'], 'approved')      # a human decision is kept
        self.assertTrue(any(c['term'] == 'label' for c in d['conflicts']))

    def test_term_keeps_source_context_and_version(self):
        self._run('v1', _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        src = self._data()['terms']['label']['renderings']['تسمية']['sources'][0]
        self.assertEqual(src['video'], 'v1')
        self.assertEqual(src['units'], ['u0'])
        self.assertIn('label', src['context'])
        self.assertTrue(src['glossary_version'].startswith('glossary_auto'))

    def test_snapshot_is_saved_and_replays_the_same_prior_without_writing(self):
        self._run('v1', _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        self._run('v2', _Agent([{'en': ['label'], 'ar': 'تسمية'}]))
        a3 = _Agent([])
        run_dir, _, _ = self._run('v3', a3)
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        snap = run_dir / 'course_memory_snapshot.json'
        self.assertTrue(snap.exists())
        self.assertEqual(manifest['language']['course_memory_snapshot_sha256'],
                         __import__('hashlib').sha256(snap.read_bytes()).hexdigest())
        self.assertEqual(a3.priors[0], [{'en': ['label'], 'ar': 'تسمية'}])  # verified term given to the agent
        before = self.mem.read_bytes()
        a4 = _Agent([])
        replay_dir, _, _ = self._run('v3', a4, snapshot=snap, label='replay')
        # same prior -> same glossary cache key -> the agent is not called again, same glossary
        self.assertEqual(a4.calls, 0)
        self.assertEqual((replay_dir / 'glossary.json').read_text(encoding='utf-8'),
                         (run_dir / 'glossary.json').read_text(encoding='utf-8'))
        self.assertEqual(self.mem.read_bytes(), before)                 # replay does not write

    def test_interleaved_writers_keep_both_contributions(self):
        from dubbing_translation.language import CourseMemory
        m1, m2 = CourseMemory(self.mem), CourseMemory(self.mem)       # both loaded before either writes
        m1.record_video('v1', [{'en': ['label'], 'ar': 'تسمية'}], meta={'glossary_version': 'g'})
        m2.record_video('v2', [{'en': ['matrix'], 'ar': 'مصفوفة'}], meta={'glossary_version': 'g'})
        d = self._data()
        self.assertEqual(sorted(d['videos']), ['v1', 'v2'])
        self.assertEqual(sorted(d['terms']), ['label', 'matrix'])
        self.assertFalse(self.mem.with_suffix('.lock').exists())



# ---------------------------------------------------------------- P0-5 spoken text (v2)
class SpokenTextTests(unittest.TestCase):
    def say(self, text):
        from dubbing_translation.text import tts_text
        return tts_text(text, version='v2')

    def test_programming_words_use_the_pronunciation_lexicon(self):
        self.assertIn('حلقة فور', self.say('نستخدم حلقة for لتكرار الأمثلة'))
        self.assertNotIn('إف أو آر', self.say('نستخدم حلقة for لتكرار الأمثلة'))
        self.assertIn('نمباي دوت دوت', self.say('نستدعي np.dot'))         # the function is named dot
        self.assertIn('إكس دوت شيب', self.say('الأمر x.shape'))
        self.assertIn('جي بي يو', self.say('على GPU'))
        self.assertIn('إيه بي آي', self.say('عبر API'))
        self.assertIn('ريلو', self.say('الدالة ReLU'))

    def test_numbers_are_read_by_one_deterministic_reader(self):
        cases = {'12288': 'اثنا عشر ألفًا ومئتان وثمانية وثمانون', '2026': 'ألفان وستة وعشرون',
                 '64': 'أربعة وستون', '255': 'مئتان وخمسة وخمسون', '1000': 'ألف', '3000': 'ثلاثة آلاف',
                 '1,000,000': 'مليون', '0': 'صفر', '11': 'أحد عشر', '100000': 'مئة ألف'}
        for digits, words in cases.items():
            self.assertEqual(self.say(digits), words, digits)

    def test_negative_decimal_percent_date_equation(self):
        self.assertEqual(self.say('-5'), 'سالب خمسة')
        self.assertEqual(self.say('3.14'), 'ثلاثة فاصلة واحد أربعة')
        self.assertEqual(self.say('50%'), 'خمسون بالمئة')
        out = self.say('25/11/2026')
        self.assertNotRegex(out, r'\d')
        self.assertIn('ألفان وستة وعشرون', out)
        self.assertEqual(self.say('y = 2x + 1'), 'واي يساوي اثنان إكس زائد واحد')
        self.assertEqual(self.say('y = x - 5'), 'واي يساوي إكس ناقص خمسة')
        self.assertEqual(self.say('x = -5'), 'إكس يساوي سالب خمسة')

    def test_letter_case_variables_are_spoken(self):
        out = self.say('X الكبيرة و x الصغيرة و nx')
        self.assertEqual(out.count('إكس'), 3)
        self.assertNotRegex(out, '[A-Za-z]')

    def test_result_does_not_depend_on_num2words(self):
        import types
        fake = types.ModuleType('num2words')
        fake.num2words = lambda *a, **k: 'WRONG'
        before = self.say('العدد 12288 و 7')
        sys.modules['num2words'] = fake
        try:
            self.assertEqual(self.say('العدد 12288 و 7'), before)
        finally:
            del sys.modules['num2words']

    def test_written_text_is_not_changed_and_v1_stays_the_default(self):
        from dubbing_translation import text as T
        src = 'نستخدم حلقة for و 12288'
        self.assertEqual(T.tts_text(src), T.tts_text(src, version='v1'))   # V1 results stay reproducible
        self.assertEqual(src, 'نستخدم حلقة for و 12288')



# ---------------------------------------------------------------- promoted C: source provenance
class SourceProvenanceTests(unittest.TestCase):
    def test_manifest_has_source_and_prepared_hashes(self):
        import tempfile
        from dubbing_translation.common import sha256_json
        from dubbing_translation.io import V2_CONFIG, load_config
        from dubbing_translation.pipeline import run
        from dubbing_translation.timing import Window
        from test_translation_step import FakeQE, FakeSpeller, FixedGen
        cfg = load_config(V2_CONFIG)
        cfg['language'] = {}                                           # no LLM needed for this check
        tasks = [{'unit_id': 'a', 'source_text': 'So if you take X1 and'},
                 {'unit_id': 'b', 'source_text': 'you know, put that first.'}]
        windows = {'a': Window('a', 2.0, 2.0), 'b': Window('b', 2.0, 2.0)}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, final, _ = run(tasks, windows, cfg, tmp, generators=[FixedGen('g', {})], qe=FakeQE(),
                                    speller=FakeSpeller(), log=lambda *_: None)
            m = json.loads((Path(run_dir) / 'manifest.json').read_text(encoding='utf-8'))
        src = sha256_json([{k: t[k] for k in ('unit_id', 'source_text')} for t in tasks])
        self.assertEqual(m['source_tasks_sha256'], src)
        self.assertNotEqual(m['prepared_tasks_sha256'], src)           # the two units were merged
        self.assertEqual(m['tasks_sha256'], m['prepared_tasks_sha256'])
        self.assertEqual([f['unit_id'] for f in final], ['a+b'])



# ---------------------------------------------------------------- promoted B: prompt cache integrity
class PromptCacheIntegrityTests(unittest.TestCase):
    def test_prompt_texts_match_their_frozen_versions(self):
        """A prompt text changed without a new version string fails here: bump the version, then add
        the new hash to configs/prompt_hashes.json (old versions stay, old caches stay valid)."""
        from dubbing_translation.prompts import PROMPT_REGISTRY, prompt_sha
        frozen = json.loads((TR / 'configs' / 'prompt_hashes.json').read_text(encoding='utf-8'))
        for version, text in PROMPT_REGISTRY.items():
            self.assertIn(version, frozen, f'{version} is not frozen')
            self.assertEqual(prompt_sha(text), frozen[version], f'{version}: text changed without a new version')

    def test_v2_generator_names_carry_the_prompt_hash(self):
        from dubbing_translation.generators import HFChatMT, HyMT, IsometricLLM
        from dubbing_translation.prompts import ISOMETRIC_V2_SYSTEM, prompt_sha

        class C:
            provider, model = 'gemini', 'm'
        g = IsometricLLM(C(), language_aware=True)
        self.assertTrue(g.name.endswith('__p' + prompt_sha(ISOMETRIC_V2_SYSTEM)[:8]))
        self.assertEqual(IsometricLLM(C()).name, 'gemini__m__isometric_v1')       # V1 name unchanged
        self.assertIn('__p', HyMT(language_aware=True).name)
        self.assertEqual(HyMT().name, 'tencent_HY-MT1.5-7B__plain_v1')
        self.assertIn('__p', HFChatMT('tencent/Hy-MT2-7B', 'hy_mt2_terms').name)
        self.assertNotIn('__p', HFChatMT('tencent/Hy-MT2-7B', 'hy_mt2').name)
        info = g.describe()
        self.assertEqual((info['model'], info['prompt_version']), ('m', 'isometric_v2'))

    def test_cached_candidate_from_another_prompt_is_not_used(self):
        import tempfile
        from dubbing_translation.common import write_json_once
        from dubbing_translation.io import load_config
        from dubbing_translation.pipeline import _candidate_path, load_candidates
        from dubbing_translation.timing import Window

        class Gen:
            uses_budget, name, prompt_sha = False, 'g', 'new'

        cfg = load_config()
        tasks = [{'unit_id': 'u0', 'source_text': 'x'}]
        windows = {'u0': Window('u0', 2.0, 2.0)}
        with tempfile.TemporaryDirectory() as tmp:
            path = _candidate_path(tmp, Gen(), 'u0', None)
            write_json_once(path, {'unit_id': 'u0', 'generator': 'g', 'items': [{'text': 'نص قديم', 'variant': 'v'}],
                                   'prompt_sha': 'old'})
            report = {}
            cands = load_candidates(tasks, windows, [Gen()], cfg, tmp, report=report)
        self.assertEqual(cands['u0'], [])
        self.assertEqual(report['stale'], {'g': ['u0']})

    def test_rank_cache_key_changes_with_the_prompt_text(self):
        import tempfile
        import dubbing_translation.pipeline as P
        from dubbing_translation.io import V2_CONFIG, load_config
        from dubbing_translation.language import Language, Policy

        class R:
            provider, model, calls = 'fake', 'r', 0

            def chat_json(self, system, user):
                self.calls += 1
                n = len(json.loads(user)['candidates'])
                return {'scores': [{'id': i + 1, 'adequacy': 50, 'fluency': 50} for i in range(n)]}

        cfg = load_config(V2_CONFIG)
        lang = Language(Policy.load('language/msa_simple_v1.toml'))
        ev = {'qe': 1.0, 'zone': 'fits', 'ratio_slot': 1.0}
        cands = [{'cand_id': 'a', 'text': 'أ', 'eval': ev}, {'cand_id': 'b', 'text': 'ب', 'eval': ev}]
        task = {'unit_id': 'u0', 'source_text': 'x'}
        r = R()
        with tempfile.TemporaryDirectory() as tmp:
            P.rank_unit(task, cands, cfg, r, tmp, lang)
            P.rank_unit(task, cands, cfg, r, tmp, lang)
            self.assertEqual(r.calls, 1)                                   # cached
            orig = P.RANK_V2_SYSTEM
            P.RANK_V2_SYSTEM = orig + ' changed'
            try:
                P.rank_unit(task, cands, cfg, r, tmp, lang)
            finally:
                P.RANK_V2_SYSTEM = orig
            self.assertEqual(r.calls, 2)                                   # new prompt text -> new key



# ---------------------------------------------------------------- promoted A: data contract with TTS
class DataContractTests(unittest.TestCase):
    def setUp(self):
        import copy
        import tempfile
        from dubbing_translation.io import V2_CONFIG, load_config
        from dubbing_translation.pipeline import run
        from dubbing_translation.timing import Window
        from test_translation_step import FakeQE, FakeSpeller, arabic_of
        self.tmp = tempfile.TemporaryDirectory()
        cfg = load_config(V2_CONFIG)
        cfg['language'] = {}
        rate = cfg['voice']['letters_per_second']
        self.tasks = [{'unit_id': 'a', 'source_text': 'So if you take X1 and', 'context_before': [], 'context_after': []},
                      {'unit_id': 'b', 'source_text': 'you know, put that first.', 'context_before': [], 'context_after': []},
                      {'unit_id': 'c', 'source_text': 'Steps 1, 2, 3, 4 follow.', 'context_before': [], 'context_after': []}]
        self.original = copy.deepcopy(self.tasks)
        windows = {u: Window(u, 2.0, 2.2) for u in 'abc'}
        self.timing = {'a': {'start': 10.0, 'end': 12.0, 'words': [{'w': 'So', 's': 10.0, 'e': 10.2}]},
                       'b': {'start': 12.1, 'end': 14.1, 'words': None},
                       'c': {'start': 15.0, 'end': 17.0, 'words': None}}

        class Gen:
            uses_budget, name, prompt_sha = False, 'mt', 'abc123'

            def describe(self):
                return {'name': 'mt', 'model': 'fake-mt', 'prompt_version': 'p1', 'prompt_sha': 'abc123'}

            def generate(self, task, budget):
                n = int(task['window_s'] * rate)
                return [{'text': arabic_of(n), 'variant': 'v'}]

        self.run_dir, self.final, _ = run(self.tasks, windows, cfg, self.tmp.name, generators=[Gen()], qe=FakeQE(),
                                          speller=FakeSpeller(), log=lambda *_: None, source_timing=self.timing)
        self.doc = json.loads((Path(self.run_dir) / 'translation_final.json').read_text(encoding='utf-8'))

    def tearDown(self):
        self.tmp.cleanup()

    def test_output_validates_against_the_schema(self):
        import jsonschema
        schema = json.loads((TR / 'schemas' / 'translation_final.schema.json').read_text(encoding='utf-8'))
        jsonschema.validate(self.doc, schema)
        self.assertEqual(self.doc['schema_version'], 'translation_final/2')

    def test_no_original_unit_or_timestamp_is_lost(self):
        parts = [p for u in self.doc['units'] for p in u['source_units']]
        self.assertEqual([p['unit_id'] for p in parts], ['a', 'b', 'c'])
        for p in parts:
            self.assertEqual((p['start'], p['end']), (self.timing[p['unit_id']]['start'], self.timing[p['unit_id']]['end']))
            self.assertEqual(p['source_text'], next(t['source_text'] for t in self.original if t['unit_id'] == p['unit_id']))
        self.assertEqual(parts[0]['words'], self.timing['a']['words'])

    def test_merged_unit_is_derived_and_maps_to_its_parts(self):
        merged = self.doc['units'][0]
        self.assertEqual((merged['unit_id'], merged['derived'], merged['merged_from']), ('a+b', True, ['a', 'b']))
        self.assertEqual((merged['start'], merged['end']), (10.0, 14.1))
        self.assertEqual(merged['source_text'].split(), 'So if you take X1 and you know, put that first.'.split())
        self.assertFalse(self.doc['units'][1]['derived'])

    def test_provenance_and_flags_are_kept(self):
        u = self.doc['units'][0]
        self.assertEqual(u['provenance']['prompt_sha'], 'abc123')
        self.assertEqual(u['provenance']['generator_info']['model'], 'fake-mt')
        self.assertEqual(u['spoken_version'], 'v2')
        self.assertEqual(self.doc['units'][1]['source_flags']['spelled_digits'], ['1, 2, 3, 4'])

    def test_input_tasks_are_not_modified(self):
        self.assertEqual(self.tasks, self.original)


if __name__ == '__main__':
    unittest.main()


# ---------------------------------------------------------------- final checks: coverage and readiness
class CoverageTests(unittest.TestCase):
    """Each original unit exactly once, unchanged; missing timings recorded, never invented."""
    setUp, tearDown = DataContractTests.setUp, DataContractTests.tearDown

    def _refs(self):
        from dubbing_translation.contract import source_reference
        return {t['unit_id']: source_reference(t, self.timing.get(t['unit_id'])) for t in self.original}

    def test_every_original_unit_appears_exactly_once(self):
        from dubbing_translation.contract import check_coverage
        order = [s['unit_id'] for u in self.doc['units'] for s in u['source_units']]
        self.assertEqual(order, ['a', 'b', 'c'])
        check_coverage(self.doc, self._refs())                          # passes on the real output

    def test_coverage_check_catches_duplicates_losses_and_changed_timing(self):
        import copy
        from dubbing_translation.contract import check_coverage
        refs = self._refs()
        dup = copy.deepcopy(self.doc)
        dup['units'].append(copy.deepcopy(dup['units'][-1]))
        lost = copy.deepcopy(self.doc)
        lost['units'] = lost['units'][:-1]
        moved = copy.deepcopy(self.doc)
        moved['units'][-1]['source_units'][0]['start'] = 99.0
        retext = copy.deepcopy(self.doc)
        retext['units'][-1]['source_units'][0]['source_text'] = 'Steps 1, 2, 2, 8, 8 follow.'
        for bad in (dup, lost, moved, retext):
            with self.assertRaises(ValueError):
                check_coverage(bad, refs)

    def test_readiness_separates_translation_test_from_audio_sync(self):
        r = self.doc['source_readiness']
        self.assertEqual((r['translation_test'], r['audio_sync'], r['word_alignment']), (True, True, False))
        self.assertEqual(r['units_without_word_alignment'], ['b', 'c'])

    def test_missing_timing_is_recorded_not_invented(self):
        from dubbing_translation.contract import source_readiness, source_reference
        refs = {t['unit_id']: source_reference(t, None) for t in self.original}
        r = source_readiness(refs)
        self.assertEqual((r['translation_test'], r['audio_sync']), (True, False))
        self.assertEqual(r['units_without_timing'], ['a', 'b', 'c'])
        self.assertTrue(all(v['start'] is None and v['end'] is None for v in refs.values()))
