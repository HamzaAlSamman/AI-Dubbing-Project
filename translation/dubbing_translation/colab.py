"""Colab driver: runs the translation step for one lesson in a background thread.

    from dubbing_translation.colab import start, status
    job = start(RUN_DIR, replay={'hy_mt15_7b_nocontext': EXP_OUT / 'hy_mt15_7b_nocontext'})
    status(job)          # poll; returns the metrics when finished

Long jobs run in a thread because notebook tool calls abort after ~30 min of silence.
"""
import csv
import json
import threading
import traceback
from pathlib import Path

from .common import sha256_text
from .io import load_config, load_source_timing, load_tasks, load_windows


def _seed_from_csv(units_csv, tasks, cache_path, metric_name, min_words=0):
    """Reuse MetricX scores already computed by the 17.4 evaluation (same model, same input format).
    Units long enough to be scored sentence by sentence are not seeded (their score is computed differently)."""
    from .text import clean_translation
    src = {t['unit_id']: t['source_text'] for t in tasks
           if not min_words or len(t['source_text'].split()) < min_words}
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    known = set()
    if cache_path.exists():
        known = {json.loads(l)['key'] for l in cache_path.read_text(encoding='utf-8').splitlines() if l.strip()}
    added = 0
    with open(units_csv, encoding='utf-8-sig') as f, cache_path.open('a', encoding='utf-8') as out:
        for r in csv.DictReader(f):
            if not r.get('natural_ar') or not r.get('metricx_natural') or r['unit_id'] not in src:
                continue
            # only the raw text was scored; a cleaned variant is a different input and is scored anew
            key = sha256_text(metric_name + '\x1f' + src[r['unit_id']] + '\x1f' + r['natural_ar'].strip())
            if key not in known and r['natural_ar'].strip() == clean_translation(r['natural_ar']):
                out.write(json.dumps({'key': key, 'score': float(r['metricx_natural'])}) + '\n')
                known.add(key)
                added += 1
    return added


def refine_client(m):
    from .generators import LLMClient
    return LLMClient(m.get('refine_provider', m['llm_provider']), m.get('refine_model', m['llm_model']),
                     min_interval_s=m.get('refine_min_interval_s', m['llm_min_interval_s']),
                     timeout_s=m.get('llm_timeout_s', 180), thinking_budget=m.get('refine_thinking_budget'))


def start(run_dir, *, work_subdir='translation_step', config=None, replay=None, use_hymt=False, use_llm=True,
          refine=True, label='', seed_csv=None, overrides=None, hf_models=(), rank=True, course_memory=None):
    """hf_models: [(model_id, template, n_samples)], e.g. [('tencent/Hy-MT2-7B', 'hy_mt2', 1)].
    course_memory: JSON path shared by all videos of the course (terminology kept the same across videos)."""
    from .generators import HFChatMT, HyMT, IsometricLLM, LLMClient, Replay
    from .pipeline import run
    from .scoring import CachedQE, Hunspell, MetricXQE, SegmentedQE

    run_dir = Path(run_dir)
    work = run_dir / work_subdir
    cfg = load_config(config, **(overrides or {})) if config else load_config(**(overrides or {}))
    tasks = load_tasks(run_dir / 'experiments_v1' / 'tasks.json')
    windows = load_windows(tasks, run_dir / 'master_source_only.json')
    job = {'log': [], 'done': False, 'error': None, 'result': None, 'work_dir': str(work)}
    log = job['log'].append

    def _target():
        try:
            from .preflight import preflight, requirements_for
            env_report = preflight(requirements_for(use_hymt=use_hymt, hf_models=hf_models, use_llm=use_llm,
                                                    refine=refine, rank=rank, language=bool(cfg.get('language'))),
                                   write_dir=work)               # fails before any model is loaded
            log('[preflight] ok: ' + ', '.join(f'{k} {v}' for k, v in env_report['versions'].items()))
            m = cfg['models']
            gens = [Replay(folder, f'replay_{name}') for name, folder in (replay or {}).items()]
            if use_hymt:
                gens.append(HyMT(m['mt'], language_aware=bool(cfg.get('language'))))
            for model_id, template, samples, *extra in hf_models:      # optional 4th item: HFChatMT kwargs
                gens.append(HFChatMT(model_id, template, samples=samples, **(extra[0] if extra else {})))
            if use_llm:
                gens.append(IsometricLLM(LLMClient(m['llm_provider'], m['llm_model'],
                                                   min_interval_s=m['llm_min_interval_s'],
                                                   timeout_s=m.get('llm_timeout_s', 180)),
                                     language_aware=bool(cfg.get('language'))))
            client = refine_client(m) if refine else None
            if not refine:
                cfg['refine']['enabled'] = False
            metric = MetricXQE(m['qe'])
            seg_words = cfg['gate'].get('qe_segment_min_words', 0)
            scorer = SegmentedQE(metric, seg_words) if seg_words else metric
            cache = work / 'cache' / 'qe_metricx24.jsonl'
            if seed_csv:
                log(f'[qe] seeded {_seed_from_csv(seed_csv, tasks, cache, scorer.name, seg_words)} saved scores')
            qe = CachedQE(scorer, cache)
            repo = Path(__file__).resolve().parents[2]
            ranker = None
            if rank and cfg.get('rank', {}).get('enabled'):
                rc = cfg['rank']
                ranker = LLMClient(rc['provider'], rc['model'], min_interval_s=m['llm_min_interval_s'],
                                   timeout_s=m.get('llm_timeout_s', 180))
            run_path, final, metrics = run(tasks, windows, cfg, work, generators=gens, qe=qe, speller=Hunspell(),
                                           refine_client=client, rank_client=ranker, repo_dir=repo, run_label=label,
                                           course_memory=course_memory, video_id=run_dir.name,
                                           environment=env_report,
                                           source_timing=load_source_timing(tasks, run_dir / 'master_source_only.json'),
                                           log=log)
            metric.close()
            job['result'] = {'run_dir': str(run_path), 'metrics': metrics}
        except Exception as exc:                     # noqa: BLE001 - reported to the notebook
            job['error'] = f'{type(exc).__name__}: {exc}\n' + traceback.format_exc(limit=4)
        finally:
            job['done'] = True

    job['thread'] = threading.Thread(target=_target, daemon=True)
    job['thread'].start()
    return job


def status(job, tail=15):
    print('\n'.join(job['log'][-tail:]))
    if job['error']:
        print('ERROR', job['error'])
    if job['done'] and job['result']:
        print(json.dumps(job['result'], ensure_ascii=False, indent=1))
    return job['done']


# ---------------------------------------------------------------- lecture evaluation (same gate for every system)
BASELINE_SYSTEMS = ('qwen3_4b', 'qwen3_8b', 'nllb_3b', 'S7_best_of_12_habibi_gate', 'hy_mt15_7b_nocontext',
                    'gemini35_flash_lite_api')


def evaluate_lecture(run_dir, *, systems=BASELINE_SYSTEMS, final_glob='*final_v1*', label='baseline_same_gate',
                     repo_dir=None, qe=None, speller=None, work_subdir='translation_step', log=print):
    """Score the final run and stored systems (experiments_v1/outputs/<name>/) with the gate of the
    final run: same windows, same MetricX cache, same spelling check. `qwen3_4b` is the baseline
    (the model of the first end-to-end run). Returns {system: metrics}; also writes
    evaluations/<eval_id>/auto.json and appends to evaluations/registry.jsonl.

    qe / speller: injectable for tests; by default MetricX-24 (sentence-level for long units,
    disk-cached) and hunspell-ar."""
    from .evaluate import Evaluation, from_run, from_unit_folder
    run_dir = Path(run_dir)
    work, exp = run_dir / work_subdir, run_dir / 'experiments_v1'
    cfg = load_config()
    tasks = load_tasks(exp / 'tasks.json')
    windows = load_windows(tasks, run_dir / 'master_source_only.json')
    found = {}
    finals = sorted((work / 'runs').glob(final_glob))
    if finals:
        found['final_v1'] = from_run(finals[-1])
    for name in systems:
        folder = exp / 'outputs' / name
        if folder.exists():
            found[name] = from_unit_folder(folder)
        else:
            log(f'missing system folder: {name}')
    metric = None
    if qe is None:
        from .scoring import CachedQE, MetricXQE, SegmentedQE
        metric = MetricXQE(cfg['models']['qe'])
        seg = cfg['gate'].get('qe_segment_min_words', 0)
        qe = CachedQE(SegmentedQE(metric, seg) if seg else metric, work / 'cache' / 'qe_metricx24.jsonl')
    if speller is None:
        from .scoring import Hunspell
        speller = Hunspell()
    ev = Evaluation(work, tasks, cfg, label=label, repo_dir=repo_dir, qe_name=qe.name)
    prov = {name: (str(finals[-1]) if name == 'final_v1' else f'experiments_v1/outputs/{name}') for name in found}
    summary = ev.run_automatic(windows, found, qe, speller, provenance=prov)
    if metric is not None:
        metric.close()
    log(f'eval_id: {ev.eval_id}')
    log(format_table(summary))
    return summary


def format_table(summary):
    cols = (('gate_accept_pct', 'accept%'), ('qe_mean', 'QE'), ('fits_natural_pct', 'fits%'),
            ('needs_speedup_pct', 'speedup%'), ('too_long_pct', 'too_long%'), ('too_short_pct', 'too_short%'),
            ('coverage_pct', 'cover%'))
    lines = ['| system | ' + ' | '.join(h for _, h in cols) + ' |', '|---' * (len(cols) + 1) + '|']
    for name, s in sorted(summary.items(), key=lambda kv: -kv[1]['gate_accept_pct']):
        lines.append(f'| {name} | ' + ' | '.join(str(s.get(k)) for k, _ in cols) + ' |')
    return '\n'.join(lines)


def compare_v1_v2(run_dir, v2_run, *, work_subdir='translation_step', v1_glob='*final_v1*', allow_degraded=False):
    """V1 (final_v1) vs a V2 run on the same lecture: acceptance, MetricX QE, and the language findings of
    the delivered lines, both measured with the V2 run's own glossary and policy."""
    import statistics
    from .common import read_json
    from .language import Glossary, Language, Policy
    runs = Path(run_dir) / work_subdir / 'runs'
    v1_dir = sorted(runs.glob(v1_glob))[-1]
    v2_dir = Path(v2_run)
    v2_manifest = read_json(v2_dir / 'manifest.json')
    status = v2_manifest.get('status') or {}
    if status.get('degraded') and not allow_degraded:
        raise ValueError(f'{v2_dir.name} is degraded ({status.get("reasons")}); it is not a complete run. '
                         'Rerun to resume, or pass allow_degraded=True for a diagnostic table.')
    cfg = v2_manifest['config']
    g = read_json(v2_dir / 'glossary.json') if (v2_dir / 'glossary.json').exists() else None
    glossary = Glossary({'version': g['version'], 'term': g['terms']}) if g else None
    lang = Language(Policy.load(cfg['language']['policy']), glossary)
    table = {}
    for name, d in (('V1 final_v1', v1_dir), ('V2', v2_dir)):
        units = read_json(d / 'translation_final.json')['units']
        found = [lang.check(u['source_text'], u.get('natural_ar') or '') for u in units]
        qes = [u['qe'] for u in units if u.get('qe') is not None]
        table[name] = {'run': d.name, 'units': len(units),
                       'accepted_pct': round(100 * sum(u['status'].startswith('accepted') for u in units) / len(units), 1),
                       'qe_mean': round(statistics.mean(qes), 3) if qes else None,
                       'term_missing_units': sum(bool(f['terms']) for f in found),
                       'stiff_pattern_units': sum(bool(f['calques']) for f in found),
                       'dialect_units': sum(bool(f['dialect']) for f in found),
                       'polished_units': sum(bool(u.get('polished_from')) for u in units)}
    return table
