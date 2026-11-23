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
from .io import load_config, load_tasks, load_windows


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
          refine=True, label='', seed_csv=None, overrides=None, hf_models=(), rank=True):
    """hf_models: [(model_id, template, n_samples)], e.g. [('tencent/Hy-MT2-7B', 'hy_mt2', 1)]."""
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
            m = cfg['models']
            gens = [Replay(folder, f'replay_{name}') for name, folder in (replay or {}).items()]
            if use_hymt:
                gens.append(HyMT(m['mt']))
            for model_id, template, samples, *extra in hf_models:      # optional 4th item: HFChatMT kwargs
                gens.append(HFChatMT(model_id, template, samples=samples, **(extra[0] if extra else {})))
            if use_llm:
                gens.append(IsometricLLM(LLMClient(m['llm_provider'], m['llm_model'],
                                                   min_interval_s=m['llm_min_interval_s'],
                                                   timeout_s=m.get('llm_timeout_s', 180))))
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
