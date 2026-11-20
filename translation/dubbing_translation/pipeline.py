"""The translation step: candidates -> gate -> guarded refine -> selection -> artifacts.

Layout (all under `work_dir`, normally the lesson folder on Drive):
  cache/candidates/<generator>/<unit>[__<budget>].json   generations, shared by every run (write-once)
  cache/qe_<metric>.jsonl                                 QE scores cache
  cache/refine/<refiner>/<unit>__<input hash>.json        refine attempts (write-once)
  runs/<run_id>/manifest.json                             config, hashes, git commit, models
  runs/<run_id>/translation_final.json                    the hand-off to the TTS stage
  runs/<run_id>/candidates_scored.csv                     every candidate with every measurement
  runs/<run_id>/metrics.json                              run metrics (also appended to registry.jsonl)
"""
import collections
import csv
import json
import statistics
from pathlib import Path

from . import __version__
from .common import git_commit, now_iso, now_stamp, read_json, sha256_json, write_json_once
from .gate import evaluate, only_duration_failed, rank_key
from .generators import QuotaExhausted
from .prompts import REFINE_SYSTEM, REFINE_VERSION, describe_issues
from .text import ar_tokens, clean_translation, tts_text
from .timing import letter_budget


def _budget(window, cfg):
    rate, b = cfg['voice']['letters_per_second'], cfg['budget']
    lo, hi = letter_budget(window, rate, low=b['low'], high=b['high'])
    return {'min': lo, 'max': hi, 'short': max(lo, int(hi * b['short_factor']))}


# ------------------------------------------------------------------ 1. candidates
def _candidate_path(work_dir, gen, unit_id, budget):
    suffix = f'__{sha256_json(budget)[:8]}' if getattr(gen, 'uses_budget', True) else ''
    return Path(work_dir) / 'cache' / 'candidates' / gen.name / f'{unit_id}{suffix}.json'


def generate_candidates(tasks, windows, generators, cfg, work_dir, log=print):
    """Resumable: cached units are skipped. A daily-quota stop leaves the remaining units for the next run."""
    stopped = {}
    for gen in generators:
        todo = []
        for t in tasks:
            budget = _budget(windows[t['unit_id']], cfg)
            path = _candidate_path(work_dir, gen, t['unit_id'], budget)
            if not path.exists():
                todo.append((t, budget, path))
        log(f'[candidates] {gen.name}: {len(todo)} to generate')
        for i, (t, budget, path) in enumerate(todo, 1):
            task = {**t, 'slot_s': windows[t['unit_id']].slot_s, 'window_s': windows[t['unit_id']].window_s}
            try:
                items = gen.generate(task, budget)
            except QuotaExhausted as exc:
                stopped[gen.name] = str(exc)[:200]
                log(f'[candidates] {gen.name}: quota reached after {i - 1} units; rerun later to resume')
                break
            write_json_once(path, {'unit_id': t['unit_id'], 'generator': gen.name, 'budget': budget,
                                   'items': items, 'created_at': now_iso()})
            if i % 10 == 0 or i == len(todo):
                log(f'  {gen.name}: {i}/{len(todo)}')
        if hasattr(gen, 'close'):
            gen.close()
    return stopped


def load_candidates(tasks, windows, generators, cfg, work_dir):
    """{unit_id: [{'cand_id','generator','variant','text','raw_text'}]}; identical texts merged."""
    out = {}
    for t in tasks:
        budget = _budget(windows[t['unit_id']], cfg)
        seen, cands = {}, []
        for gen in generators:
            path = _candidate_path(work_dir, gen, t['unit_id'], budget)
            if not path.exists():
                continue
            for item in read_json(path)['items']:
                text = clean_translation(item['text'])
                if not text:
                    continue
                if text in seen:
                    seen[text]['also_from'].append(gen.name)
                    continue
                c = {'cand_id': f'{gen.name}:{item["variant"]}', 'generator': gen.name, 'variant': item['variant'],
                     'text': text, 'raw_text': item['text'], 'also_from': []}
                seen[text] = c
                cands.append(c)
        out[t['unit_id']] = cands
    return out


# ------------------------------------------------------------------ 2. scoring
def consensus_allowlist(cands_by_unit, misspelled, extra=()):
    """Words the spell checker rejects but that >= 2 different generators chose independently
    are treated as accepted technical vocabulary (e.g. بكسلات); plus the configured list."""
    users = collections.defaultdict(set)
    for cands in cands_by_unit.values():
        for c in cands:
            for gen in [c['generator'], *c['also_from']]:
                for w in set(ar_tokens(c['text'])) & misspelled:
                    users[w].add(gen.split('__')[0])          # provider/model family, not prompt variant
    return frozenset(w for w, g in users.items() if len(g) >= 2) | frozenset(extra)


def score_all(tasks, cands_by_unit, windows, cfg, qe, speller):
    src = {t['unit_id']: t['source_text'] for t in tasks}
    pairs = [(src[u], c['text']) for u, cs in cands_by_unit.items() for c in cs]
    scores = dict(zip(pairs, qe.score(pairs))) if pairs else {}
    words = {w for cs in cands_by_unit.values() for c in cs for w in ar_tokens(c['text'])}
    misspelled = speller.misspelled(sorted(words))
    allow = consensus_allowlist(cands_by_unit, misspelled, cfg['gate'].get('spelling_allowlist', ()))
    for u, cs in cands_by_unit.items():
        for c in cs:
            c['eval'] = evaluate(c['text'], src[u], windows[u], qe=scores[(src[u], c['text'])],
                                 misspelled_words=misspelled, cfg=cfg, allowlist=allow)
    return misspelled, allow


# ------------------------------------------------------------------ 3. guarded refine
def refine_unit(task, window, cands, cfg, client, qe, misspelled_fn, allow, work_dir, log=print):
    """Ask an LLM for one corrected translation when no candidate passed the gate.

    Guard: the edit is kept only if it passes the full gate AND, when the best draft failed on
    duration alone, its QE is not worse than that draft by more than `max_qe_worse`."""
    rcfg = cfg['refine']
    budget = _budget(window, cfg)
    valid = [c for c in cands if not any(r.startswith('invalid') for r in c['eval']['reasons'])] or cands
    drafts = sorted(valid, key=lambda c: (c['eval']['qe'] if c['eval']['qe'] is not None else 99))[:rcfg['drafts_shown']]
    if not drafts:
        return None, []
    best = drafts[0]
    payload = {'english_source': task['source_text'], 'slot_seconds': round(window.slot_s, 1),
               'target_letters_min': budget['min'], 'target_letters_max': budget['max'],
               'drafts': [{'arabic': c['text'], 'letters': c['eval']['letters'],
                           'problems': describe_issues(c['eval']['reasons'], c['eval']['misspelled'])} for c in drafts]}
    name = f'{client.provider}__{client.model.replace("/", "_").replace(":", "_")}__{REFINE_VERSION}'
    path = Path(work_dir) / 'cache' / 'refine' / name / f'{task["unit_id"]}__{sha256_json([payload, rcfg])[:10]}.json'
    if path.exists():
        rec = read_json(path)
    else:
        attempts, note = [], ''
        for attempt in range(rcfg['max_attempts']):
            obj = client.chat_json(REFINE_SYSTEM, json.dumps(payload, ensure_ascii=False) + note) or {}
            text = clean_translation(obj.get('natural_ar') if isinstance(obj.get('natural_ar'), str) else '')
            if not text:
                attempts.append({'attempt': attempt, 'text': '', 'verdict': 'no_text'})
                continue
            q = qe.score([(task['source_text'], text)])[0]
            ev = evaluate(text, task['source_text'], window, qe=q, misspelled_words=misspelled_fn(ar_tokens(text)),
                          cfg=cfg, allowlist=allow)
            guard_ok = not (only_duration_failed(best['eval']) and best['eval']['qe'] is not None
                            and q > best['eval']['qe'] + rcfg['max_qe_worse'])
            verdict = 'accepted' if ev['accepted'] and guard_ok else ('meaning_guard' if ev['accepted'] else 'rejected')
            attempts.append({'attempt': attempt, 'text': text, 'eval': ev, 'verdict': verdict})
            if verdict == 'accepted':
                break
            note = ('\nYour previous answer: ' + text + '\nIts problems: ' +
                    '; '.join(describe_issues(ev['reasons'], ev['misspelled']) or ['meaning changed too much']) +
                    '\nWrite a better one.')
        rec = {'unit_id': task['unit_id'], 'refiner': name, 'payload': payload, 'attempts': attempts,
               'created_at': now_iso()}
        write_json_once(path, rec)
    accepted = [a for a in rec['attempts'] if a.get('verdict') == 'accepted']
    if not accepted:
        return None, rec['attempts']
    a = accepted[0]
    return {'cand_id': f'{name}:refine', 'generator': name, 'variant': 'refine', 'text': a['text'],
            'raw_text': a['text'], 'also_from': [], 'eval': a['eval'], 'refined_from': best['cand_id']}, rec['attempts']


# ------------------------------------------------------------------ 4. run
def run(tasks, windows, cfg, work_dir, *, generators, qe, speller, refine_client=None, repo_dir=None,
        run_label='', log=print):
    work_dir = Path(work_dir)
    cfg_hash = sha256_json(cfg)
    run_id = f'tr_{now_stamp()}_{cfg_hash[:8]}' + (f'_{run_label}' if run_label else '')
    run_dir = work_dir / 'runs' / run_id
    tasks_hash = sha256_json([{k: t[k] for k in ('unit_id', 'source_text')} for t in tasks])
    manifest = {'run_id': run_id, 'created_at': now_iso(), 'package_version': __version__,
                'git_commit': git_commit(repo_dir) if repo_dir else None, 'config': cfg, 'config_sha256': cfg_hash,
                'tasks_sha256': tasks_hash, 'units': len(tasks), 'generators': [g.name for g in generators],
                'qe_metric': qe.name, 'spell_checker': speller.name,
                'refiner': None if refine_client is None else f'{refine_client.provider}:{refine_client.model}',
                'windows': {u: w.as_dict() for u, w in windows.items()}, 'human_review': 'none'}
    write_json_once(run_dir / 'manifest.json', manifest)

    stopped = generate_candidates(tasks, windows, generators, cfg, work_dir, log)
    cands = load_candidates(tasks, windows, generators, cfg, work_dir)
    misspelled, allow = score_all(tasks, cands, windows, cfg, qe, speller)
    cache_words = dict.fromkeys(misspelled)

    def misspelled_fn(words):
        new = [w for w in words if w not in cache_words and w not in allow]
        found = speller.misspelled(new) if new else set()
        for w in new:
            cache_words[w] = None
        return set(misspelled) | found

    final, refine_log = [], {}
    for t in tasks:
        u, w = t['unit_id'], windows[t['unit_id']]
        cs = cands[u]
        accepted = sorted((c for c in cs if c['eval']['accepted']), key=lambda c: rank_key(c['eval']))
        status, chosen = 'accepted', (accepted[0] if accepted else None)
        if chosen is None and refine_client is not None and cfg['refine']['enabled'] and cs:
            try:
                chosen, refine_log[u] = refine_unit(t, w, cs, cfg, refine_client, qe, misspelled_fn, allow,
                                                    work_dir, log)
            except QuotaExhausted as exc:
                stopped['refine'] = str(exc)[:200]
                chosen = None
            if chosen is not None:
                cs.append(chosen)
                status = 'accepted_after_refine'
        if chosen is None:
            valid = [c for c in cs if not any(r.startswith('invalid') for r in c['eval']['reasons'])] or cs
            if not valid:
                final.append({'unit_id': u, 'source_text': t['source_text'], 'status': 'missing', 'natural_ar': None,
                              'tts_text': None, 'slot_s': w.slot_s, 'window_s': w.window_s, 'flags': ['no_candidate']})
                continue
            chosen = min(valid, key=lambda c: (len(c['eval']['reasons']), rank_key(c['eval'])))
            status = 'needs_review'
        ev = chosen['eval']
        flags = []
        if w.slot_s < cfg['gate']['tiny_slot_s'] and ev['zone'] in ('too_long', 'speedup'):
            flags.append('tiny_slot_consider_merging_with_neighbour')
        if ev['zone'] == 'speedup':
            flags.append(f'needs_speedup_x{ev["ratio_window"]:.2f}')
        final.append({'unit_id': u, 'source_text': t['source_text'], 'status': status,
                      'natural_ar': chosen['text'], 'tts_text': tts_text(chosen['text']),
                      'selected_from': chosen['cand_id'], 'refined_from': chosen.get('refined_from'),
                      'qe': ev['qe'], 'spelling_rate': ev['spelling_rate'], 'letters': ev['letters'],
                      'predicted_s': ev['predicted_s'], 'slot_s': w.slot_s, 'window_s': w.window_s,
                      'ratio_slot': ev['ratio_slot'], 'ratio_window': ev['ratio_window'], 'zone': ev['zone'],
                      'reasons': ev['reasons'], 'flags': flags, 'n_candidates': len(cs)})

    metrics = summarize(final, cands, stopped, refine_client, allow)
    _write_outputs(run_dir, final, cands, metrics, tasks, manifest, refine_log)
    _append_registry(work_dir / 'registry.jsonl', manifest, metrics)
    if cfg.get('tracking', {}).get('mlflow'):
        _log_mlflow(cfg, manifest, metrics, run_dir, log)
    log(f'[run] {run_id}: accepted {metrics["accepted"]}/{metrics["units"]} ({metrics["accepted_pct"]}%) | '
        f'QE mean {metrics["qe_mean"]} | refine accepted {metrics["accepted_after_refine"]} | '
        f'needs review {metrics["needs_review_units"]}')
    return run_dir, final, metrics


def summarize(final, cands, stopped, refine_client, allow):
    n = len(final)
    ok = [f for f in final if f['status'] in ('accepted', 'accepted_after_refine')]
    qes = [f['qe'] for f in final if f.get('qe') is not None]
    return {
        'units': n, 'accepted': len(ok), 'accepted_pct': round(100 * len(ok) / max(1, n), 1),
        'accepted_without_refine': sum(f['status'] == 'accepted' for f in final),
        'accepted_after_refine': sum(f['status'] == 'accepted_after_refine' for f in final),
        'needs_review_units': [f['unit_id'] for f in final if f['status'] in ('needs_review', 'missing')],
        'qe_mean': round(statistics.mean(qes), 3) if qes else None,
        'qe_median': round(statistics.median(qes), 3) if qes else None,
        'zones': dict(collections.Counter(f.get('zone') for f in final)),
        'picked_from': dict(collections.Counter((f.get('selected_from') or 'none').split(':')[0] for f in final)),
        'candidates_per_unit': round(statistics.mean(len(c) for c in cands.values()), 2) if cands else 0,
        'llm_calls_refine': getattr(refine_client, 'calls', 0) if refine_client else 0,
        'spelling_allowlist_size': len(allow), 'stopped_early': stopped,
        'flags': dict(collections.Counter(fl.split('_x')[0] for f in final for fl in f.get('flags', []))),
    }


def _write_outputs(run_dir, final, cands, metrics, tasks, manifest, refine_log):
    write_json_once(run_dir / 'translation_final.json',
                    {'run_id': manifest['run_id'], 'config_sha256': manifest['config_sha256'],
                     'tasks_sha256': manifest['tasks_sha256'], 'git_commit': manifest['git_commit'],
                     'created_at': now_iso(), 'human_approved': False, 'units': final})
    write_json_once(run_dir / 'metrics.json', metrics)
    if refine_log:
        write_json_once(run_dir / 'refine_log.json', refine_log)
    rows = []
    for t in tasks:
        for c in cands[t['unit_id']]:
            e = c['eval']
            rows.append({'unit_id': t['unit_id'], 'cand_id': c['cand_id'], 'also_from': '|'.join(c['also_from']),
                         'accepted': e['accepted'], 'reasons': '|'.join(e['reasons']), 'qe': e['qe'],
                         'spelling_rate': e['spelling_rate'], 'letters': e['letters'],
                         'predicted_s': e['predicted_s'], 'ratio_slot': e['ratio_slot'],
                         'ratio_window': e['ratio_window'], 'zone': e['zone'], 'text': c['text']})
    path = run_dir / 'candidates_scored.csv'
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['unit_id'])
        writer.writeheader()
        writer.writerows(rows)


def _log_mlflow(cfg, manifest, metrics, run_dir, log):
    """Optional MLflow mirror of the run (the JSON registry stays the source of truth)."""
    try:
        import mlflow
    except ImportError:
        log('[tracking] mlflow not installed; registry.jsonl only')
        return
    mlflow.set_tracking_uri(cfg['tracking'].get('mlflow_uri') or f'file:{run_dir.parent.parent / "mlruns"}')
    mlflow.set_experiment(cfg['tracking'].get('experiment', 'translation_step'))
    with mlflow.start_run(run_name=manifest['run_id']):
        mlflow.log_params({'config_sha256': manifest['config_sha256'], 'git_commit': manifest['git_commit'],
                           'generators': ','.join(manifest['generators']), 'refiner': manifest['refiner'],
                           **{f'gate.{k}': v for k, v in cfg['gate'].items() if not isinstance(v, (list, tuple))}})
        mlflow.log_metrics({k: metrics[k] for k in ('accepted_pct', 'accepted', 'accepted_after_refine')
                            if metrics.get(k) is not None})
        if metrics.get('qe_mean') is not None:
            mlflow.log_metric('qe_mean', metrics['qe_mean'])
        mlflow.log_artifacts(str(run_dir))


def _append_registry(path, manifest, metrics):
    rec = {k: manifest[k] for k in ('run_id', 'created_at', 'package_version', 'git_commit', 'config_sha256',
                                    'tasks_sha256', 'generators', 'qe_metric', 'refiner')}
    rec['metrics'] = {k: metrics[k] for k in ('units', 'accepted', 'accepted_pct', 'accepted_after_refine',
                                              'qe_mean', 'zones', 'picked_from', 'needs_review_units')}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
