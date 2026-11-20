"""Evaluation harness: score any set of systems on the same units with the same measurements.

Every evaluation writes
  evaluations/<eval_id>/auto.json            per-system automatic metrics (+ per-unit rows)
  evaluations/<eval_id>/judge_<a>_vs_<b>.jsonl  blind pairwise judgements (optional)
and appends one line per system / comparison to evaluations/registry.jsonl, so results of
different days, models and code versions stay comparable (same tasks hash, same gate config).

A "system" is just {unit_id: arabic_text}; loaders below read the project's artifacts.
"""
import collections
import json
import statistics
from pathlib import Path

from .common import git_commit, now_iso, now_stamp, read_json, sha256_json
from .gate import evaluate
from .judge import pairwise
from .pipeline import consensus_allowlist
from .text import ar_tokens, clean_translation


# ------------------------------------------------------------------ loaders
def from_run(run_dir):
    return {u['unit_id']: u['natural_ar'] for u in read_json(Path(run_dir) / 'translation_final.json')['units']
            if u.get('natural_ar')}


def from_unit_folder(folder):
    """experiments_v1/outputs/<system>/<unit>.json with 'natural_ar'."""
    out = {}
    for p in Path(folder).glob('*.json'):
        text = read_json(p).get('natural_ar')
        if isinstance(text, str) and text.strip():
            out[p.stem] = text
    return out


def from_candidate_cache(folder, variant):
    """cache/candidates/<generator>/<unit>.json -> the item with the given variant."""
    out = {}
    for p in Path(folder).glob('*.json'):
        for item in read_json(p)['items']:
            if item['variant'] == variant:
                out[read_json(p)['unit_id']] = item['text']
    return out


# ------------------------------------------------------------------ automatic metrics
def automatic(tasks, windows, systems, cfg, qe, speller):
    src = {t['unit_id']: t['source_text'] for t in tasks}
    texts = {name: {u: clean_translation(x) for u, x in sys.items() if u in src} for name, sys in systems.items()}
    pairs = sorted({(src[u], x) for sys in texts.values() for u, x in sys.items()})
    scores = dict(zip(pairs, qe.score(pairs)))
    words = {w for sys in texts.values() for x in sys.values() for w in ar_tokens(x)}
    misspelled = speller.misspelled(sorted(words))
    pseudo = {u: [{'generator': name, 'also_from': [], 'text': sys[u]} for name, sys in texts.items() if u in sys]
              for u in src}
    allow = consensus_allowlist(pseudo, misspelled, cfg['gate'].get('spelling_allowlist', ()))
    summary, rows = {}, []
    for name, sys in texts.items():
        evs = []
        for u in src:
            if u not in sys:
                continue
            ev = evaluate(sys[u], src[u], windows[u], qe=scores[(src[u], sys[u])], misspelled_words=misspelled,
                          cfg=cfg, allowlist=allow)
            evs.append(ev)
            rows.append({'system': name, 'unit_id': u, **{k: ev[k] for k in ('qe', 'accepted', 'zone', 'ratio_window',
                                                                              'spelling_rate')},
                         'reasons': '|'.join(ev['reasons'])})
        n = len(src)
        zones = collections.Counter(e['zone'] for e in evs)
        summary[name] = {
            'units_with_output': len(evs), 'coverage_pct': round(100 * len(evs) / n, 1),
            'qe_mean': round(statistics.mean(e['qe'] for e in evs), 3) if evs else None,
            'qe_median': round(statistics.median(e['qe'] for e in evs), 3) if evs else None,
            'gate_accept_pct': round(100 * sum(e['accepted'] for e in evs) / n, 1),
            'fits_natural_pct': round(100 * zones['fits'] / n, 1),
            'needs_speedup_pct': round(100 * zones['speedup'] / n, 1),
            'too_long_pct': round(100 * zones['too_long'] / n, 1),
            'too_short_pct': round(100 * zones['too_short'] / n, 1),
            'invalid_units': sum(any(r.startswith('invalid') for r in e['reasons']) for e in evs),
        }
    return summary, rows


# ------------------------------------------------------------------ evaluation run
class Evaluation:
    def __init__(self, work_dir, tasks, cfg, *, label='', repo_dir=None, qe_name=None, eval_id=None):
        """eval_id: reopen an earlier evaluation (e.g. to finish a judge run after a quota reset)."""
        self.work_dir = Path(work_dir)
        self.tasks, self.cfg = tasks, cfg
        self.tasks_sha = sha256_json([{k: t[k] for k in ('unit_id', 'source_text')} for t in tasks])
        self.eval_id = eval_id or (f'ev_{now_stamp()}' + (f'_{label}' if label else ''))
        self.dir = self.work_dir / 'evaluations' / self.eval_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {'eval_id': self.eval_id, 'tasks_sha256': self.tasks_sha, 'gate_config_sha256':
                     sha256_json(cfg['gate']), 'git_commit': git_commit(repo_dir) if repo_dir else None,
                     'qe_metric': qe_name}

    def _register(self, rec):
        path = self.work_dir / 'evaluations' / 'registry.jsonl'
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({**self.meta, 'created_at': now_iso(), **rec}, ensure_ascii=False) + '\n')

    def run_automatic(self, windows, systems, qe, speller, provenance=None):
        summary, rows = automatic(self.tasks, windows, systems, self.cfg, qe, speller)
        (self.dir / 'auto.json').write_text(json.dumps({'meta': self.meta, 'provenance': provenance or {},
                                                        'summary': summary, 'rows': rows},
                                                       ensure_ascii=False, indent=1), encoding='utf-8')
        for name, s in summary.items():
            self._register({'kind': 'automatic', 'system': name,
                            'provenance': (provenance or {}).get(name), 'metrics': s})
        return summary

    def run_judge(self, name_a, sys_a, name_b, sys_b, client, log=print):
        out = self.dir / f'judge_{name_a}_vs_{name_b}.jsonl'
        summary = pairwise(self.tasks, sys_a, sys_b, client, out, log=log)
        self._register({'kind': 'pairwise_judge', 'system_a': name_a, 'system_b': name_b,
                        'judge': f'{client.provider}:{client.model}', 'metrics': summary})
        return summary
