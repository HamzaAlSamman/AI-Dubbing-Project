"""Reference benchmark: score translation systems against human MSA references from a public test set.

Our lecture has no human reference, so the gate there is reference-free (MetricX QE + checks).
This module adds the missing reference-based view on public data, with the same registry as
`evaluate.py`, so a model choice is supported by both kinds of evidence.

Default test set: WMT24++ (google/wmt24pp, Apache-2.0), en -> ar.
  * `target` of the Arabic configs is a post-edit into a *dialect* (ar_SA / ar_EG);
    `original_target` is the MSA human translation. Dubbing here is MSA, so the reference is
    `original_target` (configurable).
  * Domain `speech` (ASR transcripts of spoken video) is the closest to lecture dubbing.
  * Rows flagged `is_bad_source` are skipped.

No timing exists in text test sets, so a unit's slot is estimated from the English source:
  slot_s = English words / source speaking rate (2.6 words/s measured on our lecture:
  1,303 words in 503.7 s). The duration verdict then uses the same rules as the gate.

Metrics per system (all reproducible, each one optional when its library is missing):
  chrF++ and BLEU (sacrebleu), MetricX-24 with reference (lower is better),
  length ratio to the human reference (Arabic letters), LC10 = share within +-10 % of the
  reference length (the isometric-MT length-compliance idea, measured against the reference
  because source and target scripts differ), and duration zones against the estimated slot.
"""
import collections
import json
import statistics

from .common import now_iso
from .text import clean_translation, n_letters
from .timing import Window, duration_verdict

SOURCE_WORDS_PER_SECOND = 2.6


# ------------------------------------------------------------------ test sets
def load_wmt24pp(rows, *, domains=('speech',), reference='original_target', limit=None):
    """rows: iterable of WMT24++ records (from `datasets` or the jsonl file).
    Returns items [{unit_id, source_text, reference, domain, document_id}]."""
    items = []
    for r in rows:
        if r.get('is_bad_source'):
            continue
        if domains and r.get('domain') not in domains:
            continue
        ref = (r.get(reference) or '').strip()
        src = (r.get('source') or '').strip()
        if not ref or not src:
            continue
        items.append({'unit_id': f'wmt24pp_{r["segment_id"]}', 'source_text': src, 'reference': ref,
                      'domain': r.get('domain'), 'document_id': r.get('document_id')})
        if limit and len(items) >= limit:
            break
    return items


def load_wmt24pp_hf(config='en-ar_SA', **kw):
    from datasets import load_dataset
    return load_wmt24pp(load_dataset('google/wmt24pp', config, split='train'), **kw)


def estimated_window(item, words_per_second=SOURCE_WORDS_PER_SECOND):
    slot = max(1.0, len(item['source_text'].split()) / words_per_second)
    return Window(item['unit_id'], round(slot, 3), round(slot, 3))


def as_tasks(items, words_per_second=SOURCE_WORDS_PER_SECOND):
    """Items -> generator tasks (same keys the generators read; no context in a test set)."""
    return [{'unit_id': it['unit_id'], 'source_text': it['source_text'], 'context_before': [], 'context_after': [],
             'slot_s': estimated_window(it, words_per_second).slot_s} for it in items]


class DegenerateOutput(RuntimeError):
    """A generator's first outputs are not Arabic: broken load (dtype, quantisation, custom code)."""


def generate_cached(gen, tasks, cache_dir, *, rate, budget_cfg=None, log=print, probe=3, min_arabic=0.6,
                    report=None):
    """Run one generator over the benchmark tasks with a resumable jsonl cache
    (cache_dir/<generator name>.jsonl, one line per unit). Returns {unit_id: first candidate}:
    plain/greedy for MT models, 'fit' for the length-controlled LLM.

    A service failure (TransientLLMError) is not written: the unit is listed in report['deferred'] and
    retried by the next call.
    Fail fast: if the first `probe` new outputs are all non-Arabic, raise DegenerateOutput instead
    of spending hours of GPU on a model that did not load correctly (the probe lines stay in the
    cache file as evidence; delete the file before a retry with other settings)."""
    from pathlib import Path
    from .generators import InvalidLLMOutput, TransientLLMError
    from .text import arabic_fraction
    from .timing import letter_budget
    budget_cfg = budget_cfg or {'low': 0.90, 'high': 1.05, 'short_factor': 0.80}
    path = Path(cache_dir) / f'{gen.name}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec['unit_id']] = rec['items']
    todo = [t for t in tasks if t['unit_id'] not in done]
    bad, deferred = 0, []
    if report is not None:
        report['deferred'] = deferred
    for i, t in enumerate(todo, 1):
        w = Window(t['unit_id'], t['slot_s'], t['slot_s'])
        lo, hi = letter_budget(w, rate, low=budget_cfg['low'], high=budget_cfg['high'])
        try:
            items = gen.generate(t, {'min': lo, 'max': hi, 'short': int(hi * budget_cfg['short_factor'])})
        except (TransientLLMError, InvalidLLMOutput) as exc:   # not a result: not cached, retried later
            deferred.append(t['unit_id'])
            log(f'{gen.name}: {t["unit_id"]} deferred ({str(exc)[:80]})')
            continue
        done[t['unit_id']] = items
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'unit_id': t['unit_id'], 'items': items}, ensure_ascii=False) + '\n')
        if i <= probe:
            bad += not items or arabic_fraction(items[0]['text']) < min_arabic
            if i == probe and bad == probe:
                raise DegenerateOutput(f'{gen.name}: first {probe} outputs are not Arabic '
                                       f'(e.g. {(items[0]["text"] if items else "")[:80]!r})')
        if i % 10 == 0 or i == len(todo):
            log(f'{gen.name}: {i}/{len(todo)} new units')
    return {u: items[0]['text'] for u, items in done.items() if items}


# ------------------------------------------------------------------ metrics
def corpus_text_metrics(hyps, refs):
    """chrF++ and BLEU with sacrebleu defaults (13a tokenizer); None when sacrebleu is absent."""
    try:
        import sacrebleu
    except ImportError:
        return {'chrf_pp': None, 'bleu': None, 'sacrebleu': None}
    if not hyps:
        return {'chrf_pp': None, 'bleu': None, 'sacrebleu': sacrebleu.__version__}
    return {'chrf_pp': round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
            'bleu': round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            'sacrebleu': sacrebleu.__version__}


def length_metrics(hyp, ref):
    ratio = n_letters(hyp) / max(1, n_letters(ref))
    return {'len_ratio_ref': round(ratio, 3), 'lc10': abs(ratio - 1.0) <= 0.10}


def score_systems(items, systems, *, rate, ref_scorer=None, words_per_second=SOURCE_WORDS_PER_SECOND,
                  min_ratio=0.60, max_speedup=1.25):
    """systems: {name: {unit_id: arabic_text}}. Units a system did not translate count as missing
    (coverage) and are left out of its metrics; compare systems at equal coverage."""
    by_id = {it['unit_id']: it for it in items}
    texts = {name: {u: clean_translation(x) for u, x in sys.items() if u in by_id} for name, sys in systems.items()}
    ref_scores = {}
    if ref_scorer is not None:
        triples = sorted({(by_id[u]['source_text'], x, by_id[u]['reference'])
                          for sys in texts.values() for u, x in sys.items()})
        ref_scores = dict(zip(triples, ref_scorer.score(triples)))
    summary, rows = {}, []
    for name, sys in texts.items():
        units = [u for u in by_id if u in sys]
        hyps, refs, per = [sys[u] for u in units], [by_id[u]['reference'] for u in units], []
        for u in units:
            it, hyp = by_id[u], sys[u]
            dv = duration_verdict(n_letters(hyp), estimated_window(it, words_per_second), rate,
                                  min_ratio=min_ratio, max_speedup=max_speedup)
            row = {'system': name, 'unit_id': u, **length_metrics(hyp, it['reference']), 'zone': dv['zone'],
                   'ratio_slot': dv['ratio_slot'],
                   'metricx_ref': ref_scores.get((it['source_text'], hyp, it['reference']))}
            per.append(row)
            rows.append(row)
        n = len(by_id)
        zones = collections.Counter(r['zone'] for r in per)
        mx = [r['metricx_ref'] for r in per if r['metricx_ref'] is not None]
        summary[name] = {
            'units': len(units), 'coverage_pct': round(100 * len(units) / max(1, n), 1),
            **corpus_text_metrics(hyps, refs),
            'metricx_ref_mean': round(statistics.mean(mx), 3) if mx else None,
            'len_ratio_ref_median': round(statistics.median(r['len_ratio_ref'] for r in per), 3) if per else None,
            'lc10_pct': round(100 * sum(r['lc10'] for r in per) / max(1, len(per)), 1),
            'fits_or_speedup_pct': round(100 * (zones['fits'] + zones['speedup']) / max(1, len(per)), 1),
            'too_long_pct': round(100 * zones['too_long'] / max(1, len(per)), 1),
            'too_short_pct': round(100 * zones['too_short'] / max(1, len(per)), 1),
        }
    # the human reference itself, as an upper bound for length/duration behaviour
    ref_rows = [duration_verdict(n_letters(it['reference']), estimated_window(it, words_per_second), rate,
                                 min_ratio=min_ratio, max_speedup=max_speedup)['zone'] for it in items]
    rz = collections.Counter(ref_rows)
    summary['_human_reference'] = {
        'units': len(items),
        'fits_or_speedup_pct': round(100 * (rz['fits'] + rz['speedup']) / max(1, len(items)), 1),
        'too_long_pct': round(100 * rz['too_long'] / max(1, len(items)), 1),
        'too_short_pct': round(100 * rz['too_short'] / max(1, len(items)), 1),
    }
    return summary, rows


def run_reference(evaluation, items, systems, *, rate, ref_scorer=None, provenance=None, test_set='wmt24pp',
                  **kw):
    """Write evaluations/<eval_id>/reference.json and one registry line per system."""
    summary, rows = score_systems(items, systems, rate=rate, ref_scorer=ref_scorer, **kw)
    meta = {**evaluation.meta, 'test_set': test_set, 'n_items': len(items),
            'ref_metric': getattr(ref_scorer, 'name', None)}
    (evaluation.dir / 'reference.json').write_text(
        json.dumps({'meta': meta, 'created_at': now_iso(), 'provenance': provenance or {}, 'summary': summary,
                    'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
    for name, s in summary.items():
        if not name.startswith('_'):
            evaluation._register({'kind': 'reference', 'test_set': test_set, 'system': name,
                                  'ref_metric': meta['ref_metric'], 'provenance': (provenance or {}).get(name),
                                  'metrics': s})
    return summary
