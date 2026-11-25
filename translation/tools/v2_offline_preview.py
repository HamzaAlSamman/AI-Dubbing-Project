"""Before/after preview of Translation V2 language features on an existing V1 run, without a GPU.

What it does (all inputs are artifacts of a V1 run; nothing is regenerated with the MT models):
  1. glossary : the terminology agent (LLM) reads the whole transcript once and proposes the glossary
  2. before   : language findings of the delivered V1 lines (glossary terms, dialect, stiff patterns)
  3. regate   : the V2 gate on the V1 candidates (stored MetricX QE + new language checks); the V1 line is
                kept when it still passes, otherwise the best accepted by MetricX order (no LLM ranking)
  4. polish   : the V2 polish prompt on the selected lines that still have findings (LLM only)

Limits, printed in the output: step 4 is NOT meaning-checked here (MetricX needs a GPU); the full V2
run (language-aware generators + guarded polish) runs on Colab. Outputs stay local (private lecture).

usage: python v2_offline_preview.py --tasks tasks.json --final translation_final.json
                                    --candidates candidates_scored.csv --out <folder>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubbing_translation.common import now_iso, read_json, sha256_json, write_json_once   # noqa: E402
from dubbing_translation.generators import LLMClient, QuotaExhausted                      # noqa: E402
from dubbing_translation.language import Language, Policy, build_glossary                 # noqa: E402
from dubbing_translation.prompts import POLISH_SYSTEM, POLISH_VERSION                     # noqa: E402
from dubbing_translation.text import clean_translation                                    # noqa: E402


def findings(lang, src, text):
    f = lang.check(src, text or '')
    return {**f, 'reject': lang.reasons(f)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--final', required=True)
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--policy', default='language/msa_simple_v1.toml')
    ap.add_argument('--glossary-model', default='gemini-3.5-flash')
    ap.add_argument('--polish-model', default='gemini-3.5-flash-lite')
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = read_json(args.tasks)
    tasks = tasks['tasks'] if isinstance(tasks, dict) else tasks
    src = {t['unit_id']: t['source_text'] for t in tasks}
    final = read_json(args.final)
    rows = list(csv.DictReader(open(args.candidates, encoding='utf-8-sig')))

    glossary_client = LLMClient('gemini', args.glossary_model, min_interval_s=4.5)
    glossary, gpath = build_glossary(tasks, glossary_client, out / 'cache')
    lang = Language(Policy.load(args.policy), glossary)
    print(f'glossary: {len(glossary.terms)} terms ({gpath.name})')

    # 2. before
    before = {u['unit_id']: findings(lang, src[u['unit_id']], u['natural_ar']) for u in final['units']}

    # 3. regate the V1 candidates with the V2 gate (stored QE; language reasons added)
    by_unit = {}
    for r in rows:
        f = findings(lang, src[r['unit_id']], r['text'])
        reasons = [x for x in r['reasons'].split('|') if x] + f['reject']
        by_unit.setdefault(r['unit_id'], []).append({
            'cand_id': r['cand_id'], 'text': r['text'], 'qe': float(r['qe']) if r['qe'] else 99.0,
            'zone': r['zone'], 'ratio_slot': float(r['ratio_slot']), 'reasons': reasons, 'findings': f,
            'accepted': not reasons, 'v1_accepted': r['accepted'] == 'True'})
    key = lambda c: (c['qe'], c['zone'] == 'speedup', abs(1.0 - c['ratio_slot']))   # gate.rank_key
    selected = {}
    for u in final['units']:
        cs = by_unit.get(u['unit_id'], [])
        acc = sorted((c for c in cs if c['accepted']), key=key)
        same = [c for c in acc if c['text'] == u['natural_ar']]
        if same:                                     # the V1 choice still passes the V2 gate: keep it
            selected[u['unit_id']] = {**same[0], 'status': 'accepted'}
        elif acc:
            selected[u['unit_id']] = {**acc[0], 'status': 'accepted'}
        else:
            fb = sorted(cs, key=lambda c: (len(c['reasons']), key(c)))
            selected[u['unit_id']] = {**fb[0], 'status': 'needs_review'} if fb else None

    # 4. polish preview (LLM edit; meaning guard pending on Colab)
    polish_client = LLMClient('gemini', args.polish_model, min_interval_s=4.5)
    cache = out / 'cache' / 'polish_preview'
    polished = {}
    for u in final['units']:
        uid, sel = u['unit_id'], selected.get(u['unit_id'])
        if not sel:
            continue
        f = sel['findings']
        if not (f['calques'] or f['terms'] or f['dialect']):
            continue
        payload = {'english_source': src[uid], 'arabic': sel['text'],
                   'problems': lang.issue_hints(src[uid], f), **lang.prompt_block(src[uid])}
        path = cache / f'{uid}__{sha256_json([payload, args.polish_model, POLISH_VERSION])[:10]}.json'
        if path.exists():
            rec = read_json(path)
        else:
            try:
                obj = polish_client.chat_json(POLISH_SYSTEM, json.dumps(payload, ensure_ascii=False)) or {}
            except QuotaExhausted as exc:
                print('quota reached:', str(exc)[:120])
                break
            text = clean_translation(obj.get('natural_ar') if isinstance(obj.get('natural_ar'), str) else '')
            rec = {'unit_id': uid, 'payload': payload, 'text': text, 'model': args.polish_model,
                   'created_at': now_iso()}
            write_json_once(path, rec)
        if rec['text']:
            polished[uid] = {'text': rec['text'], 'findings': findings(lang, src[uid], rec['text'])}

    def count(d):
        vals = [v for v in d.values() if v]
        return {'units': len(vals), 'term_missing': sum(bool(v['terms']) for v in vals),
                'dialect': sum(bool(v['dialect']) for v in vals), 'stiff': sum(bool(v['calques']) for v in vals),
                'stiff_total': sum(len(v['calques']) for v in vals)}

    after_select = {u: (s['findings'] if s else None) for u, s in selected.items()}
    after_polish = {u: (polished[u]['findings'] if u in polished else after_select[u]) for u in after_select}
    summary = {'before_v1': count(before), 'after_v2_select': count(after_select),
               'after_v2_select_plus_polish_preview': count(after_polish),
               'v2_select_changed_units': sum(1 for u in final['units']
                                              if selected.get(u['unit_id']) and
                                              selected[u['unit_id']]['text'] != u['natural_ar']),
               'v2_needs_review': [u for u, s in selected.items() if s and s['status'] == 'needs_review'],
               'polished_units': len(polished)}
    result = {'created_at': now_iso(), 'run_id': final.get('run_id'), 'glossary_file': str(gpath),
              'glossary': glossary.as_dict(), 'policy': lang.policy.version, 'summary': summary,
              'limits': ['polish preview is not meaning-checked (MetricX needs a GPU); run V2 on Colab',
                         'V2 selection uses MetricX order only (no LLM ranking) on V1 candidates'],
              'units': [{'unit_id': u['unit_id'], 'source': src[u['unit_id']],
                         'v1': {'text': u['natural_ar'], 'findings': before[u['unit_id']]},
                         'v2_select': selected.get(u['unit_id']),
                         'v2_polish_preview': polished.get(u['unit_id'])} for u in final['units']]}
    (out / 'v2_preview.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
