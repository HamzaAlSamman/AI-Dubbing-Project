"""Independent check of a run with an LLM judge (GEMBA-style), blind and pairwise.

The selection inside the step is driven by MetricX; scoring the result with MetricX again would
reward the selector for its own preferences. Here a model from another family compares two systems
unit by unit: order randomised per unit, system names hidden, adequacy and fluency scored 0-100,
plus an overall preference. Results are cached one JSON line per unit (resumable within quotas).
"""
import json
import random
from pathlib import Path

JUDGE_VERSION = 'pairwise_gemba_v1'
JUDGE_SYSTEM = (
    "You are a professional English to Arabic translator reviewing the script of an Arabic voice-over for an "
    "educational video. You get one English segment and two Arabic translations, A and B.\n"
    "Score each translation from 0 to 100 for:\n"
    "- adequacy: the full meaning is conveyed correctly (facts, numbers, variables, negation, technical terms); "
    "omissions and additions lower the score;\n"
    "- fluency: natural, grammatical Modern Standard Arabic suitable to be spoken aloud.\n"
    "Then choose which one is better overall for the voice-over, or 'tie' if they are equally good. "
    "Shorter is fine as long as no meaning is lost; spoken filler like 'you know' may be dropped. "
    "Variables may stay in Latin letters.\n"
    'Return only JSON: {"A": {"adequacy": 0, "fluency": 0}, "B": {"adequacy": 0, "fluency": 0}, '
    '"better": "A" | "B" | "tie", "reason": "one short sentence"}'
)


def pairwise(tasks, system_a, system_b, client, out_path, *, seed=13, log=print):
    """system_a/system_b: {unit_id: arabic_text}. Returns summary dict; per-unit records in out_path (jsonl)."""
    out_path = Path(out_path)
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec['unit_id']] = rec
    rng = random.Random(seed)
    order = {t['unit_id']: rng.random() < 0.5 for t in tasks}          # True -> system_a shown as "A"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        u = t['unit_id']
        a, b = system_a.get(u), system_b.get(u)
        if u in done or not a or not b:
            continue
        if a.strip() == b.strip():
            rec = {'unit_id': u, 'identical': True, 'winner': 'tie'}
        else:
            first, second = (a, b) if order[u] else (b, a)
            reply = client.chat_json(JUDGE_SYSTEM, json.dumps({'english': t['source_text'], 'A': first, 'B': second},
                                                              ensure_ascii=False))
            if not reply or reply.get('better') not in ('A', 'B', 'tie'):
                log(f'[judge] {u}: unusable reply')
                continue
            sa, sb = (reply['A'], reply['B']) if order[u] else (reply['B'], reply['A'])
            pick = reply['better']
            winner = 'tie' if pick == 'tie' else ('a' if (pick == 'A') == order[u] else 'b')
            rec = {'unit_id': u, 'identical': False, 'winner': winner, 'a': sa, 'b': sb,
                   'reason': reply.get('reason'), 'judge': f'{client.provider}:{client.model}',
                   'version': JUDGE_VERSION}
        with out_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        done[u] = rec
    return summarize(done.values())


def summarize(records):
    recs = list(records)
    judged = [r for r in recs if not r.get('identical')]

    def mean(side, key):
        vals = [float(r[side][key]) for r in judged if isinstance(r.get(side), dict) and key in r[side]]
        return round(sum(vals) / len(vals), 1) if vals else None

    return {'units': len(recs), 'identical': len(recs) - len(judged),
            'a_wins': sum(r['winner'] == 'a' for r in judged), 'b_wins': sum(r['winner'] == 'b' for r in judged),
            'ties': sum(r['winner'] == 'tie' for r in judged),
            'adequacy_a': mean('a', 'adequacy'), 'adequacy_b': mean('b', 'adequacy'),
            'fluency_a': mean('a', 'fluency'), 'fluency_b': mean('b', 'fluency')}
