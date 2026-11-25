"""Source preparation before translation (V2): fix ASR artefacts that the translation would copy.

merge_fragments   : the ASR cut a sentence in two ("So if you take X1 and" | "you know, put that ...").
                    A unit is merged with the next one when it does not end a sentence (no . ? !) or the
                    next one starts in lower case. The merged unit keeps the ids of its parts
                    ("tu_0028+tu_0029") and the sum of their time windows, so the TTS stage can place it.
spelled_digit_flags: the ASR may write a number digit by digit ("1, 2, 2, 8, 8" for 12288). Four or more
                    single digits separated by commas are only FLAGGED; the source is never rewritten,
                    because the same pattern is also a real enumeration ("Steps 1, 2, 3, 4").
"""
import re

from .timing import Window

_END = re.compile(r'[.?!]["\')\]]?\s*$')
_SPELLED = re.compile(r'(?<![\d.,])\b(\d)(?:\s*,\s*(\d)\b){3,}(?!\.\d|\d)')


def is_fragment(text, next_text):
    t, n = (text or '').strip(), (next_text or '').strip()
    return bool(t) and bool(n) and (not _END.search(t) or n[:1].islower())


def merge_fragments(tasks, windows, *, max_words=120):
    """Returns (tasks, windows, merges). merges = [{'unit_id': 'a+b', 'merged_from': ['a', 'b']}]."""
    out_t, out_w, merges = [], {}, []
    i = 0
    while i < len(tasks):
        group = [tasks[i]]
        while (i + 1 < len(tasks) and is_fragment(group[-1]['source_text'], tasks[i + 1]['source_text'])
               and sum(len(t['source_text'].split()) for t in group + [tasks[i + 1]]) <= max_words):
            i += 1
            group.append(tasks[i])
        i += 1
        if len(group) == 1:
            out_t.append(group[0])
            out_w[group[0]['unit_id']] = windows[group[0]['unit_id']]
            continue
        ids = [t['unit_id'] for t in group]
        uid = '+'.join(ids)
        merged = {**group[0], 'unit_id': uid, 'merged_from': ids,
                  'source_text': ' '.join(t['source_text'].strip() for t in group),
                  'context_after': group[-1].get('context_after') or [],
                  'source_duration_s': sum(t.get('source_duration_s') or 0 for t in group)}
        out_t.append(merged)
        ws = [windows[x] for x in ids]
        out_w[uid] = Window(uid, round(sum(w.slot_s for w in ws), 3), round(sum(w.window_s for w in ws), 3))
        merges.append({'unit_id': uid, 'merged_from': ids})
    return out_t, out_w, merges


def find_spelled_digits(text):
    """Spans that look like a number read digit by digit; the text itself is not changed."""
    return [m.group(0) for m in _SPELLED.finditer(text or '')]


def prepare_sources(tasks, windows, cfg):
    """Apply the [segmentation] options of the config. Returns (tasks, windows, report)."""
    scfg = cfg.get('segmentation') or {}
    if scfg.get('fix_spelled_digits'):
        raise ValueError('segmentation.fix_spelled_digits was removed: the source is never rewritten. '
                         'Use spelled_digits = "flag".')
    report = {'merges': [], 'spelled_digit_flags': {}}
    if scfg.get('spelled_digits', 'flag') == 'flag':
        for t in tasks:
            spans = find_spelled_digits(t['source_text'])
            if spans:
                report['spelled_digit_flags'][t['unit_id']] = spans
    if scfg.get('merge_fragments'):
        tasks, windows, report['merges'] = merge_fragments(tasks, windows, max_words=scfg.get('max_merged_words', 120))
    return tasks, windows, report
