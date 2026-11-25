"""Source preparation before translation (V2): fix ASR artefacts that the translation would copy.

merge_fragments   : the ASR cut a sentence in two ("So if you take X1 and" | "you know, put that ...").
                    Two units are merged only when BOTH text signals agree (the first does not end a
                    sentence AND the next starts in lower case or with a digit) and, when timings are known,
                    the pause between them is short. One signal alone ("...xm." | "transpose, ...", or
                    "That is it." | "iPhone apps ...") is reported as a suspected fragment and NOT merged.
                    A merged unit gets a derived id ("a+b"), keeps the ids, texts and timings of its parts,
                    and its text is the parts joined in order (no word lost, none reordered).
spelled_digit_flags: the ASR may write a number digit by digit ("1, 2, 2, 8, 8" for 12288). Four or more
                    single digits separated by commas are only FLAGGED; the source is never rewritten,
                    because the same pattern is also a real enumeration ("Steps 1, 2, 3, 4").
"""
import re

from .timing import Window

_END = re.compile(r'[.?!]["\')\]]?\s*$')
_SPELLED = re.compile(r'(?<![\d.,])\b(\d)(?:\s*,\s*(\d)\b){3,}(?!\.\d|\d)')


def fragment_signals(text, next_text):
    """(no sentence end, next continues): the two text signals of a cut sentence."""
    t, n = (text or '').strip(), (next_text or '').strip()
    if not t or not n:
        return False, False
    return not _END.search(t), n[:1].islower() or n[:1].isdigit()


def is_fragment(text, next_text):
    """Both text signals agree (kept for callers that have no timings)."""
    return all(fragment_signals(text, next_text))


def merge_fragments(tasks, windows, *, timings=None, max_words=120, max_pause_s=0.6):
    """Returns (tasks, windows, merges, suspects).
    merges   = [{'unit_id': 'a+b', 'merged_from': ['a', 'b']}]
    suspects = [{'units': ['a', 'b'], 'reason': '...'}]  doubtful cuts, left unmerged."""
    timings = timings or {}
    out_t, out_w, merges, suspects = [], {}, [], []

    def decide(prev, nxt, words):
        no_end, cont = fragment_signals(prev['source_text'], nxt['source_text'])
        if not (no_end or cont):
            return False, None
        if not (no_end and cont):
            why = 'sentence end but next starts in lower case' if cont else 'no sentence end but next starts a sentence'
            return False, why
        a, b = timings.get(prev['unit_id']), timings.get(nxt['unit_id'])
        if a and b and b[0] - a[1] > max_pause_s:
            return False, f'pause of {b[0] - a[1]:.2f}s between the parts'
        if words > max_words:
            return False, f'merged unit would have {words} words'
        return True, None

    i = 0
    while i < len(tasks):
        group = [tasks[i]]
        while i + 1 < len(tasks):
            words = sum(len(t['source_text'].split()) for t in group + [tasks[i + 1]])
            ok, why = decide(group[-1], tasks[i + 1], words)
            if not ok:
                if why:
                    suspects.append({'units': [group[-1]['unit_id'], tasks[i + 1]['unit_id']], 'reason': why})
                break
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
                  'source_parts': [{'unit_id': t['unit_id'], 'source_text': t['source_text']} for t in group],
                  'context_after': group[-1].get('context_after') or [],
                  'source_duration_s': sum(t.get('source_duration_s') or 0 for t in group)}
        out_t.append(merged)
        ws = [windows[x] for x in ids]
        out_w[uid] = Window(uid, round(sum(w.slot_s for w in ws), 3), round(sum(w.window_s for w in ws), 3))
        merges.append({'unit_id': uid, 'merged_from': ids})
    return out_t, out_w, merges, suspects


def find_spelled_digits(text):
    """Spans that look like a number read digit by digit; the text itself is not changed."""
    return [m.group(0) for m in _SPELLED.finditer(text or '')]


def prepare_sources(tasks, windows, cfg, *, timings=None):
    """Apply the [segmentation] options of the config. Returns (tasks, windows, report).
    timings: {unit_id: (start_s, end_s)} of the original units, used to refuse merges across long pauses."""
    scfg = cfg.get('segmentation') or {}
    if scfg.get('fix_spelled_digits'):
        raise ValueError('segmentation.fix_spelled_digits was removed: the source is never rewritten. '
                         'Use spelled_digits = "flag".')
    report = {'merges': [], 'suspected_fragments': [], 'spelled_digit_flags': {}}
    if scfg.get('spelled_digits', 'flag') == 'flag':
        for t in tasks:
            spans = find_spelled_digits(t['source_text'])
            if spans:
                report['spelled_digit_flags'][t['unit_id']] = spans
    if scfg.get('merge_fragments'):
        tasks, windows, report['merges'], report['suspected_fragments'] = merge_fragments(
            tasks, windows, timings=timings, max_words=scfg.get('max_merged_words', 120),
            max_pause_s=scfg.get('max_merge_pause_s', 0.6))
    return tasks, windows, report
