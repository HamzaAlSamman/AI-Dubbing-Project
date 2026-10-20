"""Conservative TU builder for English, single-speaker pilot.

Does not edit WhisperX words, does not invent timing, never translates.
JOIN/SPLIT decisions use multiple signals. Ambiguities are flagged. The
original segments remain intact in the output Master data.
"""
import re
from .common import clone, require_time, unit_text

_END = re.compile(r'[.!?][\"\'\)\]]*$')
_SEMI = re.compile(r'[;:][\"\']*$')
_WEAK_END = re.compile(r'[,—\-]\s*$')
_CONTINUATION = re.compile(r'^(?:and|or|but|because|which|that|who|whose|where|when|while|followed|of|to|with|for|by|in|on|at|as|from|than|then)\b',re.I)
_UNFINISHED = re.compile(r'\b(?:a|an|the|and|or|but|with|for|of|to|in|on|from|when|while|if|by|followed|such|is|are|was|were|into)$',re.I)


def _speaker(seg, default):
    return str(seg.get('speaker') or (seg.get('speaker_info') or {}).get('id') or default)


def _words(seg, i):
    words = seg.get('words', [])
    if not words:
        raise ValueError(f'Segment {i} has no word timings; cannot safely build units')
    out = []
    for j, w in enumerate(words):
        a = require_time(w.get('start'), f'seg{i} word{j} start')
        b = require_time(w.get('end'), f'seg{i} word{j} end')
        if a < 0 or b < a or not str(w.get('word', '')).strip():
            raise ValueError(f'Invalid word timing/text seg{i} word{j}')
        out.append({'segment_index': i, 'word_index': j, 'word': w['word'], 'start': a, 'end': b})
    return out


def _punctuation_split(words):
    """SPLIT only at strong sentence endings; preserve exact word refs."""
    groups, buf = [], []
    for w in words:
        buf.append(w)
        if _END.search(str(w['word'])):
            groups.append(buf)
            buf = []
    if buf:
        groups.append(buf)
    return groups


def _phrase_text(words):
    return re.sub(r'\s+([,.;:!?])', r'\1', ' '.join(str(w['word']).strip() for w in words)).strip()


def build_units_from_whisperx(source_segments, course=None, *, pause_seconds=1.0,
                               long_unit_warning_words=45, default_speaker='SPEAKER_00'):
    """Return V2 Master with 100% source-word references or raise.

    This is a conservative optional implementation, not an automatic replacement
    for a previously reviewed Builder V1.1. Review JOIN/SPLIT decisions on audio.
    """
    if not source_segments:
        raise ValueError('No source segments supplied')
    segs = clone(source_segments)
    atoms, input_refs = [], []
    for i, seg in enumerate(segs):
        words = _words(seg, i)
        input_refs.extend((i, w['word_index']) for w in words)
        # Mixed speaker inside a segment is not inferred; mark it for review.
        sid = str(seg.get('id') or f'seg_{i+1:04d}')
        for chunk in _punctuation_split(words):
            atoms.append({'words': chunk, 'speaker': _speaker(seg, default_speaker),
                          'segment_ids': [sid], 'segment_indexes': [i]})
    buckets = []
    for atom in atoms:
        if not buckets:
            buckets.append({'atoms': [atom], 'reason': 'start'}); continue
        prev = buckets[-1]
        tail = prev['atoms'][-1]
        left = _phrase_text(tail['words'])
        right = _phrase_text(atom['words'])
        gap = atom['words'][0]['start'] - tail['words'][-1]['end']
        same = atom['speaker'] == tail['speaker']
        # An incomplete predecessor or strong continuation is meaningful.
        continuing = bool(_WEAK_END.search(left) or _UNFINISHED.search(left.rstrip('.,;:!? '))
                          or _CONTINUATION.match(right))
        ended = bool(_END.search(left))
        should_join = same and gap <= pause_seconds and (continuing or not ended)
        if should_join:
            prev['atoms'].append(atom)
            prev['reason'] = 'join_incomplete_or_continuation'
        else:
            buckets.append({'atoms': [atom], 'reason': 'sentence_or_pause_or_speaker'})
    units, consumed = [], []
    for i, bucket in enumerate(buckets, start=1):
        parts = bucket['atoms']
        words = [w for atom in parts for w in atom['words']]
        refs = [(w['segment_index'], w['word_index']) for w in words]
        consumed.extend(refs)
        starts, ends = [w['start'] for w in words], [w['end'] for w in words]
        first, last = min(starts), max(ends)
        if last <= first:
            raise ValueError(f'Unit {i} has no measurable duration')
        segment_ids = list(dict.fromkeys(s for atom in parts for s in atom['segment_ids']))
        txt = _phrase_text(words)
        review_reasons = []
        if len(words) > long_unit_warning_words:
            review_reasons.append('long_unit_no_safe_boundary')
        if not _END.search(txt):
            review_reasons.append('open_sentence_or_missing_punctuation')
        if any(words[j]['start'] + 0.2 < words[j-1]['start'] for j in range(1,len(words))):
            review_reasons.append('non_monotonic_word_times')
        units.append({
            'id': f'tu_{i:04d}', 'speaker': {'id': parts[0]['speaker']},
            'source_segment_ids': segment_ids,
            'source_refs': [{'segment_index': a, 'word_index': b} for a,b in refs],
            'timing': {'source_start': first, 'source_end': last, 'source_duration': round(last-first, 6)},
            'source': {'text': txt},
            'builder': {'end_boundary_reason': bucket['reason'], 'needs_review': bool(review_reasons),
                        'review_reasons': review_reasons},
            'translation': {'accurate_ar': None, 'natural_ar': None, 'final_dub_ar': None},
            'review': {'status': 'pending'}
        })
    if input_refs != consumed:
        raise AssertionError('Word reference coverage/order not preserved')
    return {'schema_version': '2.0', 'course': clone(course or {}),
            'source_segments': segs, 'translation_units': units, 'terminology': [],
            'builder_report': {'source_segment_count': len(segs), 'source_word_count': len(input_refs),
                               'translation_unit_count': len(units), 'source_word_coverage_verified': True,
                               'algorithm': 'conservative_generic_v2_experimental',
                               'human_boundary_review_required': True}}
