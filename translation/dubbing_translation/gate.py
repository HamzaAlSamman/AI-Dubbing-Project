"""Acceptance gate for one candidate translation.

A candidate is ACCEPTED when all checks pass:
  validity      : mostly Arabic, plausible length for the unit
  qe            : reference-free QE (MetricX) <= qe_max
  spelling      : misspelled-word rate <= spelling_max (technical terms on the allow-list excluded)
  mechanical    : numbers and negation of the source are kept
  duration      : predicted speech fits its window (natural or with bounded speed-up)
Thresholds are heuristics documented in the config and the run manifest; they are not
calibrated against human judgements.
"""
import re

from .text import ar_tokens, n_letters, tts_text, validity
from .timing import duration_verdict

# Only Latin letters/digits block a match: Arabic proclitics attach to numbers ("و231" = "and 231").
_NUMBER = re.compile(r'(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?%?(?![A-Za-z0-9_])')
_AR_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')
_NEG_EN = re.compile(r"\b(?:no|not|never|neither|nor|without|cannot|can't|don't|doesn't|won't|isn't|aren't|"
                     r"shouldn't|mustn't|wasn't|weren't|didn't|haven't|hasn't)\b", re.I)
_NEG_AR = re.compile(r'(?:^|[\s،,.])(?:لا|لم|لن|ليس|ليست|ليسوا|دون|بدون|غير|ما|عدم|بلا)(?=[\s،,.]|$)')


def numbers(text):
    return sorted(_NUMBER.findall((text or '').translate(_AR_DIGITS)))


def spoken_letters(text):
    return n_letters(tts_text(text))


def mechanical_issues(source, arabic):
    issues = []
    src_nums, ar_nums = numbers(source), numbers(arabic)
    missing = [n for n in src_nums if n not in ar_nums]
    if missing:
        issues.append('numbers_missing:' + ','.join(missing))
    if _NEG_EN.search(source or '') and not _NEG_AR.search(arabic or ''):
        issues.append('negation_lost')
    return issues


def evaluate(text, source, window, *, qe, misspelled_words, cfg, allowlist=frozenset()):
    """Return a dict with every measurement and the list of failed checks ('reasons')."""
    gate, rate = cfg['gate'], cfg['voice']['letters_per_second']
    toks = ar_tokens(text)
    bad = [w for w in toks if w in misspelled_words and w not in allowlist]
    spelling_rate = len(bad) / max(1, len(toks))
    # Duration is predicted on what the voice will actually say: variables and symbols are
    # spoken as Arabic words (x -> إكس), so they take time although they are not Arabic letters.
    spoken = spoken_letters(text) if cfg['voice'].get('count_spoken_form', True) else n_letters(text)
    dur = duration_verdict(spoken, window, rate, min_ratio=gate['min_ratio'],
                           max_speedup=gate['max_speedup'])
    reasons = []
    reasons += ['invalid:' + r for r in validity(text, source)]
    if qe is None or qe > gate['qe_max']:
        reasons.append('qe')
    if spelling_rate > gate['spelling_max']:
        reasons.append('spelling')
    reasons += mechanical_issues(source, text)
    if dur['zone'] in ('too_long', 'too_short'):
        reasons.append('duration_' + dur['zone'])
    return {'qe': None if qe is None else round(qe, 3), 'spelling_rate': round(spelling_rate, 3),
            'misspelled': bad, 'letters': n_letters(text), 'spoken_letters': spoken, **dur, 'reasons': reasons,
            'accepted': not reasons}


def only_duration_failed(result):
    return bool(result['reasons']) and all(r.startswith('duration_') for r in result['reasons'])


def rank_key(result):
    """Among accepted candidates: best QE; prefer no speed-up; then closest to a full slot."""
    return (result['qe'] if result['qe'] is not None else 99.0,
            result['zone'] == 'speedup',
            abs(1.0 - result['ratio_slot']))
