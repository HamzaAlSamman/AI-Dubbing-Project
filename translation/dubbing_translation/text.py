"""Arabic text utilities: letter counting, output cleanup, validity and TTS text.

Letter counting matches the calibration of the speaking rate (Arabic letters only,
diacritics and tatweel removed), so predicted durations stay comparable across runs.
"""
import re
from pathlib import Path

_AR_MARKS = re.compile('[ً-ْٰـ]')       # harakat, superscript alef, tatweel
_AR_WORD = re.compile('[ء-ي]+')
_LATIN_TOKEN = re.compile(r'(?<![A-Za-z0-9_])([A-Za-z]+)(\d*)(?![A-Za-z0-9_])')


def ar_tokens(text):
    return _AR_WORD.findall(_AR_MARKS.sub('', text or ''))


def n_letters(text):
    return len(''.join(ar_tokens(text)))


def arabic_fraction(text):
    letters = [c for c in (text or '') if c.isalpha()]
    return sum('؀' <= c <= 'ۿ' for c in letters) / max(1, len(letters))


def validity(text, source_text, *, min_arabic=0.6, ratio_range=(0.4, 1.8)):
    """Reasons why an output cannot be a translation of this unit ([] = valid).

    Catches degenerate text and outputs whose length is implausible for the unit
    (e.g. a model that translated the surrounding context as well)."""
    reasons = []
    if not (text or '').strip():
        return ['empty']
    if arabic_fraction(text) < min_arabic:
        reasons.append('not_arabic')
    ratio = len(ar_tokens(text)) / max(1, len((source_text or '').split()))
    if ratio > ratio_range[1]:
        reasons.append('too_long_for_unit')
    if ratio < ratio_range[0]:
        reasons.append('too_short_for_unit')
    return reasons


# ---------------------------------------------------------------- cleanup of model output
_QUOTED_LATIN = re.compile(r'[“”"\'«»`]\s*([A-Za-z][A-Za-z0-9_.]*(?:\s*[×x,،]\s*[A-Za-z0-9_.]+)*)\s*[“”"\'«»`]')
_FILLER_PAIRS = (
    (re.compile(r'^\s*(?:إذًا|إذاً|إذن|اذا|إذا)\s*[،,]\s*(إذا)\b'), r'\1'),   # "إذًا، إذا" -> "إذا"
    (re.compile(r'^\s*(?:لذا|لذلك)\s*[،,]\s*(لذا|لذلك)\b'), r'\1'),
)


# full-width / CJK punctuation some models emit inside Arabic text
_PUNCT = str.maketrans({'，': '،', '；': '؛', '：': ':', '？': '؟', '！': '!', '（': '(', '）': ')'})


def clean_translation(text):
    """Deterministic cleanup applied to every candidate before scoring.

    - code fences / surrounding quotes from chatty models
    - backticks and quotes around variables (`x`, “Y”) -> plain x, Y
    - doubled discourse openers ("إذًا، إذا ...") that cost time and sound redundant
    - full-width punctuation ("，") -> Arabic punctuation
    """
    t = (text or '').translate(_PUNCT).strip()
    if t.startswith('```'):
        t = re.sub(r'^```\w*\s*', '', t).rsplit('```', 1)[0].strip()
    if len(t) > 2 and t[0] in '"“«' and t[-1] in '"”»':
        t = t[1:-1].strip()
    t = _QUOTED_LATIN.sub(r'\1', t)
    t = t.replace('`', '')
    for pattern, repl in _FILLER_PAIRS:
        t = pattern.sub(repl, t)
    return re.sub(r'[ \t]+', ' ', t).strip()


# ---------------------------------------------------------------- text sent to the TTS voice
LETTER_NAMES = {
    'a': 'إيه', 'b': 'بي', 'c': 'سي', 'd': 'دي', 'e': 'إي', 'f': 'إف', 'g': 'جي', 'h': 'إتش',
    'i': 'آي', 'j': 'جيه', 'k': 'كيه', 'l': 'إل', 'm': 'إم', 'n': 'إن', 'o': 'أو', 'p': 'بي',
    'q': 'كيو', 'r': 'آر', 's': 'إس', 't': 'تي', 'u': 'يو', 'v': 'في', 'w': 'دبليو', 'x': 'إكس',
    'y': 'واي', 'z': 'زد',
}
KNOWN_WORDS = {
    'python': 'بايثون', 'numpy': 'نمباي', 'shape': 'شيب', 'reshape': 'ري شيب', 'dot': 'دوت',
    'sigmoid': 'سيغمويد', 'relu': 'ريلو', 'rgb': 'آر جي بي', 'gpu': 'جي بي يو', 'cpu': 'سي بي يو',
}
_DIGITS_AR = {'0': 'صفر', '1': 'واحد', '2': 'اثنان', '3': 'ثلاثة', '4': 'أربعة', '5': 'خمسة',
              '6': 'ستة', '7': 'سبعة', '8': 'ثمانية', '9': 'تسعة'}
_INDIC_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')
_ENUM = re.compile(r'(\b[A-Za-z]+\d*)\s*[,،]\s*([A-Za-z]+\d*)\s*[,،]\s*(?:…|\.\.\.)\s*[,،]?\s*([A-Za-z]+\d*\b)')
_DIM = re.compile(r'(?<=\d)\s*[xX×]\s*(?=\d)')


def _number_words(digits):
    try:
        from num2words import num2words
        return num2words(int(digits), lang='ar') if len(digits) <= 9 else digits
    except Exception:            # num2words is optional; spell digit by digit
        return ' '.join(_DIGITS_AR[d] for d in digits)


def _speak_latin(match):
    word, digits = match.group(1), match.group(2)
    low = word.lower()
    if low in KNOWN_WORDS:
        spoken = KNOWN_WORDS[low]
    elif len(word) <= 3:                       # variables: x, nx, Y, ym -> letter names
        spoken = ' '.join(LETTER_NAMES[c] for c in low)
    else:
        return match.group(0)                  # longer English words are left to the voice
    if digits:
        spoken += ' ' + _number_words(digits)
    return spoken


def tts_text(text, version='v1'):
    """Arabic text as it should be spoken by an Arabic-only TTS voice. The written translation is not
    changed; this is a separate string.

    version 'v1' : the original reader (kept unchanged so V1 runs reproduce; numbers use num2words when it
                   is installed and digit-by-digit otherwise, so its output depends on the environment).
    version 'v2' : one deterministic reader (arabic_numbers.py) and the pronunciation lexicon
                   (configs/pronunciation/common_v1.toml); signs, decimals, percents, dates and simple
                   equations are read as words. Selected by [voice] spoken_version in the config."""
    if version == 'v2':
        return _tts_v2(text)
    if version != 'v1':
        raise ValueError(f'unknown spoken-text version {version!r}')
    t = clean_translation(text).translate(_INDIC_DIGITS)
    t = re.sub(r'[*_#«»“”"]', ' ', t)
    t = _ENUM.sub(r'\1 و\2 وصولًا إلى \3', t)
    t = _DIM.sub(' في ', t)
    t = t.replace('×', ' في ').replace('=', ' يساوي ').replace('…', '، ')
    t = re.sub(r'(?<=[A-Za-z0-9])\.(?=[A-Za-z])', ' دوت ', t)   # x.shape
    t = _LATIN_TOKEN.sub(_speak_latin, t)
    t = re.sub(r'\d+', lambda m: _number_words(m.group()), t)
    t = re.sub(r'\s*([،,.؟?!])\s*', r'\1 ', t)
    return re.sub(r'\s+', ' ', t).strip()



# ---------------------------------------------------------------- spoken text v2
_LEXICON = None
_SPOKEN_V2_NUMBER = re.compile(r'(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])\d+(?:\.\d+)?(?![\w])')
_DATE = re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b')
_ISO_DATE = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')
_PERCENT = re.compile(r'(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)\s*%')
_MINUS = re.compile(r'[-−]\s*(?=\d)')
_ACRONYM = re.compile(r'^[A-Z]{2,}$')


def pronunciation_lexicon():
    """{lower-case latin token: arabic spoken form} from configs/pronunciation/common_v1.toml."""
    global _LEXICON
    if _LEXICON is None:
        import tomllib
        path = Path(__file__).resolve().parent.parent / 'configs' / 'pronunciation' / 'common_v1.toml'
        _LEXICON = {k.lower(): v for k, v in tomllib.loads(path.read_text(encoding='utf-8'))['words'].items()}
    return _LEXICON


def _speak_latin_v2(match):
    from .arabic_numbers import number_words
    word, digits = match.group(1), match.group(2)
    lex = pronunciation_lexicon()
    low = word.lower()
    if low in lex:
        spoken = lex[low]
    elif _ACRONYM.match(word) or len(word) <= 3:     # acronyms and variables: letter by letter
        spoken = ' '.join(LETTER_NAMES[c] for c in low)
    else:
        return match.group(0)                         # other English words are left to the voice
    if digits:
        spoken += ' ' + number_words(int(digits))
    return spoken


def _minus(match_obj, text):
    """Unary minus ('= -5', start) -> سالب ; binary minus ('x - 5') -> ناقص."""
    i = match_obj.start()
    prev = text[:i].rstrip()
    return ' ناقص ' if prev and (prev[-1].isalnum() or prev[-1] == ')') else ' سالب '


def _tts_v2(text):
    from .arabic_numbers import read_number
    t = clean_translation(text).translate(_INDIC_DIGITS)
    t = re.sub(r'[*_#«»“”"]', ' ', t)
    t = _ENUM.sub(r'\1 و\2 وصولًا إلى \3', t)
    t = _DATE.sub(lambda m: f'{read_number(m.group(1))}، {read_number(m.group(2))}، {read_number(m.group(3))}', t)
    t = _ISO_DATE.sub(lambda m: f'{read_number(m.group(3))}، {read_number(m.group(2))}، {read_number(m.group(1))}', t)
    t = _DIM.sub(' في ', t)
    t = _PERCENT.sub(lambda m: read_number(m.group(1)) + ' بالمئة', t)
    t = _MINUS.sub(lambda m: _minus(m, t), t)
    for sym, word in (('≤', ' أصغر من أو يساوي '), ('≥', ' أكبر من أو يساوي '), ('×', ' في '),
                      ('=', ' يساوي '), ('+', ' زائد '), ('<', ' أصغر من '), ('>', ' أكبر من '),
                      ('…', '، ')):
        t = t.replace(sym, word)
    t = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', t)                           # 2x -> 2 x
    t = re.sub(r'(?<=[A-Za-z0-9])\.(?=[A-Za-z])', ' دوت ', t)            # x.shape, np.dot
    t = _LATIN_TOKEN.sub(_speak_latin_v2, t)
    t = _SPOKEN_V2_NUMBER.sub(lambda m: read_number(m.group()), t)
    t = re.sub(r'\s*([،,.؟?!])\s*', r'\1 ', t)
    return re.sub(r'\s+', ' ', t).strip(' ،,')
