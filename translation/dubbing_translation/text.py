"""Arabic text utilities: letter counting, output cleanup, validity and TTS text.

Letter counting matches the calibration of the speaking rate (Arabic letters only,
diacritics and tatweel removed), so predicted durations stay comparable across runs.
"""
import re

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


def tts_text(text):
    """Arabic text as it should be spoken by an Arabic-only TTS voice.

    Variables are spelled in Arabic (x -> إكس, nx -> إن إكس), × and = are read as words,
    enumerations 'a1, a2, …, am' become 'a1 و a2 وصولًا إلى am', markup and quotes are dropped,
    digits become Arabic number words. The written translation is not changed."""
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
