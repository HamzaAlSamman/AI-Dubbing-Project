"""Arabic number words for the voice, with no third-party dependency.

One reader for every environment: the spoken form (and so the predicted duration) of a number is the
same on a laptop, Colab and Kaggle. Cardinal numbers are read in the masculine form, as a standalone
number is read aloud (the counted noun is not known here):
  12288 -> اثنا عشر ألفًا ومئتان وثمانية وثمانون   2026 -> ألفان وستة وعشرون   -5 -> سالب خمسة
  3.14 -> ثلاثة فاصلة واحد أربعة   (decimals digit by digit)
Numbers of 13 digits or more are read digit by digit.
"""

ONES = ['صفر', 'واحد', 'اثنان', 'ثلاثة', 'أربعة', 'خمسة', 'ستة', 'سبعة', 'ثمانية', 'تسعة']
TENS = {20: 'عشرون', 30: 'ثلاثون', 40: 'أربعون', 50: 'خمسون', 60: 'ستون', 70: 'سبعون', 80: 'ثمانون',
        90: 'تسعون'}
HUNDREDS = ['', 'مئة', 'مئتان', 'ثلاثمئة', 'أربعمئة', 'خمسمئة', 'ستمئة', 'سبعمئة', 'ثمانمئة', 'تسعمئة']
# (value, one, two, 3..10, 11..99)
SCALES = [(10 ** 9, 'مليار', 'ملياران', 'مليارات', 'مليارًا'),
          (10 ** 6, 'مليون', 'مليونان', 'ملايين', 'مليونًا'),
          (10 ** 3, 'ألف', 'ألفان', 'آلاف', 'ألفًا')]


def _below_100(n):
    if n < 10:
        return ONES[n]
    if n == 10:
        return 'عشرة'
    if n == 11:
        return 'أحد عشر'
    if n == 12:
        return 'اثنا عشر'
    if n < 20:
        return ONES[n - 10] + ' عشر'
    t, u = (n // 10) * 10, n % 10
    return TENS[t] if u == 0 else f'{ONES[u]} و{TENS[t]}'


def _below_1000(n):
    h, r = divmod(n, 100)
    parts = ([HUNDREDS[h]] if h else []) + ([_below_100(r)] if r else [])
    return ' و'.join(parts)


def _scaled(n, one, two, few, many):
    """n (1..999) thousands / millions / billions."""
    if n == 1:
        return one
    if n == 2:
        return two
    r = n % 100
    if 3 <= n <= 10 or (n > 100 and 3 <= r <= 10):
        return f'{_below_1000(n)} {few}'
    if r >= 11:
        return f'{_below_1000(n)} {many}'
    if r == 0 and n // 100 == 2:
        return f'مئتا {one}'                                  # 200 000 -> مئتا ألف
    return f'{_below_1000(n)} {one}'


def number_words(n):
    """Non-negative integer -> Arabic words."""
    n = int(n)
    if n == 0:
        return ONES[0]
    if n >= 10 ** 12:
        return ' '.join(ONES[int(d)] for d in str(n))
    parts = []
    for value, one, two, few, many in SCALES:
        q, n = divmod(n, value)
        if q:
            parts.append(_scaled(q, one, two, few, many))
    if n:
        parts.append(_below_1000(n))
    return ' و'.join(parts)


def read_number(token):
    """'12,288', '-5', '3.14', '0.5' -> Arabic words (sign and decimal point included)."""
    t = token.strip().replace('٫', '.').replace('٬', ',')
    sign = ''
    if t.startswith(('-', '−')):
        sign, t = 'سالب ', t[1:]
    t = t.replace(',', '')
    if '.' in t:
        whole, frac = t.split('.', 1)
        words = number_words(int(whole or 0)) + ' فاصلة ' + ' '.join(ONES[int(d)] for d in frac if d.isdigit())
    else:
        words = number_words(int(t))
    return sign + words
