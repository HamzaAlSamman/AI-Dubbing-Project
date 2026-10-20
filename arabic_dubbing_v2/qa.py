"""Conservative educational QA: mechanical checks + explicit UNKNOWNs.

This module does not claim to verify semantic equivalence without a separate
bilingual audit. It never approves final_dub_ar on its own.
"""
import re
from .common import unit_text

_NUMBER = re.compile(r'(?<![\w])[-+]?\d+(?:[.,]\d+)?%?(?![\w])')
_ARABIC_NUM_MAP = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹','01234567890123456789')
_CODE = re.compile(r'`([^`]+)`|\b([A-Za-z_$][A-Za-z_0-9$]*\(\))')
_SYMBOLS = re.compile(r'(?<![a-zA-Z])(?:[A-Za-z]+_\w+|[A-Za-z][A-Za-z\d]*\.[A-Za-z_][\w]*)(?![A-Za-z])')
_NEG_EN = re.compile(r"\b(?:no|not|never|neither|without|cannot|can't|don't|doesn't|won't|isn't|aren't|shouldn't|mustn't)\b", re.I)
_NEG_AR = re.compile(r'\b(?:لا|لم|لن|ليس|ليست|ليسوا|دون|بدون|غير|ما)\b')


def _num(text):
    return _NUMBER.findall(text.translate(_ARABIC_NUM_MAP))


def _code(text):
    spans=[m.group(1) or m.group(2) for m in _CODE.finditer(text)]
    spans.extend(_SYMBOLS.findall(text))
    return sorted(set(spans))


def mechanical_qa(source, arabic, *, builder_review=False):
    if not arabic or not arabic.strip():
        return {'status':'missing_translation', 'requires_review':True,
                'checks':{'numbers_preserved': None, 'code_preserved': None,
                          'negation_preserved':None, 'meaning_preserved':None,
                          'technical_terms_preserved':None, 'instruction_strength_preserved':None,
                          'sequence_preserved':None, 'hallucination_detected':None},
                'issues':['no_natural_ar_translation'],
                'note':'Semantic QA requires separate bilingual validation; null is unknown, not passed.'}
    numbers_before, numbers_after = _num(source), _num(arabic)
    codes_before, codes_after = _code(source), _code(arabic)
    missing_num = [n for n in numbers_before if n not in numbers_after]
    missing_code = [code for code in codes_before if code not in codes_after]
    negated_before = bool(_NEG_EN.search(source))
    negated_after = bool(_NEG_AR.search(arabic))
    issues=[]
    if missing_num: issues.append('numeric_fidelity_needs_review')
    if missing_code: issues.append('code_or_identifier_needs_review')
    if negated_before and not negated_after: issues.append('possible_lost_negation')
    if builder_review: issues.append('builder_boundary_needs_review')
    issues.append('bilingual_semantic_qa_not_performed')
    return {'status':'needs_review', 'requires_review':True,
            'checks':{
                'numbers_preserved':not bool(missing_num),
                'code_preserved':not bool(missing_code),
                'negation_preserved': None if not negated_before else negated_after,
                'meaning_preserved':None, 'technical_terms_preserved':None,
                'instruction_strength_preserved':None, 'sequence_preserved':None,
                'hallucination_detected':None},
            'issues':issues, 'missing_numbers':missing_num, 'missing_code':missing_code,
            'note':'Heuristic checks cannot substitute for bilingual semantic and domain review.'}


def qa_unit(unit, text):
    return mechanical_qa(unit_text(unit),text,
                         builder_review=bool((unit.get('builder') or {}).get('needs_review')))
