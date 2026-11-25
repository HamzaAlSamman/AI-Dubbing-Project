"""Language policy and terminology for the Arabic voice-over (Translation V2).

Two project resources, both versioned TOML files under `configs/`:
  - a language policy (simple MSA): rules shown to every LLM step, dialect words that reject a
    candidate, stiff translation patterns ("calques") that are flagged;
  - a glossary (terminology database): one approved Arabic rendering per English term of the course.

`Language.check(source, arabic)` returns what the gate needs:
  terms    : glossary terms found in the source whose approved rendering is missing in the Arabic
  dialect  : dialect words in the Arabic
  calques  : ids of stiff patterns in the Arabic
Matching is deliberately lenient (stems after normalisation), so a flag means a real deviation.
"""
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11
    tomllib = None

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'configs'

_MARKS = re.compile('[ً-ْٰـ]')     # harakat, superscript alef, tatweel
_AR_TOKEN = re.compile('[ء-ي]+')
_CLITIC = 'وفبلك'


def normalize_ar(text):
    """Comparison form: no diacritics/tatweel, أ/إ/آ -> ا, ى -> ي, ة -> ه, lower-case Latin."""
    t = _MARKS.sub('', text or '')
    t = re.sub('[أإآ]', 'ا', t).replace('ى', 'ي').replace('ة', 'ه')
    return t.lower()


def _read_toml(path):
    if tomllib is None:
        raise RuntimeError('language files need Python 3.11+ (tomllib)')
    return tomllib.loads(Path(path).read_text(encoding='utf-8'))


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() or p.exists() else CONFIG_DIR / p


_SUFFIXES = ('ات', 'ين', 'ون', 'ان', 'يه', 'ه', 'ي')


def stem_ar(word):
    """Lenient matching stem: normalised, without the article and one common suffix (keeps >= 4 letters)."""
    w = normalize_ar(word)
    for pre in ('وال', 'بال', 'لل', 'ال'):
        if w.startswith(pre) and len(w) - len(pre) >= 3:
            w = w[len(pre):]
            break
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= (3 if suf == 'ه' else 4):
            return w[:-len(suf)]
    return w


def require_from_forms(forms):
    """'require' stems from an approved rendering and its other forms, word position by position:
    ['متجه الخصائص'] -> [['متجه'], ['خصائص']]; ['مثال تدريب', 'أمثلة التدريب'] -> [['امثل', 'مثال'], ['تدريب']]."""
    split = [_AR_TOKEN.findall(_MARKS.sub('', f)) for f in forms if f]
    split = [ws for ws in split if ws]
    if not split:
        return []
    n = len(split[0])
    req = []
    for i in range(n):
        alts = sorted({stem_ar(ws[i]) for ws in split if len(ws) == n} - {''})
        if alts:
            req.append(alts)
    return req


class Glossary:
    def __init__(self, data):
        self.version = data.get('version', 'glossary')
        self.terms = []
        for t in data.get('term', []):
            pats = []
            for form in t['en']:
                if form.startswith('re:'):
                    pats.append(form[3:])
                else:
                    words = [re.escape(w) for w in form.split()]
                    pats.append(r'\b' + r'[\s-]+'.join(words) + r'(?:s|es)?\b')
            self.terms.append({
                'id': t['id'], 'en': t['en'], 'ar': t['ar'],
                'en_re': re.compile('|'.join(f'(?:{p})' for p in pats), re.I),
                'require': ([[normalize_ar(x) for x in (alt if isinstance(alt, list) else alt.split('|'))] for alt in t['require']] if t.get('require')
                            else require_from_forms([t['ar']] + list(t.get('ar_forms', [])))),
                'avoid': list(t.get('avoid', [])),
            })

    @classmethod
    def load(cls, path):
        return cls(_read_toml(_resolve(path)))

    @classmethod
    def from_terms(cls, terms, version):
        """Glossary from a list of term dicts (LLM-built); ids are derived from the English form."""
        data, seen = {'version': version, 'term': []}, set()
        for t in terms:
            en = [e for e in (t.get('en') or []) if isinstance(e, str) and e.strip()]
            ar = t.get('ar')
            if not en or not isinstance(ar, str) or not ar.strip():
                continue
            tid = re.sub(r'[^a-z0-9]+', '_', en[0].lower()).strip('_')
            if not tid or tid in seen:
                continue
            seen.add(tid)
            data['term'].append({'id': tid, 'en': en, 'ar': ar.strip(),
                                 'ar_forms': [f for f in t.get('ar_forms') or [] if isinstance(f, str)],
                                 'avoid': [a for a in t.get('avoid') or [] if isinstance(a, str)]})
        return cls(data)

    def merged(self, other):
        """This glossary plus `other`'s terms whose English it does not cover yet (this one wins)."""
        out = Glossary({'version': f'{self.version}+{other.version}', 'term': []})
        out.terms = list(self.terms)
        for t in other.terms:
            if any(t2['id'] == t['id'] or t2['en'] == t['en'] for t2 in self.terms):
                continue
            probe = ' '.join(e for e in t['en'] if not e.startswith('re:'))
            if not probe or not any(t2['en_re'].search(probe) for t2 in self.terms):
                out.terms.append(t)
        return out

    def as_dict(self):
        return {'version': self.version, 'terms': [{'id': t['id'], 'en': t['en'], 'ar': t['ar'],
                                                     'require': t['require'], 'avoid': t['avoid']} for t in self.terms]}

    def in_source(self, source):
        return [t for t in self.terms if t['en_re'].search(source or '')]

    def missing(self, source, arabic):
        """Terms of the source whose approved rendering is not in the Arabic."""
        ar = normalize_ar(arabic)
        return [t for t in self.in_source(source)
                if not all(any(stem in ar for stem in alt) for alt in t['require'])]

    def hint_lines(self, source):
        """'english -> arabic' lines for the terms that occur in this source (for prompts)."""
        out = []
        for t in self.in_source(source):
            line = f"{t['en'][0] if not t['en'][0].startswith('re:') else t['id'].replace('_', ' ')} -> {t['ar']}"
            if t['avoid']:
                line += ' (not: ' + '، '.join(t['avoid']) + ')'
            out.append(line)
        return out

    def hymt_pairs(self, source):
        """(english, arabic) pairs for HY-MT's terminology-intervention template; plain phrases only."""
        pairs = []
        for t in self.in_source(source):
            m = t['en_re'].search(source)
            if m:
                pairs.append((m.group(0), re.sub(r'\s*\(.*?\)', '', t['ar']).strip()))
        return pairs


class Policy:
    def __init__(self, data):
        self.version = data.get('version', 'policy')
        self.rules = list(data.get('rules', []))
        self.dialect = {normalize_ar(w) for w in data.get('dialect_words', [])}
        self.calques = [{'id': c['id'], 're': re.compile(c['pattern']), 'hint': c.get('hint', '')}
                        for c in data.get('calque', [])]
        # subject-independent notation terms, merged into every glossary
        self.terms = Glossary({'version': self.version + '_terms', 'term': data.get('term', [])})

    @classmethod
    def load(cls, path):
        return cls(_read_toml(_resolve(path)))

    def dialect_words(self, arabic):
        found = []
        for tok in _AR_TOKEN.findall(_MARKS.sub('', arabic or '')):
            n = normalize_ar(tok)
            cands = [n] + ([n[1:]] if n[:1] in _CLITIC and len(n) > 3 else [])
            hit = next((c for c in cands if c in self.dialect), None)
            if hit:
                found.append(hit)
        return found

    def calque_ids(self, arabic):
        t = _MARKS.sub('', arabic or '')
        return [c['id'] for c in self.calques if c['re'].search(t)]

    def calque_hints(self, ids):
        by_id = {c['id']: c['hint'] for c in self.calques}
        return [by_id[i] for i in ids if i in by_id]


class Language:
    """Policy + glossary + how the gate treats each kind of finding ('reject' | 'flag' | 'off')."""

    def __init__(self, policy=None, glossary=None, *, terminology='reject', dialect='reject', calques='flag'):
        if policy is not None and policy.terms.terms:
            glossary = glossary.merged(policy.terms) if glossary is not None else policy.terms
        self.policy, self.glossary = policy, glossary
        self.modes = {'terminology': terminology, 'dialect': dialect, 'calques': calques}
        self.name = '+'.join(x.version for x in (policy, glossary) if x is not None) or 'none'

    @classmethod
    def from_config(cls, cfg, *, auto_glossary=None):
        """None when the config has no [language] section (V1 behaviour).

        glossary = "auto"          : the glossary the LLM built from this video's transcript (dynamic)
        glossary = "auto+<file>"   : the automatic one plus a reviewed file whose terms win
        glossary = "<file>"        : a fixed file only
        glossary = ""              : no terminology check"""
        lc = cfg.get('language')
        if not lc:
            return None
        policy = Policy.load(lc['policy']) if lc.get('policy') else None
        spec, glossary = lc.get('glossary') or '', None
        if spec.startswith('auto'):
            manual = spec.split('+', 1)[1] if '+' in spec else None
            glossary = auto_glossary
            if manual:
                fixed = Glossary.load(manual)
                glossary = fixed.merged(auto_glossary) if auto_glossary else fixed
        elif spec:
            glossary = Glossary.load(spec)
        return cls(policy, glossary, terminology=lc.get('terminology', 'reject'),
                   dialect=lc.get('dialect', 'reject'), calques=lc.get('calques', 'flag'))

    def check(self, source, arabic):
        terms = [t['id'] for t in self.glossary.missing(source, arabic)] if self.glossary else []
        dialect = self.policy.dialect_words(arabic) if self.policy else []
        calques = self.policy.calque_ids(arabic) if self.policy else []
        return {'terms': terms, 'dialect': dialect, 'calques': calques}

    def reasons(self, found):
        """Gate reasons for the findings whose mode is 'reject'."""
        out = []
        if self.modes['terminology'] == 'reject':
            out += ['term:' + t for t in found['terms']]
        if self.modes['dialect'] == 'reject' and found['dialect']:
            out.append('dialect:' + ','.join(sorted(set(found['dialect']))))
        if self.modes['calques'] == 'reject':
            out += ['style:' + c for c in found['calques']]
        return out

    def prompt_block(self, source):
        """Policy rules and the glossary lines relevant to this source, for LLM prompts."""
        block = {}
        if self.policy:
            block['style_rules'] = self.policy.rules
        if self.glossary:
            lines = self.glossary.hint_lines(source)
            if lines:
                block['glossary'] = lines
        return block

    def issue_hints(self, source, found):
        """Readable problems for the refine / polish prompts."""
        hints = []
        if self.glossary and found['terms']:
            by_id = {t['id']: t for t in self.glossary.terms}
            hints += [f"use the approved term: {by_id[i]['ar']}" for i in found['terms'] if i in by_id]
        if found['dialect']:
            hints.append('replace dialect words with Modern Standard Arabic: ' + '، '.join(sorted(set(found['dialect']))))
        if self.policy and found['calques']:
            hints += self.policy.calque_hints(found['calques'])
        return hints


def build_glossary(tasks, client, cache_dir, *, max_terms=40, log=print):
    """Terminology agent: the LLM reads the whole transcript once, before translation, and proposes
    the glossary of this video. Write-once cache keyed by transcript + prompt + model, so a rerun reuses
    the same decisions and the file can be reviewed or edited by a person."""
    import json
    from .common import now_iso, read_json, sha256_json, write_json_once
    from .prompts import GLOSSARY_SYSTEM, GLOSSARY_VERSION
    transcript = [{'unit': t['unit_id'], 'text': t['source_text']} for t in tasks]
    key = sha256_json([transcript, GLOSSARY_SYSTEM, client.provider, client.model])[:12]
    path = Path(cache_dir) / 'glossary' / f'{GLOSSARY_VERSION}__{key}.json'
    if path.exists():
        rec = read_json(path)
    else:
        reply = client.chat_json(GLOSSARY_SYSTEM, json.dumps({'transcript': transcript}, ensure_ascii=False)) or {}
        terms = [t for t in reply.get('terms') or [] if isinstance(t, dict)][:max_terms]
        rec = {'version': f'{GLOSSARY_VERSION}__{key}', 'builder': f'{client.provider}:{client.model}',
               'subject': reply.get('subject'), 'created_at': now_iso(), 'terms': terms, 'human_review': 'none'}
        write_json_once(path, rec)
        log(f'[glossary] {len(terms)} terms for "{rec["subject"]}" by {rec["builder"]} -> {path}')
    return Glossary.from_terms(rec['terms'], rec['version']), path
