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


TERM_STATUSES = ('proposed', 'auto_verified', 'approved', 'conflicting', 'rejected')


def term_key(en_form):
    return re.sub(r'[^a-z0-9]+', '_', (en_form or '').lower()).strip('_')


class Glossary:
    """Terms with a status: 'approved' (a person or a reviewed file), 'auto_verified' (proposed
    independently with the same rendering by two or more videos), 'proposed' (one agent run),
    'conflicting' (different renderings proposed). Only the statuses in Language.reject_statuses can
    reject a candidate; the others are suggestions for the prompts and findings in the report."""

    def __init__(self, data, *, default_status='approved'):
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
                'status': t.get('status', default_status),
            })

    @classmethod
    def load(cls, path):
        return cls(_read_toml(_resolve(path)))

    @classmethod
    def from_terms(cls, terms, version, status='proposed'):
        """Glossary from a list of term dicts (LLM-built, status 'proposed' unless a term carries one);
        ids are derived from the English form."""
        data, seen = {'version': version, 'term': []}, set()
        for t in terms:
            en = [e for e in (t.get('en') or []) if isinstance(e, str) and e.strip()]
            ar = t.get('ar')
            if not en or not isinstance(ar, str) or not ar.strip():
                continue
            tid = term_key(en[0])
            if not tid or tid in seen:
                continue
            seen.add(tid)
            data['term'].append({'id': tid, 'en': en, 'ar': ar.strip(),
                                 'ar_forms': [f for f in t.get('ar_forms') or [] if isinstance(f, str)],
                                 'avoid': [a for a in t.get('avoid') or [] if isinstance(a, str)],
                                 'status': t.get('status', status)})
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
                                                     'require': t['require'], 'avoid': t['avoid'],
                                                     'status': t['status']} for t in self.terms]}

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
        simp = data.get('simplicity', {})
        self.max_sentence_words = int(simp.get('max_sentence_words', 0) or 0)
        self.rare_words = {normalize_ar(k): v for k, v in (simp.get('rare_words') or {}).items()}
        addr = data.get('address', {})
        self.address = addr.get('form')
        self.address_other = {normalize_ar(w) for w in addr.get('other_forms', [])}
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

    def simplicity_ids(self, arabic):
        """Readability findings: 'long_sentence:<words>', 'rare:<word>', 'address:<form>'."""
        out = []
        t = _MARKS.sub('', arabic or '')
        if self.max_sentence_words:
            for sent in re.split(r'[.!?؟؛]+', t):
                n = len(sent.split())
                if n > self.max_sentence_words:
                    out.append(f'long_sentence:{n}')
        norm = ' ' + ' '.join(normalize_ar(w) for w in re.findall(r'\S+', t)) + ' '
        for word in self.rare_words:
            if re.search(r'(?:^|\s)[وفبل]?' + re.escape(word) + r'(?=[\s،.؟!]|$)', norm):
                out.append('rare:' + word)
        if self.address_other:
            for tok in _AR_TOKEN.findall(t):
                n = normalize_ar(tok)
                if n in self.address_other or (n[:1] in _CLITIC and n[1:] in self.address_other):
                    out.append('address:' + tok)
                    break
        return out

    def simplicity_hints(self, ids):
        hints = []
        for i in ids:
            kind, _, val = i.partition(':')
            if kind == 'long_sentence':
                hints.append(f'a sentence has {val} words: split it into short sentences of one idea each')
            elif kind == 'rare':
                hints.append(f'replace the rare word «{val}» with the common word «{self.rare_words.get(val, "")}»')
            elif kind == 'address':
                hints.append(f'address the viewer in the {self.address} (أنت), not «{val}»')
        return hints

    def calque_hints(self, ids):
        by_id = {c['id']: c['hint'] for c in self.calques}
        return [by_id[i] for i in ids if i in by_id]


class Language:
    """Policy + glossary + how the gate treats each kind of finding ('reject' | 'flag' | 'off')."""

    def __init__(self, policy=None, glossary=None, *, terminology='reject', dialect='reject', calques='flag',
                 simplicity='flag', reject_statuses=('approved', 'auto_verified')):
        if policy is not None and policy.terms.terms:
            glossary = glossary.merged(policy.terms) if glossary is not None else policy.terms
        self.policy, self.glossary = policy, glossary
        self.subject = None          # subject of the video, found by the terminology agent
        self.modes = {'terminology': terminology, 'dialect': dialect, 'calques': calques, 'simplicity': simplicity}
        self.reject_statuses = tuple(reject_statuses)
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
                   dialect=lc.get('dialect', 'reject'), calques=lc.get('calques', 'flag'),
                   simplicity=lc.get('simplicity', 'flag'),
                   reject_statuses=tuple(lc.get('reject_statuses', ('approved', 'auto_verified'))))

    def check(self, source, arabic):
        missing = self.glossary.missing(source, arabic) if self.glossary else []
        terms = [t['id'] for t in missing]
        enforced = [t['id'] for t in missing if t['status'] in self.reject_statuses]
        dialect = self.policy.dialect_words(arabic) if self.policy else []
        calques = self.policy.calque_ids(arabic) if self.policy else []
        simple = self.policy.simplicity_ids(arabic) if self.policy else []
        return {'terms': terms, 'terms_enforced': enforced, 'dialect': dialect, 'calques': calques,
                'simplicity': simple}

    def reasons(self, found):
        """Gate reasons for the findings whose mode is 'reject'."""
        out = []
        if self.modes['terminology'] == 'reject':                  # only verified / approved terms reject
            out += ['term:' + t for t in found.get('terms_enforced', found['terms'])]
        if self.modes['dialect'] == 'reject' and found['dialect']:
            out.append('dialect:' + ','.join(sorted(set(found['dialect']))))
        if self.modes['calques'] == 'reject':
            out += ['style:' + c for c in found['calques']]
        if self.modes['simplicity'] == 'reject':
            out += ['simple:' + c for c in found.get('simplicity', [])]
        return out

    def prompt_block(self, source, task=None):
        """Policy rules, the video subject, the speaker's gender (when known) and the glossary lines
        relevant to this source, for LLM prompts."""
        block = {}
        if self.subject:
            block['video_subject'] = self.subject
        gender = (task or {}).get('speaker_gender')
        if gender in ('male', 'female'):
            block['speaker_gender'] = gender            # first-person agreement (أنا متأكد / متأكدة)
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
        if self.policy and found.get('simplicity'):
            hints += self.policy.simplicity_hints(found['simplicity'])
        return hints

    @staticmethod
    def needs_polish(found):
        return bool(found.get('calques') or found.get('simplicity'))


class _FileLock:
    """Exclusive lock file next to the memory (O_CREAT | O_EXCL). A lock older than `stale_s` is taken
    over (a crashed run). Reliable on one machine; on Google Drive, two Colab sessions may not see each
    other's lock in time (Drive sync), so concurrent runs of the SAME course on two machines are not
    supported; writes re-read the file under the lock and merge, so one machine's runs never lose data."""

    def __init__(self, path, timeout_s=60.0, stale_s=600.0):
        self.path, self.timeout_s, self.stale_s = Path(path), timeout_s, stale_s

    def __enter__(self):
        import os
        import time
        deadline = time.time() + self.timeout_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale_s:
                        self.path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.time() > deadline:
                    raise TimeoutError(f'course memory is locked: {self.path}')
                time.sleep(0.2)

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class CourseMemory:
    """Terminology memory of a whole course (one JSON shared by its videos). Fully automatic; a person
    may review it, nobody has to.

    Each term (key = English form) keeps every rendering proposed for it, with its sources: video,
    units, context sentence, glossary version, model, time. Status per term:
      proposed      one video proposed it                         -> suggestion only
      auto_verified two or more videos proposed the SAME rendering independently (the terminology agent is
                    shown only verified terms, so a repeated proposal is an independent one)  -> enforced
      conflicting   different renderings were proposed             -> suggestion only, listed in 'conflicts'
      approved / rejected  set by a person (set_status); never set automatically   -> approved is enforced
    A video is recorded once (a rerun replaces its own contribution). The memory is written only by
    record_video / set_status, under a lock, by re-reading the file and merging (no lost update), with an
    atomic replace and a copy of every revision in history/."""

    SCHEMA = 'course_memory/v2'

    def __init__(self, path):
        self.path = Path(path)
        self.data = self._read()

    # ------------------------------------------------------------ io
    def _empty(self):
        return {'schema': self.SCHEMA, 'revision': 0, 'terms': {}, 'videos': {}, 'conflicts': []}

    def _read(self):
        import json
        if not self.path.exists():
            return self._empty()
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if data.get('schema') != self.SCHEMA:                      # v1 list format -> v2, all 'proposed'
            old, data = data, self._empty()
            for t in old.get('terms', []):
                en = [e for e in t.get('en', []) if isinstance(e, str)]
                if en and t.get('ar'):
                    self._add_rendering(data, en, t, t.get('first_video', 'v1-import'), {}, None)
            for k in data['terms']:
                self._restatus(data, k)
        return data

    def _write(self, data):
        import json
        import os
        data['revision'] = int(data.get('revision', 0)) + 1
        text = json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True)
        hist = self.path.parent / 'history' / f'{self.path.stem}__r{data["revision"]:04d}.json'
        hist.parent.mkdir(parents=True, exist_ok=True)
        hist.write_text(text, encoding='utf-8')
        tmp = self.path.with_name(f'{self.path.name}.{os.getpid()}.tmp')
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, self.path)
        self.data = data

    def _locked_update(self, fn):
        with _FileLock(self.path.with_suffix('.lock')):
            data = self._read()                                     # latest version, not the one loaded earlier
            result = fn(data)
            self._write(data)
        return result

    # ------------------------------------------------------------ reading
    @property
    def terms(self):
        """Verified terms (approved / auto_verified) with their chosen rendering: the prior of a new video."""
        return self.snapshot()['verified']

    def snapshot(self):
        """What a run uses: the verified terms, the memory revision and its content hash."""
        import hashlib
        import json
        verified = []
        for key in sorted(self.data['terms']):
            t = self.data['terms'][key]
            if t['status'] in ('approved', 'auto_verified') and t.get('chosen'):
                r = t['renderings'][t['chosen']]
                verified.append({'en': t['en'], 'ar': t['chosen'], 'ar_forms': r.get('ar_forms', []),
                                 'avoid': r.get('avoid', []), 'status': t['status']})
        body = json.dumps(self.data, ensure_ascii=False, sort_keys=True).encode('utf-8')
        return {'schema': self.SCHEMA, 'memory_path': str(self.path), 'revision': self.data.get('revision', 0),
                'memory_sha256': hashlib.sha256(body).hexdigest(), 'verified': verified}

    def glossary(self):
        v = self.terms
        return Glossary.from_terms(v, 'course_memory') if v else None

    # ------------------------------------------------------------ writing
    @staticmethod
    def _add_rendering(data, en, t, video, meta, context):
        from .common import now_iso
        key = term_key(en[0])
        term = data['terms'].setdefault(key, {'en': list(en), 'status': 'proposed', 'renderings': {},
                                              'chosen': None, 'history': []})
        for e in en:
            if e not in term['en']:
                term['en'].append(e)
        ar = t['ar'].strip()
        r = term['renderings'].setdefault(ar, {'ar_forms': [], 'avoid': [], 'videos': [], 'sources': []})
        r['ar_forms'] = sorted(set(r['ar_forms']) | {f for f in t.get('ar_forms') or [] if isinstance(f, str)})
        r['avoid'] = sorted(set(r['avoid']) | {a for a in t.get('avoid') or [] if isinstance(a, str)})
        if video not in r['videos']:
            r['videos'].append(video)
        r['sources'].append({'video': video, 'units': (context or {}).get('units', []),
                             'context': (context or {}).get('sentence', ''),
                             'glossary_version': meta.get('glossary_version'), 'builder': meta.get('builder'),
                             'run_id': meta.get('run_id'), 'at': now_iso()})
        return key

    @staticmethod
    def _restatus(data, key):
        from .common import now_iso
        term = data['terms'][key]
        live = {ar: r for ar, r in term['renderings'].items() if r['videos']}
        old = term['status']
        if old in ('approved', 'rejected'):                          # a person's decision is never overwritten
            others = [ar for ar in live if ar != term.get('chosen')]
            if old == 'approved' and others:
                CourseMemory._conflict(data, key, live)
            return
        if len(live) > 1:
            term['status'] = 'conflicting'
            term['chosen'] = max(live, key=lambda ar: (len(live[ar]['videos']), -list(term['renderings']).index(ar)))
            CourseMemory._conflict(data, key, live)
        elif len(live) == 1:
            ar = next(iter(live))
            term['chosen'] = ar
            term['status'] = 'auto_verified' if len(live[ar]['videos']) >= 2 else 'proposed'
        else:
            term['status'], term['chosen'] = 'proposed', None
        if term['status'] != old:
            term['history'].append({'at': now_iso(), 'event': f'{old} -> {term["status"]}', 'by': 'automatic'})

    @staticmethod
    def _conflict(data, key, live):
        entry = {'term': key, 'renderings': {ar: sorted(r['videos']) for ar, r in live.items()}}
        data['conflicts'] = [c for c in data.get('conflicts', []) if c['term'] != key] + [entry]

    def record_video(self, video_id, terms, *, contexts=None, meta=None):
        """Record the terms one video's agent proposed. Idempotent per video: a rerun first removes that
        video's earlier contribution. Returns {'added', 'conflicts', 'verified'}."""
        meta, contexts = dict(meta or {}), contexts or {}

        def apply(data):
            for term in data['terms'].values():                      # drop this video's earlier contribution
                for r in term['renderings'].values():
                    if video_id in r['videos']:
                        r['videos'].remove(video_id)
                        r['sources'] = [s for s in r['sources'] if s['video'] != video_id]
            keys = set()
            for t in terms:
                en = [e for e in (t.get('en') or []) if isinstance(e, str) and e.strip()]
                if not en or not isinstance(t.get('ar'), str) or not t['ar'].strip():
                    continue
                keys.add(self._add_rendering(data, en, t, video_id, meta, contexts.get(term_key(en[0]))))
            for key in list(data['terms']):
                self._restatus(data, key)
            data['videos'][video_id] = {'subject': meta.get('subject'), 'run_id': meta.get('run_id'),
                                        'glossary_version': meta.get('glossary_version'), 'terms': sorted(keys)}
            return {'terms': len(keys),
                    'verified': sum(t['status'] == 'auto_verified' for t in data['terms'].values()),
                    'conflicts': len(data['conflicts'])}

        return self._locked_update(apply)

    def set_status(self, key, status, *, rendering=None, by='reviewer', note=''):
        """A person's decision (optional review): approve or reject a term, choosing a rendering."""
        from .common import now_iso
        if status not in TERM_STATUSES:
            raise ValueError(status)

        def apply(data):
            term = data['terms'][key]
            if rendering is not None:
                term['renderings'].setdefault(rendering, {'ar_forms': [], 'avoid': [], 'videos': [], 'sources': []})
                term['chosen'] = rendering
            term['history'].append({'at': now_iso(), 'event': f'{term["status"]} -> {status}', 'by': by,
                                    'note': note})
            term['status'] = status

        self._locked_update(apply)


def term_contexts(terms, tasks):
    """{term_key: {'units': [...], 'sentence': first source sentence that contains the term}}."""
    g = Glossary.from_terms(terms, 'contexts')
    out = {}
    for t in g.terms:
        units = [task['unit_id'] for task in tasks if t['en_re'].search(task['source_text'])]
        sentence = ''
        for task in tasks:
            m = t['en_re'].search(task['source_text'])
            if m:
                parts = re.split(r'(?<=[.?!])\s+', task['source_text'])
                sentence = next((p for p in parts if t['en_re'].search(p)), task['source_text'])[:240]
                break
        out[t['id']] = {'units': units, 'sentence': sentence}
    return out


def build_glossary(tasks, client, cache_dir, *, max_terms=40, prior_terms=(), log=print):
    """Terminology agent: the LLM reads the whole transcript once, before translation, and proposes
    the glossary of this video. `prior_terms` are the course's approved terms from earlier videos: the
    agent must reuse them and only add new ones, and they win over its proposals.
    Write-once cache keyed by transcript + prior terms + prompt + model, so a rerun reuses the same
    decisions and the file can be reviewed or edited by a person.
    Returns (glossary, cache path, subject)."""
    import json
    from .common import now_iso, read_json, sha256_json, write_json_once
    from .prompts import GLOSSARY_SYSTEM, GLOSSARY_VERSION
    transcript = [{'unit': t['unit_id'], 'text': t['source_text']} for t in tasks]
    prior = [{k: t[k] for k in ('en', 'ar') if k in t} for t in prior_terms]
    key = sha256_json([transcript, prior, GLOSSARY_SYSTEM, client.provider, client.model])[:12]
    path = Path(cache_dir) / 'glossary' / f'{GLOSSARY_VERSION}__{key}.json'
    if path.exists():
        rec = read_json(path)
    else:
        payload = {'transcript': transcript}
        if prior:
            payload['approved_terms_from_earlier_videos'] = prior
        reply = client.chat_json(GLOSSARY_SYSTEM, json.dumps(payload, ensure_ascii=False)) or {}
        terms = [t for t in reply.get('terms') or [] if isinstance(t, dict)][:max_terms]
        rec = {'version': f'{GLOSSARY_VERSION}__{key}', 'builder': f'{client.provider}:{client.model}',
               'subject': reply.get('subject'), 'created_at': now_iso(), 'prior_terms': len(prior),
               'terms': terms, 'human_review': 'none'}
        write_json_once(path, rec)
        log(f'[glossary] {len(terms)} terms for "{rec["subject"]}" by {rec["builder"]} '
            f'({len(prior)} course terms given) -> {path}')
    glossary = Glossary.from_terms(rec['terms'], rec['version'])           # this video's proposals
    if prior_terms:                                                        # verified course terms win
        glossary = Glossary.from_terms(list(prior_terms), 'course_memory').merged(glossary)
    return glossary, path, rec.get('subject')
