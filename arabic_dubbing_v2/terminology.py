"""Per-course, extractive terminology candidates. No default domain glossary.

No model calls or Arabic equivalents are fabricated. Existing glossary entries
are imported as proposals unless their same-course approval is explicit.
"""
import re
from collections import Counter, defaultdict
from .common import clone, fingerprint, unit_text

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")
_SINGLE_SYMBOL = re.compile(r"(?<![\w'’])([A-Za-z])(?![\w])")
_CONNECTOR_PREFIXES = {'for', 'to', 'of', 'in', 'on', 'by', 'with', 'without'}
_SKIP = set('the and for that with this from your what where while then than have has had they them their there here were was are you can could would should about into onto over under after before because which these those also just very will been being when how why now let lets our we it its of to in on at by is a an or as if but so no not do did does one two first second next step right use using used got get make makes made such like look see seeable some other many each any all more less most much only same new old too example examples thing things way ways today course lesson video part okay hello welcome thank thanks back basically going gonna actually really little lot name known say says said talk talking take taking give given find found different various well idea ideas stuff something someone somebody he she him her his hers their its who what'.split())
_GENERIC_TITLES = {'course pilot','pilot course','untitled','lesson','unknown','none',''}


def effective_course(course, units, overrides=None):
    course = clone(course if isinstance(course, dict) else {})
    overrides = overrides or {}
    for field in ('id','title','description','chapter','lesson','topic'):
        if overrides.get(field) not in (None, ''):
            course[field] = overrides[field]
    old_id = str(course.get('id') or '').strip()
    title = str(course.get('title') or '').strip()
    if title.lower() in _GENERIC_TITLES:
        title = ''
    if old_id.lower() in ('pilot_course', 'course_pilot', 'example', 'test', 'unknown', ''):
        seed = '\n'.join(unit_text(u)[:200] for u in units[:12]) + ':' + title
        old_id = 'course_' + fingerprint(seed)[:12]
    course['id'] = old_id
    course['title'] = title or None
    course['metadata_status'] = 'provided' if title else 'incomplete_needs_review'
    return course


def detect_domain(course, units, candidates=None):
    """Metadata-grounded domain hint; *never* pretend generic noun clusters are factual taxonomy."""
    labels = [str(course.get(k) or '').strip() for k in ('title','chapter','lesson','topic')]
    labels = [s for s in labels if s and s.lower() not in _GENERIC_TITLES]
    topic = str(course.get('topic') or '').strip()
    title = str(course.get('title') or '').strip()
    if topic and topic.lower() not in _GENERIC_TITLES:
        return {'domain_hint': topic, 'evidence': 'course.topic', 'status': 'metadata_derived_unvalidated'}
    if title and title.lower() not in _GENERIC_TITLES:
        return {'domain_hint': title, 'evidence': 'course.title', 'status': 'metadata_derived_unvalidated'}
    # Extractive fallback: a prominent repeated multiword phrase suggests a topic,
    # NOT a validated academic domain. No preloaded labels or domain vocabulary.
    strong = [x for x in (candidates or []) if len(x['source_term'].split()) >= 2
              and x['occurrences_in_units'] >= 2]
    if strong:
        term = sorted(strong, key=lambda x:(-len(x['source_term'].split()),-x['occurrences_in_units'],x['source_term']))[0]
        return {'domain_hint':term['source_term'], 'evidence':'repeated_source_phrase',
                'status':'text_derived_needs_review', 'source_refs':term['source_refs'],
                'note':'Topical phrase detected automatically; do not treat as validated domain.'}
    return {'domain_hint': None, 'evidence': 'insufficient_metadata_and_source_evidence',
            'status': 'needs_domain_validation',
            'note': 'No defensible subject inference; leave unknown instead of inventing.'}


def _tokens(text):
    # End-of-sentence punctuation is not part of an educational term.
    # Preserve internal dots in code identifiers such as numpy.mean.
    return [m.group(0).rstrip(".") for m in _WORD.finditer(text)]


def _eligible(word):
    return (len(word) >= 3 and word.lower() not in _SKIP and not word.isdigit()
            and not re.fullmatch(r'\d+[.]\d+', word))


def extract_candidates(units, course, *, max_terms=70):
    """Extract repeated terms and metadata phrases without domain dictionary leakage.

    A candidate is not an approved translated glossary entry. Support phrase
    frequency, technical identifiers and metadata terms; store source refs.
    """
    counts, refs, examples = Counter(), defaultdict(list), {}
    symbol_counts, symbol_refs, symbol_examples = Counter(), defaultdict(list), {}
    for u in units:
        source = unit_text(u)
        toks = _tokens(source)
        uid = str(u.get('id'))
        seen = set()
        seen_symbols = set()
        for match in _SINGLE_SYMBOL.finditer(source):
            symbol = match.group(1)
            # Articles and first-person pronouns are not notation evidence.
            if symbol.casefold() in ('a','i') or symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            symbol_counts[symbol] += 1
            if len(symbol_refs[symbol]) < 12: symbol_refs[symbol].append(uid)
            symbol_examples.setdefault(symbol, source[:360])
        for n in (1, 2, 3):
            for i in range(len(toks) - n + 1):
                part = toks[i:i+n]
                if not all(_eligible(x) for x in part):
                    # Phrases can begin with a generic connector such as 'for'.
                    # Keep candidates only if the source phrase repeats across units.
                    if not (n == 2 and part[0].casefold() in _CONNECTOR_PREFIXES
                            and _eligible(part[1])):
                        continue
                if n == 1 and not (re.search(r'[A-Z][a-z]+[A-Z]|[a-z]+[A-Z]|[_+#]|[A-Z]{2,}', part[0])
                                   or len(part[0]) >= 5):
                    continue
                term = ' '.join(part).lower()
                if term in seen: continue
                seen.add(term)
                counts[term] += 1
                if len(refs[term]) < 12: refs[term].append(uid)
                examples.setdefault(term, source[:360])
    metadata_text = ' '.join(str(course.get(x) or '') for x in ('title','description','chapter','lesson','topic'))
    metadata_tokens = {w.lower() for w in _tokens(metadata_text) if _eligible(w)}
    ranked = []
    for term, n in counts.items():
        words = term.split()
        matches_metadata = any(w in metadata_tokens for w in words)
        tech = bool(re.search(r'[+#_.]|\d', term))
        # Singletons may be generic words; do not flood the glossary.
        variant_of_repeated_connector = (
            len(words)==2 and words[0] in _CONNECTOR_PREFIXES and
            (counts.get(term+'s',0)>0 or
             (term.endswith('s') and counts.get(term[:-1],0)>0)))
        if n < 2 and not (matches_metadata and len(words) > 1) and not tech and not variant_of_repeated_connector:
            continue
        score = n*2 + len(words)*0.4 + (2 if matches_metadata else 0) + (2 if tech else 0)
        ranked.append((score, term, n, matches_metadata))
    ranked.sort(key=lambda x:(-x[0],x[1]))
    entries=[]
    # Avoid candidate flooding: ordinary multiword phrases have priority.
    for score, term, n, mm in ranked[:max_terms - min(4,len(symbol_counts))]:
        entries.append({'source_term': term, 'domain': None, 'sense': None,
                        'preferred_ar': None, 'alternatives': [], 'keep_english': None,
                        'source_refs': refs[term], 'context_examples': [examples[term]],
                        'occurrences_in_units': n, 'review_status': 'candidate',
                        'spoken_form': None, 'provenance': {'method':'extractive_ngrams_v1',
                                                          'metadata_overlap': mm,
                                                          'score_for_ranking_only': round(score,2)}})
    # Single-letter symbols matter for instructional speech (e.g. variable M/y),
    # but may also be nontechnical. Preserve case; do not infer pronunciation.
    for symbol, n in sorted(symbol_counts.items(), key=lambda x:(-x[1],x[0])):
        if len(entries) >= max_terms: break
        entries.append({'source_term':symbol,'domain':None,'sense':None,
                        'preferred_ar':None,'alternatives':[],'keep_english':None,
                        'source_refs':symbol_refs[symbol],
                        'context_examples':[symbol_examples[symbol]],
                        'occurrences_in_units':n,'review_status':'candidate',
                        'spoken_form':None,
                        'provenance':{'method':'single_letter_symbol_candidate_v2',
                                      'case_sensitive':True,
                                      'note':'May be notation or a nontechnical letter; context review required.'}})
    return entries


def import_approved_terms(raw_terminology, course_id):
    approved, historical = [], []
    for t in raw_terminology or []:
        if not isinstance(t, dict): continue
        record = clone(t)
        # Existing historical terms without explicit course id can be displayed,
        # but MUST NOT be imported into an approved glossary automatically.
        scoped = str(record.get('course_id') or record.get('scope_course_id') or '') == course_id
        approved_status = str(record.get('review_status') or record.get('status') or '').lower() == 'approved'
        if scoped and approved_status and record.get('preferred_ar'):
            approved.append(record)
        else:
            record['import_notice'] = 'not_approved_for_this_course'
            historical.append(record)
    return approved, historical


def glossary_for_course(course, units, historical_terms=None):
    extracted = extract_candidates(units,course)
    analysis = detect_domain(course,units,extracted)
    approved,historical = import_approved_terms(historical_terms,course['id'])
    for e in extracted: e['domain'] = analysis['domain_hint']
    return {'course_id':course['id'], 'domain_analysis': analysis,
            'approved': approved, 'candidates': extracted,
            'unverified_existing_terms': historical,
            'review_required': True,
            'policy': 'No preloaded course/domain-specific glossary; validate sense within this course.'}


def validation_plans(glossary, course):
    """Dynamic extraction -> contextual validation input, per course and sense.

    Plans can be processed by an explicitly chosen local/remote model later.
    Until then Arabic equivalents are null; every term remains a candidate.
    """
    return {'course_id':course['id'], 'status':'plans_only_NO_LLM_CALLS',
            'plans':[{'source_term':x['source_term'],
                      'scope_course_id':course['id'],
                      'domain_hint':glossary['domain_analysis']['domain_hint'],
                      'source_refs':x['source_refs'],
                      'context_examples':x['context_examples'],
                      'instruction':('Identify this educational term’s contextual meaning and propose a precise spoken-friendly '
                                     'Arabic equivalent only if supported by the source. Keep code and variables intact. '
                                     'Return JSON: {"sense":string|null,"preferred_ar":string|null,'
                                     '"alternatives":string[],"keep_english":boolean|null,'
                                     '"spoken_form":string|null,"needs_review":boolean,"rationale":string}. '
                                     'Do not invent domain facts; do not mark approved.')}
                     for x in glossary.get('candidates',[])]}


def propose_glossary_entries(glossary, course, backend, *, allow_generation=False):
    """Backend opt-in. Returns proposals; never writes approved terminology."""
    if not allow_generation:
        raise PermissionError('Terminology model calls disabled. Explicit consent required.')
    proposals=[]
    for p in validation_plans(glossary,course)['plans']:
        data=backend(p['instruction'],p,'term_context_v2')
        if not isinstance(data,dict):raise TypeError('Term backend must return a dict')
        proposals.append({**{k:p[k] for k in ('source_term','source_refs','scope_course_id')},
                          'sense':data.get('sense'),'preferred_ar':data.get('preferred_ar'),
                          'alternatives':data.get('alternatives') or [],
                          'spoken_form':data.get('spoken_form'),
                          'keep_english':data.get('keep_english'),
                          'review_status':'needs_review','model_provenance':getattr(backend,'provider_name','user_injected_backend'),
                          'context_validation_rationale':data.get('rationale')})
    return {'course_id':course['id'],'status':'candidate_not_approved','proposals':proposals}