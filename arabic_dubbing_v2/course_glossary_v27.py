"""Figma 02.27: conservative course identity, terminology review and consistency.

This is an OFFLINE report generator. It never calls an LLM or TTS, changes
original ASR/timings, or writes approval into final_dub_ar. Inferred topic
and candidate terms must not be represented as human-validated knowledge.
"""
from datetime import datetime
import re
import unicodedata
from collections import defaultdict

from .common import clone, fingerprint, unit_text

SCHEMA_VERSION = 'course-glossary-v2.7'
_PLACEHOLDER_IDS = {'', 'pilot_course', 'course_pilot', 'example', 'test', 'unknown'}
_AR_DIACRITICS = re.compile(r'[\u064b-\u065f\u0670\u0640]')


def _occurs(text, term):
    if not isinstance(term, str) or not term.strip():
        return False
    # Prevent 'net' matching 'internet'; punctuation in C++, C#, etc. is literal.
    return bool(re.search(r'(?<!\w)' + re.escape(term.strip()) + r'(?!\w)', str(text), re.I))


def _arabic_match(target, expression):
    if not expression or not target:
        return False
    def normalize(t):
        return _AR_DIACRITICS.sub('', unicodedata.normalize('NFKC', str(t))).casefold()
    return normalize(expression) in normalize(target)


def source_evidence(prepared, source_term):
    evidence = []
    for u in prepared['translation_units']:
        if _occurs(unit_text(u), source_term):
            evidence.append({'unit_id': u['id'],
                             'source_refs': clone(u.get('source_refs') or []),
                             'source_text': unit_text(u)})
    return evidence


def identity_audit(prepared):
    """Do not turn a source-bound fallback hash into a verified course title."""
    original = prepared.get('course') or {}
    resolved = prepared['translation_v2_metadata']['course']
    original_id = str(original.get('id') or '').strip()
    original_title = str(original.get('title') or '').strip()
    is_fallback = original_id.lower() in _PLACEHOLDER_IDS
    # Callers might supply a validated non-placeholder ID via metadata override.
    status = 'source_placeholder_not_identified' if is_fallback else 'source_course_id_provided'
    return {'resolved_id': resolved['id'], 'original_id': original_id or None,
            'provided_title': resolved.get('title'),
            'original_title': original_title or None,
            'identity_status': status,
            'scope': 'source_instance_only' if is_fallback else 'provided_course_id_unverified',
            'cross_lesson_reuse_permitted': False,
            'warning': ('A fallback ID fingerprints this source, not an identified course. '
                        'Confirm canonical course ID before sharing approvals across lessons.'
                        if is_fallback else 'Course ID must be independently confirmed before cross-lesson reuse.')}


def validate_suggestion(prepared, proposal):
    """Structural/context evidence gate; cannot assert semantic correctness."""
    course_id = prepared['translation_v2_metadata']['course']['id']
    if not isinstance(proposal, dict):
        raise TypeError('Term proposal must be a dictionary')
    term = str(proposal.get('source_term') or '').strip()
    if not term:
        raise ValueError('Missing source_term')
    scoped = str(proposal.get('course_id') or '').strip()
    if scoped != course_id:
        raise ValueError('Cross-course term proposal rejected')
    evidence = source_evidence(prepared, term)
    if not evidence:
        raise ValueError(f'Term has no literal occurrence in source TUs: {term}')
    arabic = str(proposal.get('preferred_ar') or '').strip()
    sense = str(proposal.get('sense') or '').strip()
    keep = proposal.get('keep_english')
    if keep not in (True, False, None):
        raise ValueError('keep_english must be boolean or null')
    if not sense or not arabic:
        return {'source_term': term, 'course_id': course_id, 'source_refs': evidence,
                'review_status': 'needs_review', 'validation': 'missing_meaning_or_arabic',
                'preferred_ar': arabic or None, 'sense': sense or None}
    return {'source_term': term, 'course_id': course_id,
            'term_id': 'term_' + fingerprint([course_id, term.casefold(), sense.casefold()])[:14],
            'source_refs': evidence, 'sense': sense, 'preferred_ar': arabic,
            'alternatives': list(proposal.get('alternatives') or []),
            'keep_english': keep,
            'spoken_form': proposal.get('spoken_form'),
            'review_status': 'validated_pending_human_approval',
            'validation': 'source_presence_and_required_fields_only',
            'provenance': clone(proposal.get('provenance') or {}),
            'version': proposal.get('version') or 1,
            'warning': 'Meaning and Arabic rendering NOT automatically verified.'}


def record_human_approval(prepared, proposal, *, approval):
    """Explicit human evidence required to transition to approved.

    approval must be a dictionary with approved=True, reviewer, decision_id,
    approved_at and confirmed_course_id. Does not mutate a Master or memory.
    """
    validated = validate_suggestion(prepared, proposal)
    if validated['review_status'] != 'validated_pending_human_approval':
        raise ValueError('Cannot approve incomplete contextual proposal')
    if not isinstance(approval, dict) or approval.get('approved') is not True:
        raise PermissionError('Explicit human approval required')
    required = ('reviewer', 'decision_id', 'approved_at', 'confirmed_course_id')
    if any(not str(approval.get(k) or '').strip() for k in required):
        raise PermissionError('Human approval audit evidence incomplete')
    if str(approval['confirmed_course_id']) != validated['course_id']:
        raise ValueError('Approved course ID mismatch')
    try:
        timestamp = str(approval['approved_at']).replace('Z', '+00:00')
        dt = datetime.fromisoformat(timestamp)
        if dt.tzinfo is None: raise ValueError('Timestamp missing timezone')
    except ValueError as exc:
        raise ValueError('approved_at must be an ISO-8601 timezone-aware timestamp') from exc
    validated['review_status'] = 'approved'
    validated['approval'] = {k: str(approval[k]) for k in required}
    validated['scope'] = 'confirmed_course_by_reviewer'
    return validated


def _unit_refs_for_term(prepared, term):
    return [x['unit_id'] for x in source_evidence(prepared, term)]


def consistency_audit(prepared, approved_terms):
    """Conservative suggestions, not a translation-semantic judgement.

    A source occurrence whose expected Arabic phrase is absent is flagged for
    review. Morphology/paraphrase can yield false positives, never auto-fail.
    """
    units = {u['id']: u for u in prepared['translation_units']}
    findings = []
    for term in approved_terms:
        if term.get('review_status') != 'approved':
            raise ValueError('Only approved entries may drive terminology consistency QA')
        if term.get('course_id') != prepared['translation_v2_metadata']['course']['id']:
            raise ValueError('Cross-course approved entry')
        english = term['source_term']
        variants = [term.get('preferred_ar')] + list(term.get('alternatives') or [])
        for uid in _unit_refs_for_term(prepared, english):
            arabic = str((units[uid].get('translation') or {}).get('natural_ar') or '')
            matched = bool(arabic) and any(_arabic_match(arabic, t) for t in variants if t)
            findings.append({'unit_id': uid, 'source_term': english,
                             'preferred_ar': term.get('preferred_ar'),
                             'status': ('literal_arabic_match' if matched else
                                        'potential_mismatch_needs_review'),
                             'semantic_equivalence_verified': False})
    return {'findings': findings, 'potential_mismatch_count': sum(x['status']=='potential_mismatch_needs_review' for x in findings),
            'note': 'Literal substring comparison only; inflections and paraphrases may be valid.'}


def term_change_impact(prepared, current_approved, previous_approved):
    """Suggest units for re-QA when a previously approved rendering changes."""
    for t in list(previous_approved) + list(current_approved):
        if t.get('course_id', t.get('scope_course_id')) != prepared['translation_v2_metadata']['course']['id']:
            raise ValueError('Cross-course term impact comparison rejected')
    def key(t):
        return (t.get('source_term','').casefold(), str(t.get('sense') or '').casefold())
    prev = {key(t): t for t in previous_approved if t.get('review_status') == 'approved'}
    now = {key(t): t for t in current_approved if t.get('review_status') == 'approved'}
    changes=[]
    for k, old in prev.items():
        new = now.get(k)
        if not new or (new.get('preferred_ar'), new.get('keep_english')) != (old.get('preferred_ar'), old.get('keep_english')):
            changes.append({'source_term': old['source_term'], 'sense': old.get('sense'),
                            'before': old.get('preferred_ar'), 'after': new.get('preferred_ar') if new else None,
                            'affected_unit_ids': _unit_refs_for_term(prepared, old['source_term']),
                            'action': 'requires_reqa_review'})
    return changes


def build_course_glossary_report(prepared, *, human_approved=(), previous_approved=(), suggestions=()):
    """Build a derived, read-only full-domain glossary report for any course."""
    expected = prepared['translation_v2_metadata']['course']['id']
    gloss = prepared['terminology_v2']
    if gloss['course_id'] != expected:
        raise ValueError('Inconsistent course identity between metadata and glossary')
    validated=[validate_suggestion(prepared,p) for p in suggestions]
    existing = clone(gloss.get('approved') or [])
    for old in existing:
        # Normalize explicit same-course legacy approvals only in this derived report.
        old['review_status'] = 'approved'
        old['course_id'] = str(old.get('course_id') or old.get('scope_course_id') or '')
    incoming = clone(list(human_approved))
    for fresh in incoming:
        fields = fresh.get('approval') or {}
        if any(not fields.get(k) for k in ('reviewer','decision_id','approved_at','confirmed_course_id')):
            raise PermissionError('New approvals require explicit audited human decision')
        if fields['confirmed_course_id'] != expected:
            raise ValueError('Cross-course human approval rejected')
    approved = existing + incoming
    for t in approved:
        if t.get('review_status',t.get('status')) != 'approved' or t.get('course_id',t.get('scope_course_id')) != expected:
            raise ValueError('Unapproved or cross-course memory record rejected')
        if not source_evidence(prepared, t.get('source_term','')):
            # No reusing a term from unrelated material without a source occurrence.
            raise ValueError('Approved term not present in current lesson source')
    conflict_keys=defaultdict(set)
    for t in approved:
        conflict_keys[(t['source_term'].casefold(),str(t.get('sense') or '').casefold())].add(str(t.get('preferred_ar')))
    conflicts=[{'source_term':k[0], 'sense':k[1], 'conflicting_ar':sorted(v)}
               for k,v in conflict_keys.items() if len(v)>1]
    source_terms=[]
    for t in gloss['candidates']:
        proof=source_evidence(prepared,t['source_term'])
        if not proof:
            continue
        source_terms.append({**clone(t), 'source_evidence': proof,
                             'review_status':'candidate'})
    consistency=consistency_audit(prepared, approved)
    impact=term_change_impact(prepared, approved, previous_approved)
    identity=identity_audit(prepared)
    return {'schema_version':SCHEMA_VERSION, 'source_master_data_fingerprint':prepared['translation_v2_metadata']['source_master_data_fingerprint'],
            'course_identity':identity, 'domain_detection':clone(gloss['domain_analysis']),
            'course_id':expected, 'candidate_count':len(source_terms), 'candidates':source_terms,
            'validated_proposals':validated, 'approved_memory':approved,
            'unverified_historical_terms':clone(gloss.get('unverified_existing_terms') or []),
            'conflicts':conflicts, 'consistency_qa':consistency,
            'changed_approved_term_impact':impact,
            'status':'needs_human_validation',
            'gates':{'domain_confirmed':False,'all_terms_approved':False,
                     'translation_semantic_approved':False,'final_dub_ar_approved':False},
            'network_calls':0,'tts_calls':0,
            'notes':['Model-free extraction cannot decide subject meaning by itself.',
                     'No newly approved entry is created unless record_human_approval is called with an explicit reviewer decision.',
                     'No original source, unit, word refs, timing, or translation is overwritten.']}