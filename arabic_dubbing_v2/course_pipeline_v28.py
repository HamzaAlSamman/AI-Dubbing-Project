"""Figma 02.27: evidence-first multi-lesson course registry and terminology review.

Pure offline, no LLM or TTS calls. The source Master, its IDs, words and
alignments are never mutated. A canonical course ID/title is user-supplied,
never inferred from speech; without that identity, sharing is prohibited.
"""
from collections import defaultdict
from datetime import datetime
from .common import clone, fingerprint, unit_text
from .course_glossary_v27 import _occurs, source_evidence, _arabic_match
from .terminology import extract_candidates, detect_domain

SCHEMA_VERSION = 'course-registry-v2.8'
_BLOCKED = {'', 'course pilot', 'pilot course', 'unknown', 'untitled', 'test', 'example', 'pilot_course', 'course_pilot'}


def _clean_name(value):
    return str(value or '').strip()


def _id(value):
    s = _clean_name(value)
    if not s or s.casefold() in _BLOCKED:
        raise ValueError('A real non-placeholder ID is required')
    if len(s)>128 or not all(x.isascii() and (x.isalnum() or x in '_-:.') for x in s):
        raise ValueError('Identifiers must be ASCII letters/numbers/_-:. and <=128 characters')
    return s


def _prepared_info(prepared):
    if not isinstance(prepared,dict):raise TypeError('Expected a prepared Master dictionary')
    meta=prepared.get('translation_v2_metadata') or {}
    units=prepared.get('translation_units') or []
    if not units or not meta.get('source_master_data_fingerprint'):
        raise ValueError('Prepared Master lacks source fingerprint or TUs')
    ids=[_clean_name(u.get('id')) for u in units]
    if any(not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('Missing or duplicate TU IDs within a lesson')
    return meta, units


def build_course_registry(lessons, *, canonical_course=None):
    """Read-only registry. `lessons`: [(lesson_id, prepared_master), ...].

    Canonical identity is explicitly supplied by user and audited as such.
    If absent, source-scoped registry does NOT permit cross-lesson approvals.
    IDs are namespaced to avoid tu_0001 collisions across lessons.
    """
    if not isinstance(lessons, (list, tuple)) or not lessons:
        raise ValueError('Provide at least one lesson')
    if canonical_course is not None:
        if not isinstance(canonical_course,dict):raise TypeError('canonical_course must be a dict')
        course_id=_id(canonical_course.get('id'))
        title=_clean_name(canonical_course.get('title'))
        if not title or title.casefold() in _BLOCKED:
            raise ValueError('Canonical course title must be explicitly supplied, not a placeholder')
        if not _clean_name(canonical_course.get('confirmed_by')):
            raise PermissionError('Explicit confirmed_by is required to share glossary across lessons')
        identity={'id':course_id,'title':title,'confirmed_by':_clean_name(canonical_course['confirmed_by']),
                  'status':'explicitly_confirmed_by_user', 'cross_lesson_reuse_permitted':True}
    else:
        identity={'id':None,'title':None,'status':'source_only_unconfirmed',
                  'cross_lesson_reuse_permitted':False}
    seen=set(); details=[]; sources=[]
    for item in lessons:
        if not isinstance(item,(tuple,list)) or len(item)!=2:raise ValueError('Expected (lesson_id, prepared_master) pairs')
        lesson_id=_id(item[0])
        if lesson_id in seen:raise ValueError('Duplicate lesson IDs')
        seen.add(lesson_id)
        prepared=item[1]
        meta, units=_prepared_info(prepared)
        fingerprint_value=meta['source_master_data_fingerprint']
        if not isinstance(fingerprint_value,str) or len(fingerprint_value)<16:raise ValueError('Invalid source fingerprint')
        # A canonical identity is not retroactively written into source or derived Master.
        # The registry alone declares the reviewed relationship between lessons.
        reported_course = meta.get('course') or {}
        reported_id=_clean_name(reported_course.get('id'))
        if (identity['cross_lesson_reuse_permitted'] and reported_id and
                reported_id.casefold() not in _BLOCKED and reported_id!=identity['id']):
            raise ValueError('Source declares a different non-placeholder course ID; stop before merging')
        details.append({'lesson_id':lesson_id,'source_fingerprint':fingerprint_value,
                        'source_master_sha256':meta.get('source_master_sha256'),
                        'reported_course_id':reported_course.get('id'),
                        'reported_title':reported_course.get('title'),
                        'unit_count':len(units),
                        'scoped_unit_ids':[f'{lesson_id}/{u["id"]}' for u in units]})
        sources.append((lesson_id,fingerprint_value))
    if not identity['cross_lesson_reuse_permitted'] and len(lessons)>1:
        # Multi-lesson analysis is possible without sharing an approved glossary.
        identity['warning']='Lesson TUs may be compared, but their terminology decisions remain isolated.'
    snapshot={'schema_version':SCHEMA_VERSION,'course_identity':identity,'lessons':details,
              'lesson_count':len(details),'unit_count':sum(x['unit_count'] for x in details),
              'source_collection_fingerprint':fingerprint(sources),
              'originals_mutated':False,'network_calls':0,'tts_calls':0}
    return snapshot


def terminology_review_queue(registry, lessons, *, max_terms_per_lesson=70):
    """Source-grounded candidates, with context and disambiguation tasks only.

    A candidate is not an approved concept, translation or semantic verdict.
    """
    expected={x['lesson_id']:x for x in registry['lessons']}
    collection=defaultdict(list)
    if len(expected)!=len(lessons):raise ValueError('Lesson set does not match registry')
    for lesson_id,prepared in lessons:
        if lesson_id not in expected:raise ValueError('Unknown lesson')
        meta, units=_prepared_info(prepared)
        if meta['source_master_data_fingerprint']!=expected[lesson_id]['source_fingerprint']:
            raise ValueError('Source fingerprint changed; do not reuse review queue')
        course=meta.get('course') or {}
        candidates=extract_candidates(units, course, max_terms=max_terms_per_lesson)
        for item in candidates:
            matches=source_evidence(prepared,item['source_term'])
            collection[item['source_term']].append({'lesson_id':lesson_id,
                   'source_refs':[{'scoped_unit_id':f'{lesson_id}/{e["unit_id"]}',
                                   'original_unit_id':e['unit_id'],
                                   'source_refs':e['source_refs']} for e in matches],
                   'context_examples':[x['source_text'] for x in matches[:3]],
                   'count_units':len(matches)})
    items=[]
    for term, occurrences in collection.items():
        items.append({'source_term':term,'course_id':registry['course_identity']['id'],
                      'observed_in_lessons':[x['lesson_id'] for x in occurrences],
                      'evidence':occurrences,'review_status':'candidate',
                      'sense':None,'preferred_ar':None,'keep_english':None,
                      'context_validation':'not_performed',
                      'model_validation':'not_performed',
                      'prompt_version':'contextual_term_review_v2_8',
                      'questions':['What does this term mean in each source context?',
                                   'Could it have more than one meaning?',
                                   'What Arabic equivalent is correct for each distinct sense?',
                                   'Must code, notation or a proper name remain untranslated?']})
    items.sort(key=lambda x:(-len(x['observed_in_lessons']),x['source_term']))
    return {'schema_version':'terminology-queue-v2.8','course_identity':clone(registry['course_identity']),
            'source_collection_fingerprint':registry['source_collection_fingerprint'],
            'candidate_count':len(items),'candidates':items,
            'approved_count':0,'model_calls':0,'review_required':True}


def _decision_ok(term, course_id):
    if term.get('review_status')!='approved':raise PermissionError('Only explicitly approved terms enter memory')
    if term.get('course_id')!=course_id:raise ValueError('Cross-course approval is prohibited')
    approval=term.get('approval') or {}
    required=('reviewer','decision_id','approved_at','confirmed_course_id')
    if any(not _clean_name(approval.get(x)) for x in required):
        raise PermissionError('Auditable human approval required')
    if approval['confirmed_course_id']!=course_id:raise ValueError('Approval course mismatch')
    try:
        dt=datetime.fromisoformat(str(approval['approved_at']).replace('Z','+00:00'))
        if dt.tzinfo is None:raise ValueError('Missing timezone')
    except (ValueError,TypeError) as exc:raise ValueError('Invalid approval timestamp') from exc
    if not _clean_name(term.get('source_term')) or not _clean_name(term.get('sense')) or not _clean_name(term.get('preferred_ar')):
        raise ValueError('Approved term must include its form, contextual sense and Arabic')


def build_glossary_memory(registry, lessons, *, human_approved=(), previous_memory=None):
    """Versioned memory snapshot with complete decision provenance.

    Rejects conflicting renderings and duplicate decision IDs instead of
    overwriting historical approvals. Never creates a human approval.
    """
    identity=registry['course_identity']
    if not identity['cross_lesson_reuse_permitted']:
        if human_approved or previous_memory:
            raise PermissionError('Confirm a canonical course ID/title before sharing approved terms')
        return {'schema_version':'approved-course-memory-v2.8','version':0,
                'course_id':None,'status':'unconfirmed_no_shared_memory',
                'terms':[],'source_collection_fingerprint':registry['source_collection_fingerprint']}
    course_id=identity['id']
    if previous_memory:
        if previous_memory.get('course_id')!=course_id:raise ValueError('Cross-course memory rejected')
        old=list(previous_memory.get('terms') or [])
        version=previous_memory.get('version')
        if type(version) is not int or version<1:raise ValueError('Invalid previous memory version')
    else:old=[];version=0
    approved=[clone(t) for t in human_approved]
    all_entries=old+approved
    seen_decisions=set();terms={};audit=[]
    for t in all_entries:
        _decision_ok(t,course_id)
        decision=t['approval']['decision_id']
        if decision in seen_decisions:raise ValueError('Duplicate approval decision ID')
        seen_decisions.add(decision)
        # Validate the term was actually observed somewhere in this course snapshot.
        matches=[(lid,u['id']) for lid,p in lessons for u in p['translation_units']
                 if _occurs(unit_text(u),t['source_term'])]
        if not matches:
            historical_audit=(previous_memory or {}).get('audit') or []
            matching_audit=next((a for a in historical_audit if a.get('decision_id')==decision and
                                 a.get('source_term')==t['source_term'] and
                                 a.get('sense')==t['sense'] and a.get('evidence_unit_ids')),None)
            if t in approved or not matching_audit:
                raise ValueError('Approved term has no current or independently audited historical lesson evidence')
        key=(t['source_term'].casefold(),t['sense'].casefold())
        # A changed mapping requires an explicit revision workflow, not silent replacement.
        if key in terms:
            if (terms[key]['preferred_ar'],terms[key].get('keep_english'))!=(t['preferred_ar'],t.get('keep_english')):
                raise ValueError('Conflicting approvals: use a reviewed amendment instead of merging')
            raise ValueError('Duplicate approval for the same term and sense')
        terms[key]=t
        old_audit=next((a for a in (previous_memory or {}).get('audit',[])
                        if a.get('decision_id')==decision),None)
        audit.append({'decision_id':decision,'source_term':t['source_term'],
                      'sense':t['sense'],'evidence_unit_ids':([f'{lid}/{uid}' for lid,uid in matches]
                       or (old_audit or {}).get('evidence_unit_ids',[]))})
    # A previous snapshot cannot be silently rolled back.
    out_version=version+(1 if approved or not previous_memory else 0)
    return {'schema_version':'approved-course-memory-v2.8','version':out_version,
            'course_id':course_id,'course_title':identity['title'],'status':'reviewed_decisions_only',
            'terms':list(terms.values()),'audit':audit,
            'source_collection_fingerprint':registry['source_collection_fingerprint'],
            'source_lessons':[x['lesson_id'] for x in registry['lessons']],
            'automatic_approvals':0,'network_calls':0}


def cross_lesson_consistency(registry, lessons, memory):
    """Find potential literal inconsistencies; never assert semantic equality.

    Terms with multiple approved senses cannot be applied to every occurrence.
    An unscoped occurrence is sent to disambiguation rather than marked wrong.
    """
    if memory.get('course_id')!=registry['course_identity']['id']:
        raise ValueError('Memory does not belong to this registry')
    findings=[]
    grouped=defaultdict(list)
    for term in memory.get('terms') or []:
        _decision_ok(term,memory['course_id'])
        grouped[term['source_term'].casefold()].append(term)
    for term_group in grouped.values():
        distinct_senses={term['sense'].casefold() for term in term_group}
        multi_sense=len(distinct_senses)>1
        english=term_group[0]['source_term']
        for lid,prepared in lessons:
            for unit in prepared['translation_units']:
                if not _occurs(unit_text(unit),english):continue
                scoped=f'{lid}/{unit["id"]}'
                applicable=[t for t in term_group if not multi_sense or
                            scoped in (t.get('sense_unit_ids') or [])]
                if multi_sense and len(applicable)!=1:
                    findings.append({'scoped_unit_id':scoped,'source_term':english,
                         'status':'ambiguous_sense_needs_review','semantic_validated':False,
                         'possible_senses':[t['sense'] for t in term_group],
                         'source_refs':clone(unit.get('source_refs') or [])})
                    continue
                term=applicable[0]
                ar=_clean_name((unit.get('translation') or {}).get('natural_ar'))
                variants=[term.get('preferred_ar')]+list(term.get('alternatives') or [])
                matched=any(_arabic_match(ar,v) for v in variants if v)
                findings.append({'scoped_unit_id':scoped,
                                 'source_term':english,'sense':term['sense'],
                                 'expected_ar':term['preferred_ar'],
                                 'status':'literal_match_only' if matched else 'potential_mismatch_needs_review',
                                 'semantic_validated':False,
                                 'source_refs':clone(unit.get('source_refs') or [])})
    return {'findings':findings,
            'pending_review_count':sum(f['status']!='literal_match_only' for f in findings),
            'final_qa_approved':False,
            'note':'Ambiguous senses require disambiguation; inflection, paraphrase and context require human review.'}


def build_pipeline_snapshot(lessons, *, canonical_course=None, human_approved=(), previous_memory=None):
    registry=build_course_registry(lessons,canonical_course=canonical_course)
    queue=terminology_review_queue(registry,lessons)
    memory=build_glossary_memory(registry,lessons,human_approved=human_approved,previous_memory=previous_memory)
    consistency=cross_lesson_consistency(registry,lessons,memory)
    return {'schema_version':'course-pipeline-v2.8','registry':registry,
            'term_review_queue':queue,'glossary_memory':memory,
            'terminology_consistency_qa':consistency,
            'qa_gate':{'ready_for_final_dub':False,'semantic_review_done':False,
                       'audio_review_done':False,'timing_review_done':False},
            'source_unchanged':True,'api_calls':0,'tts_calls':0}