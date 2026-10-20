"""Read-only source Master enrichment: Figma sections 02.9–02.27.

No original ASR/word timings/translation fields are changed. A new Master is
returned; caller must save to a NEW filename.
"""
from datetime import datetime, timezone
from .common import clone, fingerprint, unit_text, unit_times, sha256_file, read_json, save_new_json
from .terminology import effective_course, glossary_for_course
from .context import link_units
from .qa import qa_unit


def enrich_master(source_master, *, course_overrides=None, source_filename=None, source_sha256=None):
    if not isinstance(source_master,dict) or not isinstance(source_master.get('translation_units'),list):
        raise ValueError('Expected master JSON with translation_units list')
    source=clone(source_master)
    units=source['translation_units']
    if not units: raise ValueError('Master has no translation_units')
    for u in units:
        if not str(u.get('id') or ''):raise ValueError('Every unit must have an ID')
        unit_times(u)
        if not unit_text(u):raise ValueError(f'Missing source text in {u["id"]}')
    links=link_units(units,window=2)
    course=effective_course(source.get('course') or {},units,course_overrides)
    old_terminology=source.get('terminology') or []
    glossary=glossary_for_course(course,units,old_terminology)
    enriched=clone(source)
    enriched['translation_v2_metadata']={
        'implementation_version':'2.0.0-dev',
        'documented_scope':'Educational courses; initial English and single speaker',
        'source_master_filename':source_filename,
        'source_master_sha256':source_sha256,
        'source_master_data_fingerprint':fingerprint(source),
        'source_fields_unchanged':True,
        'course':course,
        'context_links':links,
        'translation_execution':'NOT_RUN',
        'tts_execution':'NOT_RUN',
        'human_final_approval':False,
        'note':'All V2 data is derived. Source ASR, timestamps, existing translations, IDs and word refs are unchanged.'}
    enriched['terminology_v2']=glossary
    issues={}
    for u in enriched['translation_units']:
        old_translation=u.get('translation') or {}
        natural=old_translation.get('natural_ar') or ''
        accurate=old_translation.get('accurate_ar') or ''
        qa=qa_unit(u,natural)
        if not accurate: qa['issues'].append('no_accurate_ar_reference')
        qa['status']='needs_review'
        u['analysis_v2']={
            'translation_origin':'existing_source_master_not_regenerated',
            'accurate_ar_available':bool(str(accurate).strip()),
            'natural_ar_available':bool(str(natural).strip()),
            'final_dub_ar_source_value_preserved_not_approved':bool(str(old_translation.get('final_dub_ar') or '').strip()),
            'qa':qa,
            'review_gate':'pending_human_bilingual_review',
            'target_duration_seconds':round(unit_times(u)[1]-unit_times(u)[0],4),
            'estimated_duration_seconds':None,
            'timing_status':'not_measured_by_tts',
            'spoken_form_status':'not_reviewed',
            'processing':{'pipeline':'translation_enrichment_v2','prompt_calls':0,'tts_calls':0}
        }
        if qa['issues']: issues[u['id']]=qa['issues']
    enriched['translation_v2_report']={
        'translation_unit_count':len(units),
        'source_master_unchanged': True,
        'units_with_natural_ar':sum(bool((u.get('translation') or {}).get('natural_ar')) for u in units),
        'units_needing_bilingual_review':len(units),
        'approved_for_final_tts':0,
        'glossary_candidates':len(glossary['candidates']),
        'approved_terms_for_this_course':len(glossary['approved']),
        'previous_unverified_terms':len(glossary['unverified_existing_terms']),
        'flags':issues,
        'note':'Mechanical flags are not a pass/fail semantic score; all units remain pending.'}
    return enriched


def prepare_project(input_path, output_path, *, course_overrides=None):
    src=read_json(input_path)
    original_sha=sha256_file(input_path)
    derived=enrich_master(src,course_overrides=course_overrides,
                          source_filename=str(input_path),source_sha256=original_sha)
    save_new_json(output_path, derived)
    if sha256_file(input_path)!=original_sha:
        raise RuntimeError('Source input changed during processing; do not rely on this result')
    return derived
