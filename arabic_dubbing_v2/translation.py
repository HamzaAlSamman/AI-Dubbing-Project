"""Multi-pass translation plans. No provider credentials or automatic calls.

An explicit injected backend may be used later. Original translation values
remain immutable in the source master; generated outputs are proposals only.
"""
import json
from .common import unit_text
from .context import build_context_package

_STYLE = ('Target: natural, clear, easy-to-speak Modern Standard Arabic (فصحى مبسطة). '
          'Preserve technical meaning, negation, numbers, notation, examples, and teaching steps. '
          'Do not fabricate facts, names, gender, or visual cues. Preserve identifiers and code.')


def pass_plan(prepared, unit_id):
    gloss = prepared.get('terminology_v2') or {}
    context = build_context_package(prepared,unit_id,
                    links=prepared['translation_v2_metadata']['context_links'],glossary=gloss)
    unit = next(u for u in prepared['translation_units'] if u['id'] == unit_id)
    existing = unit.get('translation') or {}
    return {
        'id':unit_id,'model_strategy':'multi_pass_single_model_allowed',
        'context':context,
        'passes':[
            {'id':'accurate','version':'accurate_v2',
             'instruction':_STYLE+' Produce faithful accurate_ar without shortening. Output JSON {"accurate_ar": string, "flags": string[]}.',
             'input':{'source_text':unit_text(unit),'context':context,'existing_accurate_ar':existing.get('accurate_ar')}},
            {'id':'natural','version':'natural_v2',
             'instruction':_STYLE+' Transform accurate_ar into spoken natural_ar without changing meaning. Output JSON {"natural_ar": string, "flags": string[]}.',
             'input':{'accurate_ar':'<generated-or-reviewed-accurate_ar>', 'source_text':unit_text(unit),'context':context}},
            {'id':'timing_candidate','version':'timing_v2',
             'instruction':_STYLE+' Produce a timing_candidate_ar only if meaning is preserved. Never silently omit a step. Output JSON {"timing_candidate_ar": string, "flags": string[]}.',
             'input':{'natural_ar':'<generated-or-reviewed-natural_ar>', 'target_duration_seconds':unit['timing'].get('source_duration',unit['timing'].get('source_end',0)-unit['timing'].get('source_start',0))}},
            {'id':'semantic_qa','version':'semanticqa_v2',
             'instruction':('Compare English source to each Arabic candidate. Check meaning, terminology, numbers, '
                            'identifiers, negation, steps, modality, hallucination and pronouns. Return specific evidence; '
                            'NEVER set human_approved or final_dub_ar. Output JSON {"issues": string[], "needs_review": boolean, "rationale": string}.'),
             'input':{'source_text':unit_text(unit),'accurate_ar':'<draft>', 'natural_ar':'<draft>', 'timing_candidate_ar':'<draft>', 'context':context}}
        ]}


def build_all_pass_plans(prepared):
    return {'status':'prompts_only_no_provider_calls', 'course_id':prepared['translation_v2_metadata']['course']['id'],
            'plans':[pass_plan(prepared,u['id']) for u in prepared['translation_units']],
            'notice':'No LLM executed. Do not put final_dub_ar into any output before a human approval gate.'}


def generate_draft(prepared, unit_id, backend, *, allow_translation=False):
    """Explicit opt-in only; backend callable returns dict with field each pass needs.

    This function cannot approve a translation and does not modify the master.
    Backend may be locally hosted or externally metered; caller is responsible
    for permissions and cost before allow_translation=True.
    """
    if not allow_translation:
        raise PermissionError('Translation disabled. Explicit allow_translation=True required.')
    plan=pass_plan(prepared,unit_id)
    result={'unit_id':unit_id,'status':'unreviewed_generated_candidate','human_approved':False,'final_dub_ar':None,
            'passes':{},'model_provenance':[]}
    for p in plan['passes']:
        inp=dict(p['input'])
        if p['id']=='natural':inp['accurate_ar']=result['passes']['accurate']['accurate_ar']
        if p['id']=='timing_candidate':inp['natural_ar']=result['passes']['natural']['natural_ar']
        if p['id']=='semantic_qa':
            inp.update(accurate_ar=result['passes']['accurate']['accurate_ar'],
                       natural_ar=result['passes']['natural']['natural_ar'],
                       timing_candidate_ar=result['passes']['timing_candidate']['timing_candidate_ar'])
        response=backend(p['instruction'],inp,p['version'])
        if not isinstance(response,dict):raise TypeError('Backend must return parsed dict')
        expected = {'accurate':'accurate_ar','natural':'natural_ar','timing_candidate':'timing_candidate_ar'}
        if p['id'] in expected and not str(response.get(expected[p['id']]) or '').strip():
            raise ValueError(f'Missing {expected[p["id"]]} in {p["id"]} output')
        result['passes'][p['id']]=response
        result['model_provenance'].append({'pass':p['id'],'prompt_version':p['version'],
                                           'provider':getattr(backend,'provider_name','user_injected_backend')})
    return result
