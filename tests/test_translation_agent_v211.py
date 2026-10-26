import copy
import json
import tempfile
import unittest
from pathlib import Path
from arabic_dubbing_v2.common import fingerprint
from arabic_dubbing_v2.translation_agent_v211 import (
    master_from_aligned_whisperx, plan_agent, execute_agent,
    _answer_translation, _answer_review, TRANSLATOR_PROMPT, REVIEWER_PROMPT)


def master(topic='marine ecology', n=5, legacy=False):
    units=[]
    for i in range(n):
        source=f'The {topic} module explains coral reef ecosystems.'
        u={'id':f'tu_{i+1:04d}','source':{'text':source},
           'source_refs':[{'segment_id':f'seg_{i+1:04d}', 'word_start_index':0,'word_end_index':5}],
           'timing':{'source_start':i*5.,'source_end':i*5+4.}}
        if legacy:u['translation']={'natural_ar':'LEGACY SHOULD NOT APPEAR'}
        units.append(u)
    return {'course':{'id':'marine_101','title':topic},'translation_units':units}

class Mock:
    def __init__(self):self.calls=[]
    def __call__(self,model,prompt,payload):
        self.calls.append((model,prompt,copy.deepcopy(payload)))
        if prompt==TRANSLATOR_PROMPT:
            x=payload['current_unit'];uid=x['unit_id']
            return {'unit_id':uid,'accurate_ar':'تشرح الوحدة النظم البيئية للشعاب المرجانية.',
                    'natural_ar':'نتعرف في هذه الوحدة على النظم البيئية للشعاب المرجانية.',
                    'uncertainties':[], 'proposed_terms':[{'source_term':'coral reef','preferred_ar':'الشعاب المرجانية','sense':'ecology'}]}
        if prompt==REVIEWER_PROMPT:
            return {'unit_id':payload['unit_id'],'needs_review':False,'issues':[],'rationale':'لا أخطاء ظاهرة'}
        raise AssertionError('unknown role')


class AgentTests(unittest.TestCase):
    def test_plan_starts_without_translation_in_any_domain(self):
        for topic in ('oceanography','Pottery','Economics','Surgery'):
            p=plan_agent(master(topic,n=20))
            self.assertEqual(p['planned_unit_count'],20)
            self.assertEqual(len(p['jobs']),20)
            self.assertEqual(p['strategy'],'incremental_one_unit_at_a_time')
            self.assertTrue(all('source_text' in j['task'] for j in p['jobs']))

    def test_legacy_text_is_never_in_payload(self):
        p=plan_agent(master(legacy=True))
        self.assertNotIn('LEGACY SHOULD NOT APPEAR',json.dumps(p))

    def test_no_api_by_default(self):
        p=plan_agent(master())
        m=Mock();r=execute_agent(p,master(),'/unused',backend=m,review_backend=m)
        self.assertEqual(r['new_model_calls'] if 'new_model_calls' in r else r['provider_calls'],0)
        self.assertFalse(m.calls)

    def test_requires_explicit_local_or_free_online(self):
        s=master();p=plan_agent(s);mock=Mock()
        with self.assertRaises(PermissionError):
            execute_agent(p,s,'/unused',execute=True,backend=mock,review_backend=mock)
        self.assertEqual(len(mock.calls),0)

    def test_stops_with_budget_and_resumes_without_duplicate_calls(self):
        s=master(n=5);p=plan_agent(s);mock=Mock()
        with tempfile.TemporaryDirectory() as d:
            opts={'backend':mock,'review_backend':mock,'execute':True,'allow_local':True,'max_new_calls':2,'rpm_limit':1000}
            r1=execute_agent(p,s,d,**opts)
            self.assertEqual(r1['status'],'PARTIAL_PAUSED_BUDGET')
            self.assertEqual(r1['units_completed'],2)
            r2=execute_agent(p,s,d,**opts)
            self.assertEqual(r2['units_completed'],4)
            r3=execute_agent(p,s,d,**opts)
            self.assertEqual(r3['status'],'CANDIDATES_READY_FOR_HUMAN_REVIEW')
            self.assertEqual(len(mock.calls),5)
            self.assertEqual(len(list(Path(d).glob('translate_*.json'))),5)
            self.assertEqual(r3['human_approved_count'],0)

    def test_one_unit_only_per_translation_call(self):
        s=master(n=8);p=plan_agent(s);m=Mock()
        with tempfile.TemporaryDirectory() as d:
            execute_agent(p,s,d,backend=m,execute=True,allow_local=True,
                          review_policy='none',max_new_calls=8,rpm_limit=1000)
        self.assertEqual(len(m.calls),8)
        for _,prompt,payload in m.calls:
            self.assertEqual(prompt,TRANSLATOR_PROMPT)
            self.assertEqual(set(payload),{'current_unit','provisional_terms_unapproved','stage'})
            self.assertNotIn('units',payload)

    def test_provisional_memory_only_not_approved(self):
        s=master(n=2);p=plan_agent(s);m=Mock()
        with tempfile.TemporaryDirectory() as d:
            execute_agent(p,s,d,backend=m,execute=True,allow_local=True,review_policy='none',rpm_limit=1000)
        previous=m.calls[1][2]['provisional_terms_unapproved']
        self.assertEqual(previous[0]['status'],'provisional_not_approved')
        self.assertEqual(previous[0]['source_term'],'coral reef')

    def test_human_approved_glossary_course_scoped(self):
        s=master(n=2)
        s['terminology_v2']={'approved':[
            {'source_term':'coral reef','preferred_ar':'الشعاب المرجانية','course_id':'marine_101','review_status':'approved'},
            {'source_term':'ecosystems','preferred_ar':'الاقتصاد','course_id':'finance_101','review_status':'approved'}]}
        p=plan_agent(s)
        self.assertEqual([x['source_term'] for x in p['jobs'][0]['task']['approved_terms']],['coral reef'])

    def test_flags_trigger_independent_review_but_not_approval(self):
        s=master(n=2)
        s['translation_units'][0]['builder']={'review_reasons':['boundary_unverified']}
        p=plan_agent(s);m=Mock()
        with tempfile.TemporaryDirectory() as d:
            r=execute_agent(p,s,d,backend=m,review_backend=m,execute=True,allow_local=True,
                            review_policy='flagged',max_new_calls=3,rpm_limit=1000)
            self.assertEqual(len(m.calls),3)
            self.assertEqual(m.calls[1][1],REVIEWER_PROMPT)
            self.assertEqual(len(list(Path(d).glob('review_*.json'))),1)
            self.assertEqual(r['human_approved_count'],0)

    def test_review_budget_pause_keeps_translator_result(self):
        s=master(n=2)
        s['translation_units'][0]['builder']={'review_reasons':['boundary_unverified']}
        p=plan_agent(s);m=Mock()
        with tempfile.TemporaryDirectory() as d:
            first=execute_agent(p,s,d,backend=m,review_backend=m,execute=True,
                                allow_local=True,review_policy='flagged',max_new_calls=1,rpm_limit=1000)
            self.assertEqual(first['units_pending'],['tu_0001','tu_0002'])
            second=execute_agent(p,s,d,backend=m,review_backend=m,execute=True,
                                allow_local=True,review_policy='flagged',max_new_calls=2,rpm_limit=1000)
            self.assertEqual(second['status'],'CANDIDATES_READY_FOR_HUMAN_REVIEW')
            self.assertEqual([x[1] for x in m.calls],[TRANSLATOR_PROMPT,REVIEWER_PROMPT,TRANSLATOR_PROMPT])

    def test_cached_approval_tamper_blocks_call(self):
        s=master(n=2);p=plan_agent(s);m=Mock()
        with tempfile.TemporaryDirectory() as d:
            execute_agent(p,s,d,backend=m,execute=True,allow_local=True,review_policy='none',max_new_calls=1)
            path=next(Path(d).glob('translate_*.json'))
            v=json.loads(path.read_text());v['human_approved']=True
            path.write_text(json.dumps(v))
            with self.assertRaises(ValueError):execute_agent(p,s,d,backend=m,execute=True,
                allow_local=True,review_policy='none',max_new_calls=2)
            self.assertEqual(len(m.calls),1)

    def test_tampered_cached_arabic_never_trusted(self):
        s=master(n=1);p=plan_agent(s);m=Mock()
        with tempfile.TemporaryDirectory() as d:
            execute_agent(p,s,d,backend=m,execute=True,allow_local=True,review_policy='none')
            path=next(Path(d).glob('translate_*.json'))
            v=json.loads(path.read_text());v['result']['natural_ar']='نص مختلف'
            path.write_text(json.dumps(v))
            with self.assertRaises(ValueError):execute_agent(p,s,d,backend=m,execute=True,
                allow_local=True,review_policy='none')
            self.assertEqual(len(m.calls),1)

    def test_model_invented_source_term_rejected(self):
        j=plan_agent(master())['jobs'][0]
        with self.assertRaises(ValueError):_answer_translation({'unit_id':j['unit_id'],
            'accurate_ar':'أ', 'natural_ar':'ب', 'proposed_terms':[
                {'source_term':'neural network','preferred_ar':'شبكة عصبية'}]},j)

    def test_changed_source_ref_blocks_old_plan(self):
        s=master();p=plan_agent(s);s['translation_units'][0]['source_refs'][0]['word_end_index']=6
        with self.assertRaises(ValueError):execute_agent(p,s,'/unused')

    def test_aligned_word_content_change_blocks_resume(self):
        m=master(n=1)
        m['source_segments']=[{'id':'seg_0001','words':[{'word':'x','start':0.0,'end':1.0}]}]
        p=plan_agent(m)
        m['source_segments'][0]['words'][0]['word']='y'
        with self.assertRaises(ValueError):execute_agent(p,m,'/unused')

    def test_whisperx_adapter_preserves_original_segments_and_word_refs(self):
        segments=[{'id':'seg_0001','text':'Coral ecosystems.',
                  'words':[{'word':'Coral','start':0.1,'end':0.4},
                           {'word':'ecosystems.','start':0.45,'end':1.2}]}]
        before=fingerprint(segments)
        m=master_from_aligned_whisperx({'segments':segments})
        self.assertEqual(fingerprint(segments),before)
        self.assertEqual(m['builder_report']['source_word_count'],2)
        self.assertEqual(m['translation_units'][0]['translation']['natural_ar'],None)
        self.assertEqual(plan_agent(m)['planned_unit_count'],1)

    def test_missing_alignment_cannot_be_faked(self):
        with self.assertRaises(ValueError):master_from_aligned_whisperx({'segments':[
            {'text':'hello world','words':[{'word':'hello','end':0.3}]}]})

    def test_all_semantic_review_is_additional_call_not_same_batch(self):
        s=master(n=2);p=plan_agent(s);m=Mock()
        with tempfile.TemporaryDirectory() as d:
            r=execute_agent(p,s,d,backend=m,review_backend=m,execute=True,allow_local=True,
                            review_policy='all',max_new_calls=4,rpm_limit=1000)
            self.assertEqual(len(m.calls),4)
            self.assertEqual([x[1] for x in m.calls],[TRANSLATOR_PROMPT,REVIEWER_PROMPT]*2)
            self.assertEqual(r['units_completed'],2)

    def test_reviewer_response_cannot_approve(self):
        with self.assertRaises(ValueError):_answer_review({'unit_id':'tu_0001', 'needs_review':False,
            'issues':[],'rationale':'ok','human_approved':True},'tu_0001')

if __name__=='__main__':unittest.main()