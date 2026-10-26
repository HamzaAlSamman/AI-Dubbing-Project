import copy
import json
import tempfile
import unittest
from pathlib import Path

from arabic_dubbing_v2.translation_agent_v212 import (
    plan_autonomous, execute_autonomous, REPAIR_PROMPT_V212,
    REVIEW_PROMPT_V212, _artifact_path,
)
from arabic_dubbing_v2.translation_agent_v211 import TRANSLATOR_PROMPT


def new_course(count=3, text='Learn the geometry of circles.'):
    return {'course':{},'translation_units':[
        {'id':f'tu_{i:04d}', 'source':{'text':text},
         'timing':{'source_start':i*2,'source_end':i*2+1.8},
         'source_refs':[{'segment_id':f'seg_{i:04d}','word_start_index':0,'word_end_index':3}]}
        for i in range(1,count+1)]}


class Model:
    def __init__(self,bad=False,always_fail=False):
        self.calls=[];self.bad=bad;self.always_fail=always_fail
    def __call__(self,model,prompt,payload):
        self.calls.append((model,prompt,copy.deepcopy(payload)))
        if prompt==TRANSLATOR_PROMPT:
            uid=payload['current_unit']['unit_id']
            text='سنتعلم الهندسة والدائرة.'
            if self.bad: text='Learn geometry'
            return {'unit_id':uid,'accurate_ar':text,'natural_ar':text,
                    'uncertainties':[],'proposed_terms':[]}
        if prompt==REPAIR_PROMPT_V212:
            uid=payload['unit_id'];ar='سنتعلم الهندسة والدائرة.'
            return {'unit_id':uid,'accurate_ar':ar,'natural_ar':ar,
                    'uncertainties':[],'proposed_terms':[]}
        if prompt==REVIEW_PROMPT_V212:
            is_bad=self.always_fail or (self.bad and payload['candidate']['natural_ar']=='Learn geometry')
            return {'unit_id':payload['unit_id'],'needs_review':is_bad,
                    'issues':['wrong output language'] if is_bad else [],'rationale':'QA checked'}
        raise AssertionError('bad prompt')


class AutonomousTests(unittest.TestCase):
    def test_source_first_no_existing_arabic(self):
        s=new_course(5)
        p=plan_autonomous(s)
        self.assertEqual(p['planned_unit_count'],5)
        self.assertNotIn('human',p['approval'])
        self.assertEqual(p['strategy'],'autonomous_single_unit_translate_review_repair')

    def test_no_calls_by_default(self):
        s=new_course();m=Model();p=plan_autonomous(s)
        r=execute_autonomous(p,s,'/unused',backend=m,reviewer_backend=m)
        self.assertEqual(r['model_calls'],0)
        self.assertEqual(m.calls,[])

    def test_permission_gate(self):
        s=new_course();p=plan_autonomous(s);m=Model()
        with self.assertRaises(PermissionError):
            execute_autonomous(p,s,'/unused',backend=m,reviewer_backend=m,execute=True)
        self.assertFalse(m.calls)

    def test_automatic_pass_has_no_human_gate(self):
        s=new_course(2);p=plan_autonomous(s);m=Model()
        with tempfile.TemporaryDirectory() as d:
            r=execute_autonomous(p,s,d,execute=True,allow_local=True,backend=m,reviewer_backend=m,
                                 max_new_calls=4,rpm_limit=1000,sleep=lambda x:None)
            self.assertEqual(r['status'],'MACHINE_QA_CANDIDATES_READY_FOR_TTS_PREVIEW')
            self.assertEqual(r['units_passed_machine_qa'],2)
            self.assertFalse(r['final_dubbing_certified'])
            self.assertEqual([x[1] for x in m.calls],[TRANSLATOR_PROMPT,REVIEW_PROMPT_V212]*2)
            self.assertFalse(r['units'][0]['human_review_required'])

    def test_resume_with_budget_no_repeat(self):
        s=new_course(2);p=plan_autonomous(s);m=Model()
        with tempfile.TemporaryDirectory() as d:
            kw={'execute':True,'allow_local':True,'backend':m,'reviewer_backend':m,
                'max_new_calls':1,'rpm_limit':1000,'sleep':lambda x:None}
            out=[]
            for _ in range(4):out.append(execute_autonomous(p,s,d,**kw))
            self.assertEqual(out[-1]['units_passed_machine_qa'],2)
            self.assertEqual(len(m.calls),4)
            self.assertEqual(out[-1]['status'],'MACHINE_QA_CANDIDATES_READY_FOR_TTS_PREVIEW')

    def test_review_fail_triggers_repair_and_re_review(self):
        s=new_course(1);p=plan_autonomous(s,max_repairs=1);m=Model(bad=True)
        with tempfile.TemporaryDirectory() as d:
            r=execute_autonomous(p,s,d,execute=True,allow_local=True,backend=m,reviewer_backend=m,
                                 max_new_calls=4,rpm_limit=1000,sleep=lambda x:None)
            self.assertEqual(r['units_passed_machine_qa'],1)
            self.assertEqual(r['units'][0]['attempts'],2)
            self.assertEqual([x[1] for x in m.calls],[TRANSLATOR_PROMPT,REVIEW_PROMPT_V212,
                                                        REPAIR_PROMPT_V212,REVIEW_PROMPT_V212])

    def test_persistent_issues_quarantined_automatically(self):
        s=new_course(1);p=plan_autonomous(s,max_repairs=1);m=Model(always_fail=True)
        with tempfile.TemporaryDirectory() as d:
            r=execute_autonomous(p,s,d,execute=True,allow_local=True,backend=m,reviewer_backend=m,
                                 max_new_calls=4,rpm_limit=1000,sleep=lambda x:None)
            self.assertEqual(r['status'],'COMPLETED_WITH_MACHINE_BLOCKS')
            self.assertEqual(r['units_blocked_automatically'],1)
            self.assertIsNone(r['units'][0]['spoken_candidate_ar'])

    def test_missing_numbers_block_even_if_reviewer_passes(self):
        s=new_course(1,'Use 512 triangles in the example.')
        p=plan_autonomous(s,max_repairs=0);m=Model()
        with tempfile.TemporaryDirectory() as d:
            r=execute_autonomous(p,s,d,execute=True,allow_local=True,backend=m,reviewer_backend=m,
                                 max_new_calls=2,rpm_limit=1000,sleep=lambda x:None)
            self.assertEqual(r['units_blocked_automatically'],1)
            self.assertIn('numeric_fidelity_needs_review',r['units'][0]['issues'])

    def test_aligned_word_change_fails_without_calls(self):
        s=new_course(1);s['source_segments']=[{'id':'s1','words':[{'word':'circles','start':1,'end':2}]}]
        p=plan_autonomous(s);s['source_segments'][0]['words'][0]['end']=2.5;m=Model()
        with self.assertRaises(ValueError):execute_autonomous(p,s,'/unused',execute=True,
                    allow_local=True,backend=m,reviewer_backend=m)
        self.assertFalse(m.calls)

    def test_corrupt_prior_checkpoint_blocks_before_call(self):
        s=new_course(2);p=plan_autonomous(s);m=Model()
        with tempfile.TemporaryDirectory() as d:
            execute_autonomous(p,s,d,execute=True,allow_local=True,backend=m,reviewer_backend=m,
                               max_new_calls=1,rpm_limit=1000,sleep=lambda x:None)
            path=_artifact_path(Path(d),p['jobs'][0],'translate0')
            rec=json.loads(path.read_text());rec['result']['natural_ar']='tampered'
            path.write_text(json.dumps(rec))
            with self.assertRaises(ValueError):execute_autonomous(p,s,d,execute=True,allow_local=True,
                        backend=m,reviewer_backend=m,max_new_calls=5)
            self.assertEqual(len(m.calls),1)

    def test_backend_disallowed_approval_fields(self):
        s=new_course(1);p=plan_autonomous(s)
        class Bad(Model):
            def __call__(self,model,prompt,payload):
                x=super().__call__(model,prompt,payload)
                x['final_dub_ar']='مؤكد'
                return x
        m=Bad()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):execute_autonomous(p,s,d,execute=True,allow_local=True,
                   backend=m,reviewer_backend=m,max_new_calls=2)

    def test_raw_source_and_unit_ids_unchanged(self):
        s=new_course(1);s0=copy.deepcopy(s);p=plan_autonomous(s);m=Model()
        with tempfile.TemporaryDirectory() as d:
            execute_autonomous(p,s,d,execute=True,allow_local=True,backend=m,reviewer_backend=m,
                               max_new_calls=2,rpm_limit=1000,sleep=lambda x:None)
            self.assertEqual(s,s0)

    def test_invalid_plan_repair_count(self):
        for k in [-1,3,True]:
            with self.assertRaises(ValueError):plan_autonomous(new_course(),max_repairs=k)

    def test_all_stages_are_one_unit_only(self):
        s=new_course(3);p=plan_autonomous(s);m=Model()
        with tempfile.TemporaryDirectory() as d:
            execute_autonomous(p,s,d,execute=True,allow_local=True,backend=m,reviewer_backend=m,
                               max_new_calls=6,rpm_limit=1000,sleep=lambda x:None)
            self.assertEqual(len(m.calls),6)
            self.assertTrue(all('units' not in payload for _,_,payload in m.calls))

    def test_cached_pass_not_recomputed(self):
        s=new_course(1);p=plan_autonomous(s);m=Model()
        with tempfile.TemporaryDirectory() as d:
            kw={'execute':True,'allow_local':True,'backend':m,'reviewer_backend':m,
                'max_new_calls':2,'rpm_limit':1000,'sleep':lambda x:None}
            execute_autonomous(p,s,d,**kw)
            result=execute_autonomous(p,s,d,**kw)
            self.assertEqual(result['new_model_calls'],0)
            self.assertEqual(len(m.calls),2)

if __name__=='__main__': unittest.main()

class AdditionalAutonomousTests(unittest.TestCase):
    def test_machine_validated_term_memory_across_steps(self):
        s=new_course(2,'The coral reef has many fish.')
        p=plan_autonomous(s)
        class M(Model):
            def __call__(self,model,prompt,payload):
                out=super().__call__(model,prompt,payload)
                if prompt==TRANSLATOR_PROMPT:
                    out['proposed_terms']=[{'source_term':'coral reef',
                        'preferred_ar':'الشعاب المرجانية','sense':'marine biology'}]
                return out
        m=M()
        with tempfile.TemporaryDirectory() as d:
            execute_autonomous(p,s,d,execute=True,allow_local=True,backend=m,
               reviewer_backend=m,max_new_calls=4,rpm_limit=1000,sleep=lambda x:None)
            t=[z for z in m.calls if z[1]==TRANSLATOR_PROMPT]
            hints=t[1][2]['provisional_terms_unapproved']
            self.assertEqual(len(hints),1)
            self.assertEqual(hints[0]['source_term'],'coral reef')
            self.assertEqual(hints[0]['status'],'provisional_not_approved')

    def test_machine_term_memory_restored_on_resume(self):
        s=new_course(2,'The coral reef has many fish.')
        p=plan_autonomous(s)
        class M(Model):
            def __call__(self,model,prompt,payload):
                out=super().__call__(model,prompt,payload)
                if prompt==TRANSLATOR_PROMPT:
                    out['proposed_terms']=[{'source_term':'coral reef',
                        'preferred_ar':'الشعاب المرجانية','sense':'marine biology'}]
                return out
        m=M()
        with tempfile.TemporaryDirectory() as d:
            opts=dict(execute=True,allow_local=True,backend=m,reviewer_backend=m,
                      max_new_calls=2,rpm_limit=1000,sleep=lambda x:None)
            execute_autonomous(p,s,d,**opts)
            execute_autonomous(p,s,d,**opts)
            t=[z for z in m.calls if z[1]==TRANSLATOR_PROMPT]
            self.assertEqual(t[1][2]['provisional_terms_unapproved'][0]['source_term'],'coral reef')