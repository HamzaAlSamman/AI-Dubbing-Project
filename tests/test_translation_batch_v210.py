import copy
import json
import tempfile
import unittest
from pathlib import Path
from arabic_dubbing_v2.common import fingerprint
from arabic_dubbing_v2.translation_batch_v210 import (build_batch_plan, run_batch_plan,
            _validate_batch, MODEL_DEFAULT)

def master(topic, count=33, translated=False, course_id=None):
    units=[]
    for i in range(count):
        u={'id':f'tu_{i+1:04d}', 'source':{'text':f'This lesson explains {topic} with 12 demonstrations; code variable x matters.'},
           'timing':{'source_start':i*5.0,'source_end':(i+1)*5.0},
           'source_refs':[{'segment_id':f'seg_{i:04d}','word_start_index':0,'word_end_index':6}]}
        if translated: u['translation']={'natural_ar':'يجب تجاهل الترجمة القديمة'}
        units.append(u)
    return {'course':{'id':course_id,'title':topic},'translation_units':units}

def make_response(model,instruction,payload):
    return {'results':[{'unit_id':x['unit_id'], 'accurate_ar':'شرح تجريبي للرقم 12 والمتغير x',
                        'natural_ar':'يشرح هذا الدرس 12 تجربة والمتغير x.', 'uncertainties':[]}
                       for x in payload['units']]}

class CourseBatchTests(unittest.TestCase):
    def test_any_course_no_existing_arabic(self):
        for topic in ('Radiology','Pottery & Kilns','Economics','Programming'):
            m=master(topic)
            plan=build_batch_plan(m,units_per_request=12)
            self.assertEqual(plan['total_units'],33)
            self.assertEqual(plan['planned_requests'],3)
            self.assertEqual([len(x['unit_ids']) for x in plan['jobs']],[12,12,9])
            self.assertEqual(plan['model_id'],MODEL_DEFAULT)

    def test_old_translation_is_not_used(self):
        m=master('Organic chemistry',translated=True)
        p=build_batch_plan(m,units_per_request=9)
        self.assertNotIn('الترجمة القديمة',json.dumps(p,ensure_ascii=False))

    def test_empty_title_does_not_guess_domain(self):
        m=master('Hydrology',count=2)
        m['course']={}
        p=build_batch_plan(m)
        self.assertIsNone(p['course_id'])
        self.assertIsNone(p['jobs'][0]['payload']['course']['title'])

    def test_source_fingerprint_tracks_timing_and_refs(self):
        m=master('Algorithms',count=2)
        a=build_batch_plan(m)
        m['translation_units'][1]['timing']['source_start']=4.5
        b=build_batch_plan(m)
        self.assertNotEqual(a['source_fingerprint'],b['source_fingerprint'])

    def test_cross_course_approved_terms_not_leaked(self):
        m=master('Biology',count=3,course_id='biology_101')
        m['terminology_v2']={'approved':[{'source_term':'lesson','preferred_ar':'درس','course_id':'other_course','status':'approved'},
                                                 {'source_term':'demonstrations','preferred_ar':'تجارب','course_id':'biology_101','status':'approved'}]}
        plan=build_batch_plan(m)
        a=plan['jobs'][0]['payload']['approved_course_terms']
        self.assertEqual([x['source_term'] for x in a],['demonstrations'])

    def test_reject_oversized_unit_without_ellipsis(self):
        m=master('History',count=2)
        m['translation_units'][0]['source']['text']='Long sentence '*200
        with self.assertRaises(ValueError):build_batch_plan(m,max_input_chars=1000)

    def test_selected_scope_no_reordering(self):
        p=build_batch_plan(master('Literature',count=5),selected_unit_ids=['tu_0005','tu_0002'])
        self.assertEqual(p['jobs'][0]['unit_ids'],['tu_0002','tu_0005'])

    def test_plan_never_calls_backend(self):
        plan=build_batch_plan(master('Law'))
        r=run_batch_plan(plan,'/none',backend=lambda *_: 1)
        self.assertEqual(r['status'],'PLAN_ONLY_NO_API_CALLS')

    def test_reject_auto_execution_without_permission(self):
        plan=build_batch_plan(master('Math'))
        with self.assertRaises(PermissionError):run_batch_plan(plan,'/none',execute=True,backend=make_response)

    def test_requires_request_budget_before_provider(self):
        calls=[]
        def backend(*a):calls.append(a);return make_response(*a)
        plan=build_batch_plan(master('Physics',count=33),units_per_request=10)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):run_batch_plan(plan,d,execute=True,allow_online=True,
                free_tier_confirmed=True,backend=backend,max_new_requests=3)
        self.assertEqual(calls,[])

    def test_resume_saved_candidates_zero_network(self):
        m=master('Mathematics',count=7)
        origin=fingerprint(m)
        p=build_batch_plan(m,units_per_request=3)
        calls=[]
        def backend(*a):calls.append(1);return make_response(*a)
        with tempfile.TemporaryDirectory() as d:
            opts=dict(execute=True,allow_online=True,free_tier_confirmed=True,backend=backend)
            first=run_batch_plan(p,d,**opts)
            self.assertEqual(first['new_provider_requests'],3)
            again=run_batch_plan(p,d,**opts)
            self.assertEqual(again['new_provider_requests'],0)
            self.assertEqual(again['resumed_batches'],3)
            self.assertEqual(len(calls),3)
            for file in Path(d).glob('candidate_*.json'):
                data=json.loads(file.read_text())
                self.assertFalse(data['human_approved'])
                self.assertIsNone(data['final_dub_ar'])
            self.assertEqual(fingerprint(m),origin)

    def test_tampered_cache_blocks_all_calls(self):
        p=build_batch_plan(master('Gardening',count=2))
        with tempfile.TemporaryDirectory() as d:
            opts=dict(execute=True,allow_online=True,free_tier_confirmed=True)
            run_batch_plan(p,d,backend=make_response,**opts)
            path=next(Path(d).glob('candidate*.json'))
            doc=json.loads(path.read_text())
            doc['results'][0]['unit_id']='evil'
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):run_batch_plan(p,d,backend=lambda *a:self.fail('provider must not run'),**opts)

    def test_reject_missing_or_extra_unit_in_response(self):
        p=build_batch_plan(master('Astronomy',count=4))
        with self.assertRaises(ValueError): _validate_batch({'results':[{'unit_id':'bad'}]},p['jobs'][0])

    def test_rate_pacing_by_config(self):
        p=build_batch_plan(master('History',count=3),units_per_request=1)
        seen=[]; t=[0.0]
        def tick():return t[0]
        def snooze(n):seen.append(n);t[0]+=n
        with tempfile.TemporaryDirectory() as d:
            run_batch_plan(p,d,execute=True,allow_online=True,free_tier_confirmed=True,
                          backend=make_response,rpm_limit=12,sleep=snooze,clock=tick)
        self.assertEqual(len(seen),2)
        self.assertTrue(all(abs(x-5.0)<1e-6 for x in seen))

    def test_generic_preview_fits_free_tier(self):
        p=build_batch_plan(master('Accounting',count=500),units_per_request=20)
        self.assertEqual(p['planned_requests'],25)

if __name__=='__main__':unittest.main()