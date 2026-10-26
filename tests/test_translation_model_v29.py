import json
import tempfile
import unittest
from pathlib import Path
from arabic_dubbing_v2.common import fingerprint
from arabic_dubbing_v2.workflow import enrich_master
from arabic_dubbing_v2.translation_model_v29 import (build_translation_model_plan,
      run_translation_model_evaluation, PRIMARY_MODEL, COMPARISON_MODEL)
from test_pipeline import demo_master

class TranslationModelTests(unittest.TestCase):
    def setUp(self):
        self.prepared = enrich_master(demo_master())

    def fake_backend(self, model):
        def call(instruction, payload, version):
            self.calls.append((model, payload['unit_id']))
            return {'unit_id':payload['unit_id'], 'accurate_ar':'ترجمة حرفية تحافظ على المعنى',
                    'natural_ar':'ترجمة طبيعية مناسبة للإلقاء',
                    'terminology_candidates':[{'source_term':'Coral reef','preferred_ar':'الشعاب المرجانية',
                                                'sense':'Marine ecosystem','rationale':'In the source'}]
                    if 'Coral reef' in payload['source_text'] else [],
                    'uncertainties':[]}
        return call

    def test_plan_model_first_has_six_jobs_but_zero_provider_calls(self):
        r = build_translation_model_plan(self.prepared,['tu_0001','tu_0002','tu_0003'])
        self.assertEqual(r['planned_requests'],6)
        self.assertEqual(r['status'],'PLAN_ONLY_NO_API_REQUESTS')
        self.assertEqual(set(j['model_id'] for j in r['jobs']),{PRIMARY_MODEL, COMPARISON_MODEL})

    def test_pure_input_does_not_include_existing_arabic(self):
        plan=build_translation_model_plan(self.prepared,['tu_0002'],models=[PRIMARY_MODEL])
        payload=plan['jobs'][0]['inputs']
        self.assertNotIn('existing_natural_ar',payload)
        self.assertTrue(payload['previous_source_text'])
        self.assertTrue(payload['next_source_text'])
        self.assertNotIn('الشعاب المرجانية',json.dumps(payload,ensure_ascii=False))

    def test_invalid_unit_and_unknown_model_rejected(self):
        with self.assertRaises(KeyError):build_translation_model_plan(self.prepared,['fake'])
        with self.assertRaises(ValueError):build_translation_model_plan(self.prepared,['tu_0001'],models=['not-a-model'])
        with self.assertRaises(ValueError):build_translation_model_plan(self.prepared,['tu_0001','tu_0001'])

    def test_default_run_is_only_plan(self):
        self.calls=[]
        r=run_translation_model_evaluation(self.prepared,'/no-write',unit_ids=['tu_0001'],backend_factory=self.fake_backend)
        self.assertEqual(r['status'],'PLAN_ONLY_NO_API_REQUESTS')
        self.assertEqual(self.calls,[])

    def test_gate_requires_online_and_free_tier_confirmation(self):
        self.calls=[]
        for kwargs in [{'execute':True}, {'execute':True,'allow_online':True},
                       {'execute':True,'free_tier_confirmed':True}]:
            with self.assertRaises(PermissionError):
                run_translation_model_evaluation(self.prepared,'/no-write',unit_ids=['tu_0001'],
                                                 backend_factory=self.fake_backend,**kwargs)
        self.assertEqual(self.calls,[])

    def test_mock_evaluation_writes_only_candidates_and_original_intact(self):
        self.calls=[]
        before=fingerprint(self.prepared)
        with tempfile.TemporaryDirectory() as d:
            r=run_translation_model_evaluation(self.prepared,d,unit_ids=['tu_0001'],models=[PRIMARY_MODEL],
                      backend_factory=self.fake_backend,execute=True,allow_online=True,
                      free_tier_confirmed=True,max_requests=1)
            self.assertEqual(r['request_count'],1)
            doc=json.loads(Path(r['results'][0]['path']).read_text())
            self.assertFalse(doc['human_approved'])
            self.assertIsNone(doc['final_dub_ar'])
            self.assertEqual(doc['terminology_candidates'][0]['review_status'],'candidate')
            self.assertEqual(self.calls,[(PRIMARY_MODEL,'tu_0001')])
            self.assertEqual(before,fingerprint(self.prepared))

    def test_preflight_existing_output_prevents_calls(self):
        self.calls=[]
        with tempfile.TemporaryDirectory() as d:
            kw=dict(unit_ids=['tu_0001'],models=[PRIMARY_MODEL],backend_factory=self.fake_backend,
                    execute=True,allow_online=True,free_tier_confirmed=True,max_requests=1)
            run_translation_model_evaluation(self.prepared,d,**kw)
            with self.assertRaises(FileExistsError):run_translation_model_evaluation(self.prepared,d,**kw)
            self.assertEqual(len(self.calls),1)

    def test_wrong_unit_or_fabricated_term_rejected_and_no_file_written(self):
        for bad in [dict(unit_id='wrong'),dict(unit_id='tu_0001',terminology_candidates=[{'source_term':'unicorn','preferred_ar':'وحيد القرن'}])]:
            def backend(model):
                def call(*args):
                    return {'unit_id':'tu_0001','accurate_ar':'ترجمة','natural_ar':'ترجمة','uncertainties':[],
                            'terminology_candidates':[],**bad}
                return call
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):run_translation_model_evaluation(self.prepared,d,unit_ids=['tu_0001'],
                  models=[PRIMARY_MODEL],backend_factory=backend,execute=True,allow_online=True,free_tier_confirmed=True,max_requests=1)
                self.assertEqual(list(Path(d).glob('*.json')),[])

    def test_request_budget_blocks_before_api(self):
        self.calls=[]
        with self.assertRaises(ValueError):run_translation_model_evaluation(self.prepared,'/nothing',unit_ids=['tu_0001'],
             backend_factory=self.fake_backend,execute=True,allow_online=True,free_tier_confirmed=True,max_requests=1)
        self.assertEqual(self.calls,[])

if __name__=='__main__':unittest.main()