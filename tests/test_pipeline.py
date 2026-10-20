import copy
import json
import tempfile
import unittest
from pathlib import Path
from arabic_dubbing_v2.boundaries import build_units_from_whisperx
from arabic_dubbing_v2.common import fingerprint, sha256_file, read_json
from arabic_dubbing_v2.workflow import enrich_master, prepare_project
from arabic_dubbing_v2.context import build_context_package
from arabic_dubbing_v2.translation import build_all_pass_plans, generate_draft
from arabic_dubbing_v2.speech import run_edge_previews
from arabic_dubbing_v2.qa import mechanical_qa
from arabic_dubbing_v2.terminology import validation_plans,propose_glossary_entries


def seg(text, tokens, start=0, speaker='SPEAKER_00'):
    return {'text':text,'speaker':speaker,
            'words':[{'word':w, 'start':start+i*.4,'end':start+i*.4+.3,'score':.9}
                     for i,w in enumerate(tokens)]}


def demo_master(title='Marine Biology',id='course_marine',topic='Marine ecosystems'):
    sentences=[
        ('Coral reef ecosystems contain diverse organisms.', 'الشعاب المرجانية تضم كائنات متنوعة.'),
        ('Coral reef ecosystems depend on sunlight.', 'تعتمد الشعاب المرجانية على ضوء الشمس.'),
        ('Marine biology explains these ecosystems.', 'يشرح علم الأحياء البحرية هذه النظم.')]
    return {'schema_version':'2.0','course':{'id':id,'title':title,'topic':topic},
            'terminology':[{'source_term':'neural network','preferred_ar':'شبكة عصبية',
                            'status':'approved','course_id':'other_course'}],
            'source_segments':[seg('Sample speech.',['Sample','speech.'])],
            'translation_units':[
              {'id':f'tu_{i:04d}','speaker':{'id':'SPEAKER_00'},
               'source_segment_ids':[f'seg_{i:04d}'],'source_refs':[{'segment_index':0,'word_index':0}],
               'timing':{'source_start':(i-1)*5+0.1,'source_end':i*5},
               'source':{'text':en},'translation':{'accurate_ar':ar,'natural_ar':ar,'final_dub_ar':None},
               'review':{'status':'pending'}} for i,(en,ar) in enumerate(sentences,1)]}


class PipelineTests(unittest.TestCase):
    def test_enrichment_keeps_all_prior_fields(self):
        original=demo_master(); original_fingerprint=fingerprint(original)
        derived=enrich_master(original)
        assert fingerprint(original)==original_fingerprint
        for field in ('course','terminology','source_segments','schema_version'):
            self.assertEqual(original[field],derived[field])
        for a,b in zip(original['translation_units'],derived['translation_units']):
            for field in a:self.assertEqual(a[field],b[field])
        self.assertEqual(derived['translation_v2_report']['approved_for_final_tts'],0)
        self.assertEqual(derived['translation_v2_metadata']['translation_execution'],'NOT_RUN')

    def test_generic_domain_scope_and_no_glossary_leak(self):
        marine=enrich_master(demo_master())
        pottery=enrich_master(demo_master('Ceramic Glazing','course_ceramic','Kiln firing'))
        self.assertEqual(marine['terminology_v2']['domain_analysis']['domain_hint'],'Marine ecosystems')
        self.assertEqual(pottery['terminology_v2']['domain_analysis']['domain_hint'],'Kiln firing')
        self.assertFalse(marine['terminology_v2']['approved'])
        self.assertTrue(all(t['preferred_ar'] is None for t in marine['terminology_v2']['candidates']))
        self.assertEqual(marine['terminology_v2']['unverified_existing_terms'][0]['import_notice'],'not_approved_for_this_course')
        self.assertNotIn('neural network',[x['source_term'] for x in marine['terminology_v2']['approved']])

    def test_approved_glossary_is_course_specific(self):
        data=demo_master()
        data['terminology'].append({'source_term':'coral reef','preferred_ar':'الشعاب المرجانية',
                                    'review_status':'approved','course_id':'course_marine'})
        result=enrich_master(data)
        self.assertEqual(len(result['terminology_v2']['approved']),1)

    def test_context_neighbors_reference_ids(self):
        derived=enrich_master(demo_master())
        links=derived['translation_v2_metadata']['context_links']
        self.assertEqual(links['tu_0002']['previous_unit_ids'],['tu_0001'])
        self.assertEqual(links['tu_0002']['next_unit_ids'],['tu_0003'])
        self.assertEqual(build_context_package(derived,'tu_0002',links=links)['previous'][0]['id'],'tu_0001')

    def test_builder_word_coverage_and_join(self):
        sections=[seg('We calculate a', ['We','calculate','a'],0),
                  seg('gradient step.', ['gradient','step.'],2),
                  seg('Then update parameters.', ['Then','update','parameters.'],4)]
        original=copy.deepcopy(sections)
        out=build_units_from_whisperx(sections)
        self.assertEqual(sections,original)
        self.assertEqual(out['builder_report']['source_word_count'],8)
        self.assertTrue(out['builder_report']['source_word_coverage_verified'])
        self.assertEqual(out['translation_units'][0]['source']['text'],'We calculate a gradient step.')
        refs=[(w['segment_index'],w['word_index']) for u in out['translation_units'] for w in u['source_refs']]
        self.assertEqual(refs,[(0,0),(0,1),(0,2),(1,0),(1,1),(2,0),(2,1),(2,2)])

    def test_builder_no_fabricated_times(self):
        x=seg('Bad alignment', ['Bad','alignment'])
        del x['words'][0]['start']
        with self.assertRaises(ValueError):build_units_from_whisperx([x])

    def test_mechanical_qa_flags_numbers_and_code(self):
        qa=mechanical_qa('Set limit to 512 MB. Use `max_memory`.', 'اضبط الحد على 500 ميغابايت.')
        self.assertIn('numeric_fidelity_needs_review',qa['issues'])
        self.assertIn('code_or_identifier_needs_review',qa['issues'])
        self.assertIsNone(qa['checks']['meaning_preserved'])
        self.assertTrue(qa['requires_review'])

    def test_terminology_context_plans_are_non_approved(self):
        derived=enrich_master(demo_master())
        gloss=derived['terminology_v2']
        plans=validation_plans(gloss,derived['translation_v2_metadata']['course'])
        self.assertEqual(plans['status'],'plans_only_NO_LLM_CALLS')
        self.assertTrue(plans['plans'])
        with self.assertRaises(PermissionError):
            propose_glossary_entries(gloss,derived['translation_v2_metadata']['course'],lambda *x: {})

    def test_no_false_semantic_approval(self):
        qa=mechanical_qa('Open the settings.', 'افتح الإعدادات.')
        self.assertIsNone(qa['checks']['meaning_preserved'])
        self.assertEqual(qa['status'],'needs_review')

    def test_translation_is_disabled_and_plans_are_offline(self):
        prepared=enrich_master(demo_master())
        plan=build_all_pass_plans(prepared)
        self.assertEqual(len(plan['plans']),3)
        self.assertEqual(len(plan['plans'][0]['passes']),4)
        def bogus_backend(*a,**kw):raise AssertionError('Unexpected model call')
        with self.assertRaises(PermissionError):generate_draft(prepared,'tu_0001',bogus_backend)

    def test_tts_plan_makes_no_network_calls(self):
        prepared=enrich_master(demo_master())
        p=run_edge_previews(prepared,'/unused')
        self.assertEqual(p['status'],'plan_only_NO_TTS_CALLS')
        self.assertEqual(len(p['plan']),3)
        with self.assertRaises(PermissionError):run_edge_previews(prepared,'/unused',execute=True)

    def test_missing_metadata_does_not_invent_domain(self):
        base=demo_master('Course pilot','pilot_course','')
        base['course'].pop('topic',None)
        out=enrich_master(base)
        self.assertEqual(out['terminology_v2']['domain_analysis']['status'],'text_derived_needs_review')
        self.assertTrue(out['translation_v2_metadata']['course']['id'].startswith('course_'))

    def test_prepare_saves_new_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'master.json'; dest=Path(temp)/'master_v2_derived.json'
            source.write_text(json.dumps(demo_master()),encoding='utf-8')
            hash_before=sha256_file(source)
            prepare_project(source,dest)
            self.assertEqual(sha256_file(source),hash_before)
            self.assertEqual(read_json(dest)['translation_v2_metadata']['source_master_sha256'],hash_before)
            with self.assertRaises(FileExistsError):prepare_project(source,dest)

if __name__ == '__main__':unittest.main()
