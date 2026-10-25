"""Pure offline regression suite for generic course memory and source timeline."""
import copy
import json
import unittest
from pathlib import Path
from arabic_dubbing_v2.common import fingerprint,sha256_file
from arabic_dubbing_v2.workflow import enrich_master
from arabic_dubbing_v2.course_glossary_v27 import record_human_approval
from arabic_dubbing_v2.course_pipeline_v28 import (build_course_registry, terminology_review_queue,
    build_glossary_memory, cross_lesson_consistency, build_pipeline_snapshot)
from arabic_dubbing_v2.audio_timeline_v28 import audit_existing_tts_manifest
from test_pipeline import demo_master


class GenericPipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw=demo_master();self.a=enrich_master(self.raw)
        raw2=demo_master()
        raw2['translation_units'][0]['source']['text']='Coral reef ecosystems contain many organisms.'
        raw2['translation_units'][0]['translation']['natural_ar']='تحتوي الشعاب المرجانية على كائنات كثيرة.'
        raw2['translation_units'][1]['source']['text']='Coral reef ecosystems rely on clean water.'
        raw2['translation_units'][1]['translation']['natural_ar']='تعتمد مجتمعات البحار على المياه النظيفة.'
        self.b=enrich_master(raw2)
        self.lessons=[('lesson_one',self.a),('lesson_two',self.b)]
        self.canonical={'id':'course_marine','title':'Marine Biology', 'confirmed_by':'human_course_owner'}
        self.registry=build_course_registry(self.lessons,canonical_course=self.canonical)
        self.approval=record_human_approval(self.a,{'course_id':'course_marine','source_term':'coral reef',
            'sense':'reef ecosystem','preferred_ar':'الشعاب المرجانية','keep_english':False},
            approval={'approved':True,'reviewer':'human_course_owner','decision_id':'review-001',
            'approved_at':'2026-10-25T09:00:00Z','confirmed_course_id':'course_marine'})

    def test_unconfirmed_identity_does_not_share_memory(self):
        reg=build_course_registry(self.lessons)
        self.assertFalse(reg['course_identity']['cross_lesson_reuse_permitted'])
        self.assertEqual(build_glossary_memory(reg,self.lessons)['version'],0)
        with self.assertRaises(PermissionError):build_glossary_memory(reg,self.lessons,human_approved=[self.approval])

    def test_canonical_requires_title_and_human_confirmation(self):
        for attrs in ({'id':'pilot_course','title':'Marine Biology','confirmed_by':'user'},
                      {'id':'ok','title':'Course pilot','confirmed_by':'user'},
                      {'id':'ok','title':'Marine Biology'}):
            with self.assertRaises((ValueError,PermissionError)):
                build_course_registry(self.lessons,canonical_course=attrs)

    def test_namespace_prevents_tu_collisions_between_lessons(self):
        self.assertEqual(self.registry['lesson_count'],2)
        self.assertEqual(self.registry['unit_count'],6)
        self.assertNotEqual(self.registry['lessons'][0]['scoped_unit_ids'][0],self.registry['lessons'][1]['scoped_unit_ids'][0])

    def test_registry_rejects_duplicate_lessons_or_duplicate_tu(self):
        with self.assertRaises(ValueError):build_course_registry([('lesson_one',self.a),('lesson_one',self.b)])
        corrupted=copy.deepcopy(self.a);corrupted['translation_units'][1]['id']='tu_0001'
        with self.assertRaises(ValueError):build_course_registry([('duplicate',corrupted)])

    def test_extracts_from_both_lessons_with_evidence_and_no_arabic_invention(self):
        q=terminology_review_queue(self.registry,self.lessons)
        reef=next(x for x in q['candidates'] if x['source_term']=='coral reef')
        self.assertEqual(set(reef['observed_in_lessons']),{'lesson_one','lesson_two'})
        self.assertIsNone(reef['preferred_ar'])
        self.assertEqual(reef['context_validation'],'not_performed')
        self.assertIn('lesson_two/tu_0001',[x['scoped_unit_id'] for e in reef['evidence'] for x in e['source_refs']])

    def test_detects_source_fingerprint_change(self):
        broken=copy.deepcopy(self.b)
        broken['translation_v2_metadata']['source_master_data_fingerprint']='changed_unsafe_fingerprint_123456'
        with self.assertRaises(ValueError):terminology_review_queue(self.registry,[('lesson_one',self.a),('lesson_two',broken)])

    def test_approved_memory_is_audited_and_never_automatic(self):
        snapshot=build_pipeline_snapshot(self.lessons,canonical_course=self.canonical,human_approved=[self.approval])
        memory=snapshot['glossary_memory']
        self.assertEqual(memory['version'],1)
        self.assertEqual(memory['audit'][0]['decision_id'],'review-001')
        self.assertEqual(len(memory['terms']),1)
        self.assertFalse(snapshot['qa_gate']['ready_for_final_dub'])
        self.assertEqual(snapshot['tts_calls'],0)

    def test_duplicate_conflicting_or_unaudited_approvals_rejected(self):
        double=copy.deepcopy(self.approval)
        double['approval']['decision_id']='review-002'
        double['preferred_ar']='حيود المرجان'
        with self.assertRaises(ValueError):build_glossary_memory(self.registry,self.lessons,human_approved=[self.approval,double])
        with self.assertRaises(ValueError):build_glossary_memory(self.registry,self.lessons,human_approved=[self.approval,self.approval])
        forged=copy.deepcopy(self.approval);forged.pop('approval')
        with self.assertRaises(PermissionError):build_glossary_memory(self.registry,self.lessons,human_approved=[forged])

    def test_cross_course_approval_rejected(self):
        bad=copy.deepcopy(self.approval);bad['course_id']='course_mathematics'
        with self.assertRaises(ValueError):build_glossary_memory(self.registry,self.lessons,human_approved=[bad])

    def test_versioned_persistence_does_not_reapprove(self):
        prev=build_glossary_memory(self.registry,self.lessons,human_approved=[self.approval])
        current=build_glossary_memory(self.registry,self.lessons,previous_memory=prev)
        self.assertEqual(current['version'],1)
        self.assertEqual(len(current['terms']),1)

    def test_cross_lesson_terminology_only_warns_on_mismatch(self):
        m=build_glossary_memory(self.registry,self.lessons,human_approved=[self.approval])
        qa=cross_lesson_consistency(self.registry,self.lessons,m)
        self.assertEqual(qa['pending_review_count'],1)
        self.assertEqual(len(qa['findings']),4)
        self.assertFalse(qa['final_qa_approved'])

    def test_source_and_timing_unchanged(self):
        source_before=fingerprint(self.raw)
        a_before=fingerprint(self.a);b_before=fingerprint(self.b)
        _=build_pipeline_snapshot(self.lessons,canonical_course=self.canonical)
        self.assertEqual(fingerprint(self.raw),source_before)
        self.assertEqual(fingerprint(self.a),a_before)
        self.assertEqual(fingerprint(self.b),b_before)

    def test_two_senses_of_the_same_surface_are_ambiguous_not_false_fail(self):
        two=copy.deepcopy(self.approval)
        two['sense']='named reef resort in the lesson'
        two['preferred_ar']='منتجع كورال ريف'
        two['approval']['decision_id']='review-009'
        # This checks ambiguity handling, not whether the invented sense is valid.
        m=build_glossary_memory(self.registry,self.lessons,human_approved=[self.approval,two])
        qa=cross_lesson_consistency(self.registry,self.lessons,m)
        self.assertEqual(len(qa['findings']),4)
        self.assertTrue(all(f['status']=='ambiguous_sense_needs_review' for f in qa['findings']))
        self.assertEqual(qa['pending_review_count'],4)

    def test_canonical_cannot_override_another_known_course_id(self):
        wrong=dict(self.canonical,id='course_mathematics')
        with self.assertRaises(ValueError):build_course_registry(self.lessons,canonical_course=wrong)

    def test_incremental_memory_can_reuse_audited_term_absent_in_new_lesson(self):
        m=build_glossary_memory(self.registry,self.lessons,human_approved=[self.approval])
        another=demo_master()
        for u in another['translation_units']:
            u['source']['text']='This lesson discusses ocean water and species.'
        c=enrich_master(another)
        new_lessons=[('lesson_three',c)]
        reg=build_course_registry(new_lessons,canonical_course=self.canonical)
        reused=build_glossary_memory(reg,new_lessons,previous_memory=m)
        self.assertEqual(reused['version'],1)
        self.assertEqual(len(reused['terms']),1)
        self.assertTrue(reused['audit'][0]['evidence_unit_ids'])
        self.assertFalse(cross_lesson_consistency(reg,new_lessons,reused)['findings'])

    def test_real_v25_source_and_tts_history_only_read(self):
        source=Path('/mnt/data/translation_master_v2_5_editorial_candidate.json')
        pilot=Path('/mnt/data/tts_pilot_3_manifest_qa35lite_20261012T161432Z.json')
        if not source.is_file() or not pilot.is_file():self.skipTest('Pilot files unavailable')
        hash_before=sha256_file(source)
        prepared=enrich_master(json.loads(source.read_text(encoding='utf-8')),
                               source_filename=source.name,source_sha256=hash_before)
        reg=build_course_registry([('pilot_lesson',prepared)])
        q=terminology_review_queue(reg,[('pilot_lesson',prepared)])
        self.assertEqual(reg['unit_count'],17)
        self.assertTrue(q['candidate_count']>0)
        self.assertFalse(reg['course_identity']['cross_lesson_reuse_permitted'])
        timeline=audit_existing_tts_manifest(prepared,json.loads(pilot.read_text(encoding='utf-8')))
        self.assertEqual(timeline['measured_units'],3)
        self.assertTrue(all(not r['audio_approved'] for r in timeline['records']))
        self.assertEqual(sha256_file(source),hash_before)

    def test_audio_timeline_rejects_wrong_master(self):
        with self.assertRaises(ValueError):
            audit_existing_tts_manifest(enrich_master(self.raw,source_sha256='expected'),{'master_sha256':'wrong','units':{}})

    def test_audio_timeline_flags_overrun_and_stays_unapproved(self):
        rec={'master_sha256':None,'units':{'tu_0001':{'unit_id':'tu_0001','actual_seconds':12.0,
             'tts_approved':False,'audio_listened_and_verified':False}}}
        out=audit_existing_tts_manifest(self.a,rec)
        self.assertEqual(out['measured_units'],1)
        self.assertEqual(out['records'][0]['timing_status'],'major_overrun_needs_edit_or_retake')
        self.assertTrue(out['records'][0]['overlaps_next_source_if_placed_at_start'])
        self.assertFalse(out['final_sync_approved'])

if __name__=='__main__':unittest.main()