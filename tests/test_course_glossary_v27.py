"""Generic educational tests for Figma 02.27; no network/model calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from arabic_dubbing_v2.common import fingerprint, save_new_json
from arabic_dubbing_v2.workflow import enrich_master
from arabic_dubbing_v2.course_glossary_v27 import (identity_audit, source_evidence,
    validate_suggestion, record_human_approval, consistency_audit,
    term_change_impact, build_course_glossary_report)
from test_pipeline import demo_master


class GeneralCourseGlossaryTests(unittest.TestCase):
    def setUp(self):
        self.marine = enrich_master(demo_master())
        self.source = demo_master()
        self.prep = enrich_master(self.source)
        self.cid = self.prep['translation_v2_metadata']['course']['id']
        self.proposal={'course_id':self.cid,'source_term':'coral reef',
                       'sense':'reef ecosystem in a marine biology lesson',
                       'preferred_ar':'الشعاب المرجانية','keep_english':False}
        self.approval={'approved':True,'reviewer':'example human reviewer',
                       'decision_id':'demo-1','approved_at':'2026-10-25T00:00:00Z',
                       'confirmed_course_id':self.cid}

    def test_proposed_terms_are_unapproved_without_network(self):
        report=build_course_glossary_report(self.prep)
        self.assertEqual(report['network_calls'],0)
        self.assertFalse(report['approved_memory'])
        self.assertTrue(report['candidates'])
        self.assertTrue(all(x['review_status']=='candidate' and x.get('preferred_ar') is None for x in report['candidates']))
        self.assertEqual(report['gates']['final_dub_ar_approved'],False)

    def test_placeholder_course_is_not_invented_as_verified(self):
        raw=demo_master('Course pilot','pilot_course','')
        raw['course'].pop('topic')
        prepared=enrich_master(raw)
        audit=identity_audit(prepared)
        self.assertEqual(audit['identity_status'],'source_placeholder_not_identified')
        self.assertEqual(audit['scope'],'source_instance_only')
        self.assertIsNone(audit['provided_title'])
        self.assertFalse(audit['cross_lesson_reuse_permitted'])

    def test_literal_evidence_not_substring(self):
        self.assertTrue(source_evidence(self.prep,'coral reef'))
        self.assertFalse(source_evidence(self.prep,'oral reef'))
        self.assertFalse(source_evidence(self.prep,'unmentioned'))

    def test_cross_course_rejected_in_contextual_validation(self):
        q=dict(self.proposal,course_id='course_cooking')
        with self.assertRaises(ValueError):validate_suggestion(self.prep,q)

    def test_contextual_proposal_never_auto_approved(self):
        x=validate_suggestion(self.prep,self.proposal)
        self.assertEqual(x['review_status'],'validated_pending_human_approval')
        self.assertFalse(x.get('approval'))
        self.assertEqual(len(x['source_refs']),2)
        self.assertTrue(x['source_refs'][0]['source_refs'])

    def test_incomplete_sense_remains_pending(self):
        x=validate_suggestion(self.prep,dict(self.proposal,sense=''))
        self.assertEqual(x['review_status'],'needs_review')
        with self.assertRaises(ValueError):record_human_approval(self.prep,dict(self.proposal,sense=''),approval=self.approval)

    def test_human_approval_requires_explicit_audit(self):
        with self.assertRaises(PermissionError):record_human_approval(self.prep,self.proposal,approval={})
        with self.assertRaises(PermissionError):record_human_approval(self.prep,self.proposal,approval={'approved':True})
        out=record_human_approval(self.prep,self.proposal,approval=self.approval)
        self.assertEqual(out['review_status'],'approved')
        self.assertEqual(out['approval']['decision_id'],'demo-1')

    def test_consistency_matches_and_flags_variants_without_approval(self):
        approved=record_human_approval(self.prep,self.proposal,approval=self.approval)
        report=build_course_glossary_report(self.prep,human_approved=[approved])
        self.assertEqual(len(report['consistency_qa']['findings']),2)
        self.assertEqual(report['consistency_qa']['potential_mismatch_count'],0)
        self.assertTrue(all(x['semantic_equivalence_verified'] is False for x in report['consistency_qa']['findings']))
        changed=copy.deepcopy(self.prep)
        changed['translation_units'][0]['translation']['natural_ar']='هذه مجتمعات بحرية متنوعة.'
        self.assertEqual(consistency_audit(changed,[approved])['potential_mismatch_count'],1)

    def test_new_rendering_triggers_reqa_for_affected_units(self):
        prev=record_human_approval(self.prep,self.proposal,approval=self.approval)
        new=dict(prev,preferred_ar='الحيود المرجانية')
        impact=term_change_impact(self.prep,[new],[prev])
        self.assertEqual(impact[0]['affected_unit_ids'],['tu_0001','tu_0002'])
        self.assertEqual(impact[0]['action'],'requires_reqa_review')

    def test_fake_new_approval_never_accepted_directly(self):
        fake=dict(self.proposal,review_status="approved")
        with self.assertRaises(PermissionError):build_course_glossary_report(self.prep,human_approved=[fake])

    def test_approval_timestamp_must_be_timezone_aware(self):
        with self.assertRaises(ValueError):record_human_approval(self.prep,self.proposal,approval=dict(self.approval,approved_at="yesterday"))

    def test_legacy_scoped_approved_records_supported(self):
        source=demo_master()
        source["terminology"].append({"source_term":"coral reef","preferred_ar":"الشعاب المرجانية",
            "status":"approved","scope_course_id":"course_marine"})
        report=build_course_glossary_report(enrich_master(source))
        self.assertEqual(len(report["approved_memory"]),1)
        self.assertEqual(report["approved_memory"][0]["review_status"],"approved")

    def test_cross_course_glossary_memory_does_not_transfer(self):
        approved=record_human_approval(self.prep,self.proposal,approval=self.approval)
        other=enrich_master(demo_master('Cooking 101','course_cooking','Baking bread'))
        with self.assertRaises(ValueError):build_course_glossary_report(other,human_approved=[approved])

    def test_candidate_terms_drop_sentence_final_period(self):
        from arabic_dubbing_v2.terminology import extract_candidates
        candidates = extract_candidates(self.prep["translation_units"], self.prep["translation_v2_metadata"]["course"])
        self.assertFalse(any(x["source_term"].endswith(".") for x in candidates))

    def test_work_does_not_mutate_original_or_word_refs(self):
        before=fingerprint(self.source)
        report=build_course_glossary_report(enrich_master(self.source))
        self.assertEqual(fingerprint(self.source),before)
        self.assertEqual(len(report['candidates'])>0,True)
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'report.json';save_new_json(a,report)
            with self.assertRaises(FileExistsError):save_new_json(a,report)

    def test_no_fixed_ai_course_glossary_for_mathematics_and_cooking(self):
        for name, cid, topic in [('Calculus','math_101','Derivatives'),('Cooking','food_101','Baking')]:
            raw=demo_master(name,cid,topic)
            report=build_course_glossary_report(enrich_master(raw))
            self.assertEqual(report['domain_detection']['domain_hint'],topic)
            self.assertFalse(report['approved_memory'])
            self.assertNotIn('neural network',[x['source_term'] for x in report['candidates']])

if __name__=='__main__':unittest.main()