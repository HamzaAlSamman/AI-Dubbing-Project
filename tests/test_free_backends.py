"""Deterministic, entirely offline safety tests; no provider/network/TTS calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from arabic_dubbing_v2.workflow import enrich_master
from arabic_dubbing_v2.free_backends import (TEXT_MODE_CHOICES, TTS_ENGINE_CHOICES, text_jobs,
                                               preview_jobs, GeminiJSONBackend)
from test_pipeline import demo_master

class FreeBackendsTests(unittest.TestCase):
    def setUp(self):
        self.prepared=enrich_master(demo_master())

    def test_mode_choices_and_reuse_never_call_backend(self):
        self.assertIn('review_existing',TEXT_MODE_CHOICES)
        self.assertIn('gemini_flash_lite',TTS_ENGINE_CHOICES)
        self.assertEqual(text_jobs(self.prepared,'/unused',mode='reuse_existing')['request_count'],0)

    def test_text_plan_does_not_call_backend(self):
        def forbidden(*args,**kw):raise AssertionError('Unexpected API call')
        r=text_jobs(self.prepared,'/unused',mode='review_existing',unit_ids=['tu_0001'],backend=forbidden)
        self.assertEqual(r['status'],'plan_only_NO_LLM_CALLS')
        self.assertEqual(r['planned_requests'],1)

    def test_review_requires_two_flags_before_call(self):
        calls=[]
        def mock(*a): calls.append(a); return {'issues':[], 'needs_review':True, 'rationale':'ok'}
        with self.assertRaises(PermissionError):
            text_jobs(self.prepared,'/unused',mode='review_existing',unit_ids=['tu_0001'],execute=True,backend=mock)
        with self.assertRaises(ValueError):
            text_jobs(self.prepared,'/unused',mode='review_existing',unit_ids=['tu_0001'],execute=True,allow_online=True,backend=mock,max_requests=0)
        self.assertEqual(calls,[])

    def test_review_with_mock_only_saves_unapproved(self):
        before=json.dumps(self.prepared,sort_keys=True,ensure_ascii=False)
        called=[]
        def mock(*a): called.append(a); return {'issues':['mock_warning'], 'needs_review':True, 'rationale':'unreviewed'}
        with tempfile.TemporaryDirectory() as d:
            r=text_jobs(self.prepared,d,mode='review_existing',unit_ids=['tu_0002'],
                        execute=True,allow_online=True,backend=mock,max_requests=1)
            self.assertEqual(len(called),1)
            doc=json.loads(Path(r['results'][0]['path']).read_text())
            self.assertIsNone(doc['final_dub_ar'])
            self.assertFalse(doc['human_approved'])
            with self.assertRaises(FileExistsError):
                text_jobs(self.prepared,d,mode='review_existing',unit_ids=['tu_0002'],
                          execute=True,allow_online=True,backend=mock,max_requests=1)
            self.assertEqual(len(called),1)
        self.assertEqual(json.dumps(self.prepared,sort_keys=True,ensure_ascii=False),before)

    def test_regeneration_is_four_request_plan_only(self):
        r=text_jobs(self.prepared,'/unused',mode='regenerate_selected',unit_ids=['tu_0003'])
        self.assertEqual(r['planned_requests'],4)
        with self.assertRaises(ValueError):
            text_jobs(self.prepared,'/unused',mode='regenerate_selected',unit_ids=['tu_0003'],execute=True,
                      allow_online=True,backend=lambda *a: {},max_requests=3)

    def test_tts_gate_all_engines(self):
        for engine in TTS_ENGINE_CHOICES:
            r=preview_jobs(self.prepared,'/unused',engine=engine,unit_ids=['tu_0001'])
            self.assertEqual(r['status'],'plan_only_NO_TTS_CALLS')
        with self.assertRaises(PermissionError):
            preview_jobs(self.prepared,'/unused',engine='mms_local',unit_ids=['tu_0001'],execute=True)
        with self.assertRaises(PermissionError):
            preview_jobs(self.prepared,'/unused',engine='gemini_flash_lite',unit_ids=['tu_0001'],execute=True)
        with self.assertRaises(ValueError):
            preview_jobs(self.prepared,'/unused',engine='mms_local',unit_ids=['tu_0001','tu_0002'],execute=True,allow_local=True,max_units=1)

    def test_tts_mock_local_never_overwrites(self):
        def fake_render(text,dest,cache=None):
            import wave
            with wave.open(str(dest),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(b'\x00\x00'*240)
            return ('fake','fake')
        with tempfile.TemporaryDirectory() as d:
            with patch('arabic_dubbing_v2.free_backends._render_mms',fake_render):
                r=preview_jobs(self.prepared,d,engine='mms_local',unit_ids=['tu_0001'],execute=True,allow_local=True)
                self.assertEqual(r['status'],'PREVIEWS_CREATED_NOT_APPROVED')
                self.assertFalse(r['results'][0]['audio_approved'])
                with self.assertRaises(FileExistsError):
                    preview_jobs(self.prepared,d,engine='mms_local',unit_ids=['tu_0001'],execute=True,allow_local=True)

    def test_mock_gemini_backend_json(self):
        from types import SimpleNamespace
        fake_client=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw:SimpleNamespace(text='{"issues":[],"needs_review":true}')))
        backend=GeminiJSONBackend(fake_client)
        r=backend('instruction',{'source_text':'Hello'},'qa_v1')
        self.assertIsInstance(r['issues'],list)

if __name__=='__main__':unittest.main()
