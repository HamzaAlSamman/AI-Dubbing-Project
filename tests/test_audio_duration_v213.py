import math
import tempfile
import unittest
import wave
from pathlib import Path
from array import array
from arabic_dubbing_v2.audio_duration_v213 import plan_fit, fit_existing_wav, fit_manifest_unit, _duration, _sha


def sample(path,seconds,rate=24000):
    nums=array('h',(int(13000*math.sin(2*math.pi*430*i/rate)) for i in range(round(seconds*rate))))
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes(nums.tobytes())

class DurationFitTests(unittest.TestCase):
    def test_examples_from_real_pilot_manifest(self):
        self.assertTrue(plan_fit(1.322,2.64)['status'].startswith('BLOCK'))
        self.assertEqual(plan_fit(11.487,13.36)['status'],'TIME_STRETCH_PRESERVE_PITCH')
        self.assertEqual(plan_fit(4.881,5.24)['status'],'TIME_STRETCH_PRESERVE_PITCH')

    def test_large_gap_blocks(self):
        self.assertEqual(plan_fit(8.0,2.0)['status'],'BLOCK_EXCESSIVE_SILENCE_NEEDS_SLOWER_RETAKE')

    def test_reject_invalid_duration(self):
        for a,b in [(0,1),(-1,1),(1,float('nan')),(1,0)]:
            with self.assertRaises(ValueError):plan_fit(a,b)

    def test_fit_slight_overrun_preserves_original_and_exact_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp=Path(tmp)/'raw.wav'; out=Path(tmp)/'fitted.wav';sample(inp,5.24)
            raw_before=inp.read_bytes()
            result=fit_existing_wav(inp,out,4.881,source_start=112.256,next_source_start=117.6)
            self.assertTrue(result['rendered'],result)
            self.assertLessEqual(abs(_duration(out)-4.881),1/24000)
            self.assertEqual(raw_before,inp.read_bytes())
            self.assertTrue(out.with_suffix('.json').exists())
            with self.assertRaises(FileExistsError):fit_existing_wav(inp,out,4.881)

    def test_pad_short_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp=Path(tmp)/'raw.wav';out=Path(tmp)/'short.wav';sample(inp,4.6)
            result=fit_existing_wav(inp,out,4.88)
            self.assertTrue(result['rendered']);self.assertGreater(result['padding_seconds'],0)
            self.assertLess(abs(_duration(out)-4.88),1/24000)

    def test_does_not_render_excessive_compression(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp=Path(tmp)/'raw.wav';out=Path(tmp)/'fit.wav';sample(inp,2.64)
            res=fit_existing_wav(inp,out,1.322)
            self.assertFalse(res['rendered'])
            self.assertFalse(out.exists())

    def test_manifest_linkage_and_render_is_off_by_default(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);wav=base/'tu_0001_original.wav';sample(wav,5.24)
            master=base/'source_master.json';master.write_text(json.dumps({
               'translation_units':[{'id':'tu_0001','source':{'text':'lesson'},
                'timing':{'source_start':0.,'source_end':4.881}}]}))
            sha=_sha(master)
            manifest=base/'manifest.json';manifest.write_text(json.dumps({
               'master_sha256':sha,'units':{'tu_0001':{'unit_id':'tu_0001',
               'filename':wav.name,'audio_sha256':_sha(wav),'target_seconds':4.881}}}))
            before=wav.read_bytes()
            plan=fit_manifest_unit(master,manifest,base,'tu_0001',base/'out')
            self.assertTrue(plan['wav_sha256_verified'])
            self.assertTrue(plan['output_not_generated'])
            self.assertFalse((base/'out').exists())
            rendered=fit_manifest_unit(master,manifest,base,'tu_0001',base/'out',render=True)
            self.assertTrue(rendered['rendered']);self.assertTrue(Path(rendered['source_binding_report']).exists())
            self.assertEqual(before,wav.read_bytes())

    def test_manifest_source_hash_mismatch_refuses(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);wav=base/'tu.wav';sample(wav,2.0)
            master=base/'master.json';master.write_text(json.dumps({'translation_units':[
                 {'id':'tu_1','timing':{'source_start':1.0,'source_end':3.0}}]}))
            manifest=base/'tts.json';manifest.write_text(json.dumps({'master_sha256':'wrong',
              'units':{'tu_1':{'unit_id':'tu_1','filename':'tu.wav','audio_sha256':_sha(wav)}}}))
            with self.assertRaises(ValueError):fit_manifest_unit(master,manifest,base,'tu_1',base/'out')

    def test_refuses_source_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp=Path(tmp)/'raw.wav';out=Path(tmp)/'fit.wav';sample(inp,4.0)
            with self.assertRaises(ValueError):fit_existing_wav(inp,out,4.0,source_start=3,next_source_start=6)

if __name__=='__main__':unittest.main()