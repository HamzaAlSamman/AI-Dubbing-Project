"""Optional Edge TTS preview: disabled unless two explicit flags are set.

Uses external online Microsoft Edge Speech service, which is free-to-try but
not an offline, guaranteed or production-licensed engine. No Google TTS.
"""
import hashlib
import json
import re
import subprocess
import shutil
import tempfile
from pathlib import Path
from .common import sha256_file, unit_times, unit_text


def select_previews(prepared, unit_ids=None):
    units=prepared['translation_units']
    byid={u['id']:u for u in units}
    if unit_ids is None:
        # all proposals are available; no hard-coded IDs for a particular lesson
        unit_ids=list(byid)
    if not unit_ids or len(set(unit_ids)) != len(unit_ids):
        raise ValueError('Select at least one unique TU ID')
    plan=[]
    for uid in unit_ids:
        if uid not in byid:raise KeyError(f'Unknown unit {uid}')
        u=byid[uid]
        t=(u.get('translation') or {})
        text=str(t.get('natural_ar') or '').strip()
        if not text:
            plan.append({'id':uid,'status':'missing_natural_ar','text':None})
        elif not re.search(r'[\u0600-\u06FF]',text):
            plan.append({'id':uid,'status':'not_arabic','text':None})
        else:
            a,b=unit_times(u)
            plan.append({'id':uid,'status':'ready_for_unapproved_preview',
                         'text':text,'source_text':unit_text(u),'source_duration_seconds':round(b-a,3)})
    return plan


def _duration(path):
    if not shutil.which('ffprobe'):return None
    p=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                      '-of','default=noprint_wrappers=1:nokey=1',str(path)],
                     capture_output=True,text=True,check=False)
    if p.returncode:return None
    try:return float(p.stdout.strip())
    except ValueError:return None


def _render_edge(text,voice,rate,dest):
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError('Install edge-tts explicitly before enabling preview: pip install edge-tts') from exc
    # save_sync works inside Colab/Jupyter where an event loop already runs.
    edge_tts.Communicate(text,voice=voice,rate=rate).save_sync(str(dest))


def run_edge_previews(prepared, output_dir, *, unit_ids=None, voice='ar-SA-HamedNeural', rate='+0%',
                      execute=False, allow_online_tts=False):
    plan=select_previews(prepared,unit_ids)
    if not execute:
        return {'status':'plan_only_NO_TTS_CALLS','plan':plan}
    if not allow_online_tts:
        raise PermissionError('An online TTS call requires allow_online_tts=True in addition to execute=True')
    invalid=[p['id'] for p in plan if p['status']!='ready_for_unapproved_preview']
    if invalid:raise ValueError(f'Missing or invalid Arabic for: {invalid}')
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    results=[]
    for item in plan:
        settings={'unit_id':item['id'],'text':item['text'],'voice':voice,'rate':rate,
                  'source_data_fingerprint':prepared['translation_v2_metadata']['source_master_data_fingerprint']}
        sign=hashlib.sha256(json.dumps(settings,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
        dest=out / f'{item["id"]}_{sign}.mp3'
        manifest=out / f'{item["id"]}_{sign}.json'
        if not dest.exists():
            # Avoid empty output artifacts when the remote service fails.
            tmp=out / f'.{item["id"]}_{sign}.partial.mp3'
            try:
                _render_edge(item['text'],voice,rate,tmp)
                if not tmp.is_file() or not tmp.stat().st_size:raise RuntimeError('TTS returned no audio')
                tmp.rename(dest)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
        seconds=_duration(dest)
        data={'status':'preview_only_not_approved','translation_approved':False,'audio_approved':False,
              'unit_id':item['id'],'tts_engine':'edge-tts (online)','source_duration_seconds':item['source_duration_seconds'],
              'generated_duration_seconds':seconds,
              'duration_difference_seconds':None if seconds is None else round(seconds-item['source_duration_seconds'],3),
              'text':item['text'],'voice':voice,'rate':rate,'sha256':sha256_file(dest),
              'filename':dest.name}
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8'))!=data:
                raise RuntimeError(f'Existing preview manifest differs: {manifest}; refusing overwrite')
        else:
            with manifest.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
        results.append(data)
    return {'status':'PREVIEWS_CREATED_NOT_APPROVED','results':results}
