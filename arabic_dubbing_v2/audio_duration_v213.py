"""Offline, fail-closed duration fitting for pre-existing synthesized WAV clips.

V2.13 operates on an isolated source clip, never calls a model or TTS provider.
It preserves the original WAV and Master/WhisperX data. A small pitch-preserving
Rubber Band tempo change is permitted; large overrun or huge silence is blocked.
Outputs a newly-named PCM WAV and immutable JSON metadata, no final-dub approval.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

MAX_TEMPO = 1.20
MAX_PADDING_RATIO = 0.35
MAX_PADDING_SECONDS = 2.0
MAX_RETRIES = 3


def _duration(wav_path):
    with wave.open(str(wav_path), 'rb') as f:
        if f.getnframes() <= 0 or f.getframerate() <= 0:
            raise ValueError('WAV has zero frames or invalid sample rate')
        return f.getnframes() / f.getframerate()


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def plan_fit(source_seconds, generated_seconds, *, max_tempo=MAX_TEMPO,
             max_padding_ratio=MAX_PADDING_RATIO, max_padding_seconds=MAX_PADDING_SECONDS):
    """Produce a conservative decision without touching sound or text."""
    for x in (source_seconds, generated_seconds, max_tempo, max_padding_ratio, max_padding_seconds):
        if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
            raise ValueError('Finite numeric arguments required')
    if source_seconds <= 0 or generated_seconds <= 0 or not 1 < max_tempo <= 1.5:
        raise ValueError('Invalid duration/tempo bounds')
    if max_padding_ratio < 0 or max_padding_seconds < 0:
        raise ValueError('Invalid padding bounds')
    ratio = generated_seconds/source_seconds
    if ratio > max_tempo:
        action = 'BLOCK_NEEDS_SHORTER_TRANSLATION_AND_RETAKE'
    elif ratio > 1.001:
        action = 'TIME_STRETCH_PRESERVE_PITCH'
    elif source_seconds-generated_seconds > min(max_padding_seconds, source_seconds*max_padding_ratio):
        action = 'BLOCK_EXCESSIVE_SILENCE_NEEDS_SLOWER_RETAKE'
    else:
        action = 'PAD_END_SILENCE_ONLY'
    return {'status': action, 'target_seconds': round(source_seconds, 6),
            'actual_seconds': round(generated_seconds, 6),
            'required_tempo': round(ratio, 6),
            'source_is_immutable': True,
            'notes': 'Targeting source speech span, not source video speed.'}


def _rubberband(input_path, temp_out, tempo, rate, channels):
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg required; on Colab install ffmpeg with rubberband filter')
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
           '-i', str(input_path), '-vn', '-af', f'rubberband=tempo={tempo:.9f}',
           '-ar', str(rate), '-ac', str(channels), '-c:a', 'pcm_s16le', str(temp_out)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def fit_existing_wav(input_wav, output_wav, target_seconds, *, source_start=None,
                     next_source_start=None, max_tempo=MAX_TEMPO):
    """Render a new exact-duration WAV. Never overwrite source or an existing output.

    For a timing-risk clip we return a blocked plan; the caller can request a
    shorter script and NEW synthesis in a separately authorized workflow.
    """
    src = Path(input_wav); dest = Path(output_wav)
    if not src.is_file() or src.stat().st_size <= 44:
        raise FileNotFoundError('Existing nonempty WAV required')
    if dest.exists() or dest.with_suffix('.json').exists():
        raise FileExistsError('Output already exists; no overwrite')
    if src.resolve() == dest.resolve():
        raise ValueError('Output must not be the input file')
    if source_start is not None:
        if not isinstance(source_start, (float, int)) or not math.isfinite(source_start) or source_start < 0:
            raise ValueError('Invalid timeline source_start')
    if next_source_start is not None:
        if source_start is None or not math.isfinite(next_source_start) or next_source_start <= source_start:
            raise ValueError('Invalid next start')
        if target_seconds > next_source_start - source_start + 0.00001:
            raise ValueError('Source time span overlaps next source start; refuse render')
    duration = _duration(src)
    proposal = plan_fit(target_seconds, duration, max_tempo=max_tempo)
    if proposal['status'].startswith('BLOCK'):
        return {'rendered': False, **proposal, 'input_sha256': _sha(src)}
    with wave.open(str(src), 'rb') as wav:
        rate = wav.getframerate(); channels = wav.getnchannels()
        if channels not in (1, 2):
            raise ValueError('Only mono/stereo WAV are supported')
    target_frames=round(float(target_seconds)*rate)
    if target_frames < 1:
        raise ValueError('Requested duration below one sample')
    dest.parent.mkdir(parents=True, exist_ok=True)
    original_hash = _sha(src)
    with tempfile.TemporaryDirectory(prefix='dub_fit_') as d:
        tmp=Path(d)/'stretched.wav'
        in_audio=src
        applied=1.0
        if proposal['status']=='TIME_STRETCH_PRESERVE_PITCH':
            tempo=duration/target_seconds
            # Aim a fraction under target so only silence, never spoken samples,
            # need to be added. Recalculate on any unusual filter rounding.
            for attempt in range(MAX_RETRIES):
                applied=tempo*(1.005+0.01*attempt)
                if applied>max_tempo+0.0001:
                    return {'rendered':False,**proposal,'status':'BLOCK_TOO_CLOSE_TO_MAX_TEMPO',
                            'input_sha256':original_hash}
                _rubberband(src,tmp,applied,rate,channels)
                with wave.open(str(tmp),'rb') as f:
                    frames=f.getnframes()
                if frames<=target_frames:
                    in_audio=tmp
                    break
            else:
                return {'rendered':False,**proposal,'status':'BLOCK_FILTER_OVERRUN',
                        'input_sha256':original_hash}
        # Write exactly target frames, never cut any content. Silence fills gap.
        with wave.open(str(in_audio),'rb') as f:
            if f.getsampwidth()!=2:
                # Normalize other PCM formats safely through ffmpeg without tempo.
                converted=Path(d)/'converted.wav'
                subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
                    '-i',str(in_audio),'-ar',str(rate),'-ac',str(channels),'-c:a','pcm_s16le',
                    str(converted)],check=True,capture_output=True)
                in_audio=converted
        with wave.open(str(in_audio),'rb') as f:
            existing_frames=f.getnframes()
            if existing_frames>target_frames:
                return {'rendered':False,**proposal,'status':'BLOCK_AUDIO_EXCEEDS_TARGET',
                        'input_sha256':original_hash}
            data=f.readframes(existing_frames)
            params=f.getparams()
        pad_frames=target_frames-existing_frames
        temporary_dest=Path(d)/'new.wav'
        with wave.open(str(temporary_dest),'wb') as out:
            out.setnchannels(channels)
            out.setsampwidth(params.sampwidth)
            out.setframerate(rate)
            out.writeframes(data+b'\x00'*(pad_frames*channels*params.sampwidth))
        # Cross-check source before publication: no accidental source replacement.
        if _sha(src)!=original_hash:
            raise RuntimeError('Input changed during processing')
        final_seconds=_duration(temporary_dest)
        if abs(final_seconds-target_frames/rate)>1/rate:
            raise RuntimeError('Unexpected final duration')
        manifest={'schema_version':'dubbing-duration-fit-v2.13',
                  'status':'FITTED_PREVIEW_NOT_FINAL', 'input_sha256':original_hash,
                  'input_filename':src.name,'target_seconds':float(target_seconds),
                  'original_audio_seconds':duration,'fitted_seconds':final_seconds,
                  'tempo_applied':round(applied,6), 'padding_seconds':pad_frames/rate,
                  'source_start_seconds':source_start,
                  'next_source_start_seconds':next_source_start,
                  'source_fingerprint_status':'caller_must_bind_to_source_master',
                  'audio_qa_status':'duration_only_not_intelligibility_verified',
                  'final_dub_approved':False,'engine':'FFmpeg rubberband filter',
                  'output_filename':dest.name,'output_sha256':_sha(temporary_dest)}
        meta=dest.with_suffix('.json')
        # exclusive creation; avoid overriding any other process's output.
        with dest.open('xb') as final:
            with temporary_dest.open('rb') as payload:
                shutil.copyfileobj(payload,final)
        try:
            with meta.open('x',encoding='utf-8') as f:
                json.dump(manifest,f,ensure_ascii=False,indent=2)
                f.write('\n')
        except Exception:
            dest.unlink(missing_ok=True)
            raise
    return {'rendered': True, **manifest, 'output_wav':str(dest), 'manifest_path':str(meta)}


def fit_manifest_unit(source_master_path, tts_manifest_path, audio_directory,
                      unit_id, output_directory, *, render=False):
    """Safely link a TTS WAV to a source TU and optionally render a fitted WAV.

    Requires a manifest record with a SHA256 of actual WAV content. Refuses a
    mismatched source Master hash. Defaults to planning, no file writes or API.
    """
    from .common import unit_times
    mpath=Path(source_master_path); tpath=Path(tts_manifest_path)
    master=json.loads(mpath.read_text(encoding='utf-8'))
    pilot=json.loads(tpath.read_text(encoding='utf-8'))
    source_hash=_sha(mpath)
    if pilot.get('master_sha256') and pilot['master_sha256']!=source_hash:
        raise ValueError('TTS manifest belongs to a different source Master')
    units=master.get('translation_units') or []
    matched=[u for u in units if u.get('id')==unit_id]
    if len(matched)!=1:
        raise ValueError('Unknown or duplicate source unit ID')
    recs=pilot.get('units')
    rec=recs.get(unit_id) if isinstance(recs,dict) else pilot
    if not isinstance(rec,dict) or rec.get('unit_id')!=unit_id:
        raise ValueError('No matching unit in TTS manifest')
    filename=rec.get('filename') or rec.get('audio_filename')
    digest=rec.get('audio_sha256') or rec.get('sha256')
    if not isinstance(filename,str) or Path(filename).name!=filename or not isinstance(digest,str):
        raise ValueError('TTS manifest must specify a safe filename and SHA256')
    wav=Path(audio_directory)/filename
    if _sha(wav)!=digest:
        raise ValueError('Existing TTS WAV hash differs from its manifest')
    start,end=unit_times(matched[0]); duration=end-start
    claimed=rec.get('target_seconds',rec.get('source_duration_seconds'))
    if claimed is not None and abs(float(claimed)-duration)>0.015:
        raise ValueError('TTS record target duration disagrees with source')
    next_starts=[unit_times(u)[0] for u in units if u.get('id')!=unit_id
                 and unit_times(u)[0]>start+0.00001]
    next_start=min(next_starts) if next_starts else None
    if next_start is not None and end>next_start+0.00001:
        raise ValueError('Original TU overlaps next TU: source timings need QA')
    planned=plan_fit(duration,_duration(wav))
    planned.update({'unit_id':unit_id,'input_file':wav.name,'source_master_sha256':source_hash,
                    'wav_sha256_verified':True,'output_not_generated':not render})
    if not render or planned['status'].startswith('BLOCK'):
        return planned
    outdir=Path(output_directory)
    identity=hashlib.sha256((source_hash+digest+unit_id).encode('utf-8')).hexdigest()[:16]
    output_wav=outdir/f'{unit_id}_duration_fit_{identity}.wav'
    link=outdir/f'{unit_id}_source_binding_{identity}.json'
    if link.exists():
        raise FileExistsError('Existing source-binding report; refuse overwrite')
    result=fit_existing_wav(wav,output_wav,duration,source_start=start,
                            next_source_start=next_start)
    if not result.get('rendered'):
        return {**planned,'status':result['status'],'output_not_generated':True}
    # Publish an independent source-binding report, never modify other JSON.
    binding={'schema_version':'source-audio-binding-v2.13','unit_id':unit_id,
             'source_master_sha256':source_hash,'source_start':start,'source_end':end,
             'source_refs_preserved':True,'original_wav_sha256':digest,
             'fitted_wav_sha256':result['output_sha256'],'fitted_file':output_wav.name,
             'automated_duration_check':True,'audio_intelligibility_verified':False}
    with link.open('x',encoding='utf-8') as f:
        json.dump(binding,f,ensure_ascii=False,indent=2);f.write('\n')
    return {**result,'unit_id':unit_id,'source_binding_report':str(link)}