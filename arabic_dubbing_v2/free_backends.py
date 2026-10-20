"""Opt-in educational translation QA and TTS pilots.

NO external call is made merely by importing this module. Existing masters,
source timing, source words, and final_dub_ar are never edited here.

The 'free' Gemini models can incur charges if the caller's Google project has
billing enabled. The user must verify Free Tier independently before enabling.
"""
import base64
import hashlib
import json
import re
import wave
from pathlib import Path

from .common import fingerprint, save_new_json, sha256_file, unit_text
from .speech import select_previews, run_edge_previews, _duration
from .translation import generate_draft

TEXT_MODE_CHOICES = ('reuse_existing', 'review_existing', 'regenerate_selected')
TTS_ENGINE_CHOICES = ('plan_only', 'edge_online', 'mms_local', 'gemini_flash_lite')


def gemini_client(api_key):
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError('GEMINI_API_KEY is required only for opted-in Gemini execution')
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError('Install google-genai first: pip install -U google-genai') from exc
    return genai.Client(api_key=api_key.strip())


class GeminiJSONBackend:
    provider_name = 'gemini_api_explicit_opt_in'

    def __init__(self, client, model='gemini-2.5-flash'):
        self.client, self.model = client, model

    def __call__(self, instruction, inputs, prompt_version):
        prompt = instruction + '\n\nStage: ' + prompt_version + '\nINPUT_JSON:\n' + json.dumps(inputs, ensure_ascii=False)
        response = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config={'response_mime_type': 'application/json', 'temperature': 0.0},
        )
        raw = response.text
        if not raw:
            raise RuntimeError('Gemini returned no JSON text')
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError('Gemini JSON must be an object')
        return data


def _unique_known_ids(prepared, unit_ids):
    known = {u['id'] for u in prepared['translation_units']}
    if not unit_ids or len(unit_ids) != len(set(unit_ids)):
        raise ValueError('Select at least one unique unit ID')
    invalid = set(unit_ids) - known
    if invalid:
        raise KeyError('Unknown units: ' + ','.join(sorted(invalid)))
    return list(unit_ids)


def text_jobs(prepared, output_dir, *, mode='reuse_existing', unit_ids=(),
              execute=False, allow_online=False, backend=None,
              max_requests=1):
    """Plan by default. When opted in, saves QA or drafts into new files.

    Caller controls any external provider, billing status and privacy agreement.
    A mock backend may be injected in offline tests.
    """
    if mode not in TEXT_MODE_CHOICES:
        raise ValueError(f'Invalid translation mode: {mode}')
    if mode == 'reuse_existing':
        return {'status': 'EXISTING_TRANSLATIONS_REUSED_NO_LLM_CALLS',
                'units': len(prepared['translation_units']), 'request_count': 0}
    ids = _unique_known_ids(prepared, unit_ids)
    required_calls = len(ids) * (4 if mode == 'regenerate_selected' else 1)
    if not execute:
        return {'status': 'plan_only_NO_LLM_CALLS', 'mode': mode,
                'unit_ids': ids, 'planned_requests': required_calls}
    if not allow_online:
        raise PermissionError('External translation/QA needs allow_online=True and execute=True')
    if backend is None:
        raise ValueError('Explicit backend required')
    if not isinstance(max_requests, int) or max_requests < required_calls:
        raise ValueError(f'max_requests={max_requests} below required {required_calls}')
    byid = {u['id']: u for u in prepared['translation_units']}
    outdir = Path(output_dir)
    # Validate every expected file path BEFORE making any remote request.
    paths = {uid: outdir / f'{mode}_{uid}_{fingerprint({"unit": uid, "source": prepared["translation_v2_metadata"]["source_master_data_fingerprint"]})[:14]}.json' for uid in ids}
    exists = [str(path) for path in paths.values() if path.exists()]
    if exists:
        raise FileExistsError('No API calls made: candidate files already exist: ' + ', '.join(exists))
    results = []
    for uid in ids:
        if mode == 'regenerate_selected':
            proposal = generate_draft(prepared, uid, backend, allow_translation=True)
        else:
            unit = byid[uid]
            proposed = str((unit.get('translation') or {}).get('natural_ar') or '').strip()
            if not proposed:
                raise ValueError(f'Existing Arabic missing in {uid}')
            req = {'unit_id': uid, 'source_text': unit_text(unit), 'natural_ar': proposed,
                   'course': prepared['translation_v2_metadata']['course'],
                   'context_links': prepared['translation_v2_metadata']['context_links'].get(uid, {})}
            instruction = ('Perform rigorous bilingual educational QA on English source and existing Arabic. '
                           'Report omissions, additions, negation, numbers, domain terms, examples, and pronunciation risks. '
                           'Output JSON with keys issues (array of strings), needs_review (bool), rationale (string). '
                           'Do not rewrite the translation or approve it.')
            review = backend(instruction, req, 'existing_bilingual_qa_v2_6')
            if not isinstance(review, dict) or not isinstance(review.get('issues'), list):
                raise ValueError('QA backend must return issues as an array')
            proposal = {'unit_id': uid, 'status': 'unapproved_ai_bilingual_review',
                        'source_text': req['source_text'], 'existing_natural_ar': proposed,
                        'qa': review, 'human_approved': False, 'final_dub_ar': None,
                        'provider': getattr(backend, 'provider_name', 'injected_backend')}
        save_new_json(paths[uid], proposal)
        results.append({'unit_id': uid, 'path': str(paths[uid]), 'status': proposal['status']})
    return {'status': 'UNAPPROVED_OUTPUTS_SAVED', 'mode': mode,
            'request_count': required_calls, 'results': results}


def _render_mms(text, dest, *, cache=None):
    # Local inference; Hugging Face weights are downloaded on first use.
    try:
        import torch
        from transformers import VitsModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install transformers and torch before local MMS TTS') from exc
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = cache if cache is not None else (VitsModel.from_pretrained('facebook/mms-tts-ara').to(device),
                                                       AutoTokenizer.from_pretrained('facebook/mms-tts-ara'))
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        arr = model(**inputs).waveform.squeeze().detach().cpu().numpy()
    import numpy as np
    pcm = (np.clip(arr, -1, 1) * 32767).astype('<i2')
    with wave.open(str(dest), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(model.config.sampling_rate)
        w.writeframes(pcm.tobytes())
    return (model, tokenizer)


def _render_gemini_tts(text, dest, *, client, voice, style, model):
    if client is None:
        raise ValueError('Explicit Gemini API client required')
    interaction = client.interactions.create(
        model=model,
        input=[{'type': 'user_input', 'content': [{
            'type': 'text', 'text': text,
            'annotations': [{'type': 'speech_metadata', 'style': style}],
        }]}],
        response_format={'type': 'audio'},
        generation_config={'speech_config': [{'voice': voice}]},
    )
    audio = getattr(interaction, 'output_audio', None)
    encoded = getattr(audio, 'data', None)
    if not encoded:
        raise RuntimeError('Gemini TTS returned no audio')
    raw = base64.b64decode(encoded)
    if not raw.startswith(b'RIFF') or len(raw) < 44:
        raise ValueError('Expected WAV from Gemini TTS; audio format differs')
    dest.write_bytes(raw)


def preview_jobs(prepared, output_dir, *, engine='plan_only', unit_ids=None,
                 execute=False, allow_online=False, allow_local=False,
                 edge_voice='ar-SA-HamedNeural', rate='+0%',
                 gemini_client_obj=None, gemini_voice='Kore',
                 gemini_model='gemini-3.8-flash-lite-tts',
                 gemini_style='clear natural Modern Standard Arabic educational narration',
                 max_units=1):
    if engine not in TTS_ENGINE_CHOICES:
        raise ValueError(f'Invalid TTS engine: {engine}')
    plan = select_previews(prepared, unit_ids)
    if engine == 'plan_only' or not execute:
        return {'status': 'plan_only_NO_TTS_CALLS', 'engine': engine, 'plan': plan}
    if not 1 <= len(plan) <= max_units:
        raise ValueError('Too many selected units for explicit max_units; stop before synthesis')
    if engine in ('edge_online', 'gemini_flash_lite') and not allow_online:
        raise PermissionError('Online TTS requires allow_online=True')
    if engine == 'mms_local' and not allow_local:
        raise PermissionError('Local TTS requires allow_local=True')
    if any(p['status'] != 'ready_for_unapproved_preview' for p in plan):
        raise ValueError('Some units have no Arabic natural text')
    if engine == 'edge_online':
        return run_edge_previews(prepared, output_dir, unit_ids=unit_ids,
                                 voice=edge_voice, rate=rate,
                                 execute=True, allow_online_tts=True)
    if engine == 'gemini_flash_lite' and gemini_client_obj is None:
        raise ValueError('Gemini client required; no calls made')
    out = Path(output_dir)
    settings = {'engine': engine, 'voice': gemini_voice if engine == 'gemini_flash_lite' else 'mms-ara',
                'model': gemini_model if engine == 'gemini_flash_lite' else 'facebook/mms-tts-ara',
                'style': gemini_style if engine == 'gemini_flash_lite' else None,
                'source': prepared['translation_v2_metadata']['source_master_data_fingerprint']}
    filenames = {}
    for item in plan:
        sig = fingerprint({**settings, 'unit_id': item['id'], 'text': item['text']})[:18]
        suffix = 'wav'
        wav = out / f"{item['id']}_{engine}_{sig}.{suffix}"
        manifest = out / f"{item['id']}_{engine}_{sig}.json"
        if wav.exists() or manifest.exists():
            raise FileExistsError(f'Existing output; refusing duplicate request: {wav}')
        filenames[item['id']] = (wav, manifest)
    out.mkdir(parents=True, exist_ok=True)
    results=[]
    cache=None
    for item in plan:
        wav, manifest=filenames[item['id']]
        tmp=out / (wav.stem + '.partial.wav')
        try:
            if engine == 'mms_local':
                cache = _render_mms(item['text'], tmp, cache=cache)
            else:
                _render_gemini_tts(item['text'], tmp, client=gemini_client_obj, voice=gemini_voice,
                                   style=gemini_style, model=gemini_model)
            if not tmp.is_file() or tmp.stat().st_size < 44:
                raise RuntimeError('TTS returned no usable WAV audio')
            tmp.rename(wav)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        seconds = _duration(wav)
        doc={'status': 'preview_only_not_approved', 'unit_id': item['id'],
             'engine': engine, 'settings': settings, 'source_duration_seconds': item['source_duration_seconds'],
             'generated_duration_seconds': seconds, 'difference_seconds': None if seconds is None else round(seconds-item['source_duration_seconds'],3),
             'translation_approved': False, 'audio_approved': False,
             'text': item['text'], 'audio_filename': wav.name, 'sha256': sha256_file(wav)}
        save_new_json(manifest, doc)
        results.append(doc)
    return {'status': 'PREVIEWS_CREATED_NOT_APPROVED', 'results': results}
