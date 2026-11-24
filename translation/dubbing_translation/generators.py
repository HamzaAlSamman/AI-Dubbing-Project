"""Candidate generators and the LLM client.

A generator turns one task into zero or more candidate texts:
    gen.name                      -> stable id, includes model + prompt version (part of the cache path)
    gen.generate(task, budget)    -> list[{'text': str, 'variant': str}]
task   = {'unit_id', 'source_text', 'context_before', 'context_after', 'slot_s', 'window_s'}
budget = {'min': int, 'max': int, 'short': int}   (Arabic letters)

Models and network clients are created lazily on first use, never at import.
"""
import json
import os
import re
import time
from pathlib import Path

from .prompts import ISOMETRIC_SYSTEM, ISOMETRIC_VERSION


class QuotaExhausted(RuntimeError):
    """Daily quota reached: the run stops cleanly; finished units stay cached."""


def parse_json_object(raw):
    raw = (raw or '').strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*', '', raw).rsplit('```', 1)[0].strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    for i, ch in enumerate(raw):
        if ch == '{':
            try:
                obj = json.JSONDecoder().raw_decode(raw[i:])[0]
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


class LLMClient:
    """JSON-returning chat call for Gemini (google-genai) or OpenRouter ':free' models.

    Paced below per-minute limits; transient errors are retried; a daily quota raises QuotaExhausted."""

    def __init__(self, provider, model, *, min_interval_s=4.5, temperature=0.0, max_attempts=5,
                 timeout_s=180, thinking_budget=None):
        if provider not in ('gemini', 'openrouter'):
            raise ValueError(provider)
        if provider == 'openrouter' and not model.endswith(':free'):
            raise ValueError(f'refusing paid OpenRouter model {model!r}; only :free models are allowed')
        self.provider, self.model = provider, model
        self.min_interval_s, self.temperature, self.max_attempts = min_interval_s, temperature, max_attempts
        self.timeout_s, self.thinking_budget = timeout_s, thinking_budget
        self._client, self._last = None, 0.0
        self.calls = 0

    def _gemini(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            key = os.environ.get('GEMINI_API_KEY')
            if not key:
                raise RuntimeError('GEMINI_API_KEY is not loaded (run the secrets cell yourself)')
            # explicit timeout: without it a stalled request can block the run indefinitely
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))
        return self._client

    def _once(self, system, user):
        if self.provider == 'gemini':
            from google.genai import types
            extra = {}
            if self.thinking_budget is not None:
                extra['thinking_config'] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
            conf = types.GenerateContentConfig(system_instruction=system, temperature=self.temperature,
                                               response_mime_type='application/json', **extra)
            return self._gemini().models.generate_content(model=self.model, contents=user, config=conf).text or ''
        import requests
        key = os.environ.get('OPENROUTER_API_KEY')
        if not key:
            raise RuntimeError('OPENROUTER_API_KEY is not loaded')
        resp = requests.post('https://openrouter.ai/api/v1/chat/completions', timeout=180,
                             headers={'Authorization': f'Bearer {key}'},
                             json={'model': self.model, 'temperature': self.temperature,
                                   'messages': [{'role': 'system', 'content': system},
                                                {'role': 'user', 'content': user}]})
        if resp.status_code == 429 and 'per-day' in resp.text.lower().replace(' ', '-'):
            raise QuotaExhausted(resp.text[:200])
        resp.raise_for_status()
        return resp.json()['choices'][0]['message'].get('content') or ''

    def chat_json(self, system, user):
        for _ in range(self.max_attempts):
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                self.calls += 1
                raw = self._once(system, user)
            except QuotaExhausted:
                raise
            except Exception as exc:                       # noqa: BLE001 - provider errors are heterogeneous
                text = str(exc)
                if 'PerDay' in text or 'per day' in text.lower():
                    raise QuotaExhausted(text[:200]) from exc
                if any(s in text for s in ('429', 'RESOURCE_EXHAUSTED', '500', 'INTERNAL', '502', '503', '504',
                                           'UNAVAILABLE', 'DEADLINE_EXCEEDED', 'timed out', 'Timeout')):
                    time.sleep(45)
                    continue
                raise
            obj = parse_json_object(raw)
            if obj is not None:
                return obj
        return None


class IsometricLLM:
    """LLM asked for two lengths of the same translation inside an explicit letter budget
    (length-controlled prompting with multiple outputs, as in the isometric-MT literature)."""

    uses_budget = True

    def __init__(self, client):
        self.client = client
        self.name = f'{client.provider}__{client.model.replace("/", "_").replace(":", "_")}__{ISOMETRIC_VERSION}'

    def generate(self, task, budget):
        user = json.dumps({'context_before': task.get('context_before') or [],
                           'current_unit': task['source_text'],
                           'context_after': task.get('context_after') or [],
                           'slot_seconds': round(task['slot_s'], 1),
                           'target_letters_min': budget['min'], 'target_letters_max': budget['max'],
                           'short_letters': budget['short']}, ensure_ascii=False)
        obj = self.client.chat_json(ISOMETRIC_SYSTEM, user) or {}
        return [{'text': obj[k].strip(), 'variant': k} for k in ('fit', 'short')
                if isinstance(obj.get(k), str) and obj[k].strip()]


class HyMT:
    """tencent/HY-MT1.5-7B with its official plain template (no context: with the contextual
    template the 7B model sometimes translated the context too). 4-bit on a Colab T4/L4."""

    PROMPT = 'Translate the following segment into Arabic, without additional explanation.\n\n{src}'
    uses_budget = False

    def __init__(self, model_id='tencent/HY-MT1.5-7B', load_in_4bit=True):
        self.model_id, self.load_in_4bit = model_id, load_in_4bit
        self.name = model_id.replace('/', '_') + '__plain_v1'
        self._tok = self._model = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        kw = {'device_map': 'auto', 'low_cpu_mem_usage': True}
        if self.load_in_4bit:
            kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                           bnb_4bit_compute_dtype=torch.float16)
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw).eval()

    def generate(self, task, budget):
        import torch
        if self._model is None:
            self._load()
        enc = self._tok.apply_chat_template([{'role': 'user', 'content': self.PROMPT.format(src=task['source_text'])}],
                                            add_generation_prompt=True, return_tensors='pt', return_dict=True)
        with torch.inference_mode():
            out = self._model.generate(input_ids=enc['input_ids'].to(self._model.device),
                                       attention_mask=enc['attention_mask'].to(self._model.device),
                                       max_new_tokens=512, do_sample=False, repetition_penalty=1.05,
                                       pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id)
        text = self._tok.decode(out[0][enc['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        return [{'text': text, 'variant': 'plain'}] if text else []

    def close(self):
        import gc
        import torch
        self._model = self._tok = None
        gc.collect()
        torch.cuda.empty_cache()


_THINK = re.compile(r'<think>.*?</think>', re.S)


def strip_reasoning(text):
    """Drop the reasoning block of reasoning-capable chat models; an unclosed block yields ''."""
    t = _THINK.sub('', text or '')
    if '<think>' in t:
        t = t.split('<think>')[0]
    return t.replace('</think>', '').strip()


class HFChatMT:
    """Any chat-template MT model from the Hugging Face Hub (Hy-MT2, TranslateGemma-style prompts...).

    Produces the greedy translation plus `samples` sampled variants (model-card sampling settings),
    which gives the selector more candidates of different length at no extra API cost."""

    uses_budget = False
    TEMPLATES = {
        # Hy-MT2 model card (English instruction, full target-language name)
        'hy_mt2': ('Translate the following text into Arabic. Note that you should only output the translated '
                   'result without any additional explanation:\n\n{src}'),
        # general chat LLMs (Qwen3, Jais-3, ...): plain instruction, MSA explicitly
        'msa_plain': ('Translate the following English text into Modern Standard Arabic. Output only the '
                      'Arabic translation, without explanations, notes or transliteration.\n\n{src}'),
    }

    def __init__(self, model_id, template='hy_mt2', *, samples=1, load_in_4bit=True, seed=0,
                 sampling=None, trust_remote_code=False, template_kwargs=None, compute_dtype='float16'):
        """trust_remote_code: models with custom code (Jais-3). template_kwargs: extra arguments of
        apply_chat_template, e.g. {'enable_thinking': False} for Qwen3 hybrid-reasoning models.
        compute_dtype: 'float16' (default) or 'float32' for models that overflow in fp16 on T4/P100;
        a non-default value is part of the generator name, so cached outputs never mix."""
        self.model_id, self.template, self.samples = model_id, template, samples
        self.trust_remote_code, self.template_kwargs = trust_remote_code, dict(template_kwargs or {})
        self.load_in_4bit, self.seed, self.compute_dtype = load_in_4bit, seed, compute_dtype
        self.sampling = sampling or {'temperature': 0.7, 'top_p': 0.6, 'top_k': 20, 'repetition_penalty': 1.05}
        self.name = f'{model_id.replace("/", "_")}__{template}_g1s{samples}' + (
            '' if compute_dtype == 'float16' else f'_{compute_dtype}')
        self._tok = self._model = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        dtype = getattr(torch, self.compute_dtype)
        kw = {'device_map': 'auto', 'low_cpu_mem_usage': True}
        if self.load_in_4bit:
            kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                           bnb_4bit_compute_dtype=dtype)
        else:
            kw['torch_dtype'] = dtype
        rc = {'trust_remote_code': True} if self.trust_remote_code else {}
        self._tok = AutoTokenizer.from_pretrained(self.model_id, **rc)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw, **rc).eval()

    def _run(self, enc, **gen):
        import torch
        with torch.inference_mode():
            out = self._model.generate(input_ids=enc['input_ids'].to(self._model.device),
                                       attention_mask=enc['attention_mask'].to(self._model.device),
                                       max_new_tokens=768,
                                       pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id, **gen)
        return strip_reasoning(self._tok.decode(out[0][enc['input_ids'].shape[-1]:], skip_special_tokens=True))

    def generate(self, task, budget):
        import torch
        if self._model is None:
            self._load()
        prompt = self.TEMPLATES[self.template].format(src=task['source_text'])
        enc = self._tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True,
                                            return_tensors='pt', return_dict=True, **self.template_kwargs)
        items = []
        greedy = self._run(enc, do_sample=False, repetition_penalty=self.sampling.get('repetition_penalty', 1.0))
        if greedy:
            items.append({'text': greedy, 'variant': 'greedy'})
        for i in range(self.samples):
            torch.manual_seed(self.seed + i + sum(map(ord, task['unit_id'])))
            text = self._run(enc, do_sample=True, **self.sampling)
            if text and text not in {it['text'] for it in items}:
                items.append({'text': text, 'variant': f'sample{i}'})
        return items

    def close(self):
        import gc
        import torch
        self._model = self._tok = None
        gc.collect()
        torch.cuda.empty_cache()


class Replay:
    """Reuse outputs saved by an earlier run or experiment (one JSON per unit with 'natural_ar').

    Lets a new run reuse expensive generations without calling the model again; the source
    system name is kept in the generator name so provenance stays visible."""

    uses_budget = False

    def __init__(self, folder, name):
        self.folder, self.name = Path(folder), name

    def generate(self, task, budget):
        path = self.folder / f'{task["unit_id"]}.json'
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding='utf-8'))
        text = data.get('natural_ar')
        return [{'text': text.strip(), 'variant': 'replay'}] if isinstance(text, str) and text.strip() else []
