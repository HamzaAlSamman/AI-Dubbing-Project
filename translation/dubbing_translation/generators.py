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

from .prompts import ISOMETRIC_SYSTEM, ISOMETRIC_V2_SYSTEM, ISOMETRIC_V2_VERSION, ISOMETRIC_VERSION, prompt_sha


class TransientLLMError(RuntimeError):
    """The service did not answer (503, 5xx, timeout, per-minute 429) after the bounded retries.
    Not a translation result: never cached; the unit is deferred and retried by the next run."""


class PermanentLLMError(RuntimeError):
    """The request itself is refused (4xx other than rate limits, blocked content, bad request).
    Recorded once as an error for that unit so it is not retried forever; the run is degraded."""


class QuotaExhausted(RuntimeError):
    """Daily quota reached: the run stops cleanly; finished units stay cached."""


class CallLimitReached(QuotaExhausted):
    """The run's own cap on LLM requests (CallBudget) is reached: handled like a daily quota."""


class AuthenticationError(RuntimeError):
    """The API key is missing, invalid or not allowed (401/403). Not a property of a unit: never cached;
    the role stops for the run and its units are deferred until the key is fixed."""


class InvalidLLMOutput(RuntimeError):
    """The model answered, but never with a valid JSON object. Not an (empty) translation result: never
    cached as one. Recorded in the unit's failure ledger and retried by later runs, a bounded number of times."""


class CallBudget:
    """Shared cap on LLM requests across every client of a run (e.g. the development smoke test)."""

    def __init__(self, limit):
        self.limit, self.used = (None if limit is None else int(limit)), 0

    def take(self):
        if self.limit is not None and self.used >= self.limit:
            raise CallLimitReached(f'LLM request cap reached ({self.limit} requests)')
        self.used += 1


_AUTH_MARKERS = ('401', '403', 'UNAUTHENTICATED', 'PERMISSION_DENIED', 'API_KEY_INVALID', 'API key not valid',
                 'API key expired', 'API_KEY_SERVICE_BLOCKED')
_TRANSIENT_MARKERS = ('429', 'RESOURCE_EXHAUSTED', '500', 'INTERNAL', '502', '503', '504', 'UNAVAILABLE',
                      'DEADLINE_EXCEEDED', 'timed out', 'Timeout', 'ConnectionError', 'RemoteDisconnected')


def classify_llm_error(exc):
    """'quota' | 'auth' | 'transient' | 'permanent' for a provider exception (heterogeneous SDK errors)."""
    text = f'{type(exc).__name__}: {exc}'
    if 'PerDay' in text or 'per day' in text.lower():
        return 'quota'
    if isinstance(exc, AuthenticationError) or any(m in text for m in _AUTH_MARKERS):
        return 'auth'
    if any(m in text for m in _TRANSIENT_MARKERS):
        return 'transient'
    return 'permanent'


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
                 timeout_s=180, thinking_budget=None, budget=None, max_invalid=2, transient_sleep_s=45):
        if provider not in ('gemini', 'openrouter'):
            raise ValueError(provider)
        if provider == 'openrouter' and not model.endswith(':free'):
            raise ValueError(f'refusing paid OpenRouter model {model!r}; only :free models are allowed')
        self.provider, self.model = provider, model
        self.min_interval_s, self.temperature, self.max_attempts = min_interval_s, temperature, max_attempts
        self.timeout_s, self.thinking_budget = timeout_s, thinking_budget
        self._client, self._last = None, 0.0
        self.calls = 0
        self.budget = budget                        # CallBudget shared by the run's clients, or None
        self.max_invalid, self.transient_sleep_s = max_invalid, transient_sleep_s

    def _gemini(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            key = os.environ.get('GEMINI_API_KEY')
            if not key:
                raise AuthenticationError('GEMINI_API_KEY is not loaded (run the secrets cell yourself)')
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
            raise AuthenticationError('OPENROUTER_API_KEY is not loaded')
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
        """JSON object from the model. Failure types (none of them is a translation result):
          QuotaExhausted / CallLimitReached  daily quota or the run's request cap
          AuthenticationError                key missing / invalid / not allowed
          TransientLLMError                  503, 5xx, timeout, per-minute 429 on every attempt
          PermanentLLMError                  the request itself is refused
          InvalidLLMOutput                   answered, but no valid JSON object after `max_invalid` answers
        A valid JSON object without usable content is returned as is (a valid empty result)."""
        last_transient, invalid = None, 0
        for _ in range(self.max_attempts):
            if self.budget is not None:
                self.budget.take()
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                self.calls += 1
                raw = self._once(system, user)
            except (QuotaExhausted, AuthenticationError):
                raise
            except Exception as exc:                       # noqa: BLE001 - provider errors are heterogeneous
                kind = classify_llm_error(exc)
                if kind == 'quota':
                    raise QuotaExhausted(str(exc)[:200]) from exc
                if kind == 'auth':
                    raise AuthenticationError(f'{type(exc).__name__}: {str(exc)[:200]}') from exc
                if kind == 'transient':
                    last_transient = str(exc)[:200]
                    time.sleep(self.transient_sleep_s)
                    continue
                raise PermanentLLMError(f'{type(exc).__name__}: {str(exc)[:200]}') from exc
            last_transient = None
            obj = parse_json_object(raw)
            if obj is not None:
                return obj
            invalid += 1
            if invalid >= self.max_invalid:
                break
        if last_transient is not None:
            raise TransientLLMError(last_transient)
        raise InvalidLLMOutput(f'{self.provider}:{self.model}: no valid JSON object in {invalid} answers')


class IsometricLLM:
    """LLM asked for two lengths of the same translation inside an explicit letter budget
    (length-controlled prompting with multiple outputs, as in the isometric-MT literature)."""

    uses_budget = True

    def __init__(self, client, *, language_aware=False):
        """language_aware: V2 prompt that also receives the style rules and the glossary lines of the
        unit (task['language_block'], set by the pipeline when the run has a [language] section)."""
        self.client, self.language_aware = client, language_aware
        self.version = ISOMETRIC_V2_VERSION if language_aware else ISOMETRIC_VERSION
        self.system = ISOMETRIC_V2_SYSTEM if language_aware else ISOMETRIC_SYSTEM
        self.prompt_sha = prompt_sha(self.system)
        self.name = f'{client.provider}__{client.model.replace("/", "_").replace(":", "_")}__{self.version}'
        if language_aware:                                   # V2: the prompt text itself is in the cache path
            self.name += f'__p{self.prompt_sha[:8]}'

    def describe(self):
        return {'name': self.name, 'kind': 'llm', 'provider': self.client.provider, 'model': self.client.model,
                'prompt_version': self.version, 'prompt_sha': self.prompt_sha,
                'decoding': {'temperature': getattr(self.client, 'temperature', None),
                             'thinking_budget': getattr(self.client, 'thinking_budget', None)}}

    def generate(self, task, budget):
        payload = {'context_before': task.get('context_before') or [],
                   'current_unit': task['source_text'],
                   'context_after': task.get('context_after') or [],
                   'slot_seconds': round(task['slot_s'], 1),
                   'target_letters_min': budget['min'], 'target_letters_max': budget['max'],
                   'short_letters': budget['short']}
        if self.language_aware:
            payload.update(task.get('language_block') or {})
        obj = self.client.chat_json(self.system, json.dumps(payload, ensure_ascii=False))
        return [{'text': obj[k].strip(), 'variant': k} for k in ('fit', 'short')
                if isinstance(obj.get(k), str) and obj[k].strip()]


class HyMT:
    """tencent/HY-MT1.5-7B with its official plain template (no context: with the contextual
    template the 7B model sometimes translated the context too). 4-bit on a Colab T4/L4."""

    PROMPT = 'Translate the following segment into Arabic, without additional explanation.\n\n{src}'
    # official terminology-intervention template of the model card (Chinese instruction, one line per term)
    TERMS_PROMPT = ('参考下面的翻译：\n{terms}\n\n将以下文本翻译为阿拉伯语，注意只需要输出翻译后的结果，不要额外解释：\n{src}')
    uses_budget = False

    def __init__(self, model_id='tencent/HY-MT1.5-7B', load_in_4bit=True, *, language_aware=False):
        """language_aware: when the unit has glossary terms (task['term_pairs']), use the terminology template."""
        self.model_id, self.load_in_4bit, self.language_aware = model_id, load_in_4bit, language_aware
        self.prompt_sha = prompt_sha(self.PROMPT + ('\n' + self.TERMS_PROMPT if language_aware else ''))
        self.name = model_id.replace('/', '_') + ('__terms_v1' if language_aware else '__plain_v1')
        if language_aware:
            self.name += f'__p{self.prompt_sha[:8]}'
        self._tok = self._model = None

    def describe(self):
        return {'name': self.name, 'kind': 'hf', 'model': self.model_id, 'load_in_4bit': self.load_in_4bit,
                'prompt_version': 'terms_v1' if self.language_aware else 'plain_v1', 'prompt_sha': self.prompt_sha,
                'decoding': {'do_sample': False, 'repetition_penalty': 1.05, 'max_new_tokens': 512}}

    def prompt(self, task):
        pairs = (task.get('term_pairs') or []) if self.language_aware else []
        if pairs:
            terms = '\n'.join(f'{en} 翻译成 {ar}' for en, ar in pairs)
            return self.TERMS_PROMPT.format(terms=terms, src=task['source_text'])
        return self.PROMPT.format(src=task['source_text'])

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
        enc = self._tok.apply_chat_template([{'role': 'user', 'content': self.prompt(task)}],
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
        # V2: Hy-MT2 instruction with the unit's glossary terms (falls back to 'hy_mt2' when there are none)
        'hy_mt2_terms': ('Translate the following text into Modern Standard Arabic. Use these translations for '
                         'the terms:\n{terms}\nNote that you should only output the translated result without any '
                         'additional explanation:\n\n{src}'),
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
        self.language_aware = template.endswith('_terms')
        self.prompt_sha = prompt_sha(self.TEMPLATES[template] + json.dumps(self.sampling, sort_keys=True))
        self.name = f'{model_id.replace("/", "_")}__{template}_g1s{samples}' + (
            '' if compute_dtype == 'float16' else f'_{compute_dtype}')
        if self.language_aware:
            self.name += f'__p{self.prompt_sha[:8]}'

        self._tok = self._model = None

    def describe(self):
        return {'name': self.name, 'kind': 'hf', 'model': self.model_id, 'load_in_4bit': self.load_in_4bit,
                'compute_dtype': self.compute_dtype, 'prompt_version': self.template, 'prompt_sha': self.prompt_sha,
                'decoding': {'greedy': True, 'samples': self.samples, 'sampling': self.sampling, 'seed': self.seed}}

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
        if self.template.endswith('_terms'):
            pairs = task.get('term_pairs') or []
            prompt = (self.TEMPLATES[self.template].format(terms='\n'.join(f'{en} -> {ar}' for en, ar in pairs),
                                                           src=task['source_text']) if pairs else
                      self.TEMPLATES[self.template[:-len('_terms')]].format(src=task['source_text']))
        else:
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
