"""Quality scorers. Heavy libraries are imported only when a scorer is constructed.

Every scorer exposes the same tiny interface so tests can inject fakes:
  QE scorer   : .name, .score(list[(source, hypothesis)]) -> list[float]   (lower is better)
  Spell check : .name, .misspelled(list[str]) -> set[str]
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

from .common import sha256_text


class CachedQE:
    """Disk cache in front of any QE scorer: one JSON line per (metric, source, hypothesis)."""

    def __init__(self, scorer, cache_path):
        self.scorer, self.name = scorer, scorer.name
        self.path = Path(cache_path)
        self.cache = {}
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec['key']] = rec['score']

    def _key(self, src, hyp):
        return sha256_text(self.name + '\x1f' + src + '\x1f' + hyp)

    def score(self, pairs):
        missing = [(s, h) for s, h in dict.fromkeys(pairs) if self._key(s, h) not in self.cache]
        if missing:
            scores = self.scorer.score(missing)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as f:
                for (s, h), sc in zip(missing, scores):
                    k = self._key(s, h)
                    self.cache[k] = float(sc)
                    f.write(json.dumps({'key': k, 'score': float(sc)}) + '\n')
        return [self.cache[self._key(s, h)] for s, h in pairs]


_EN_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
_AR_SENT = re.compile(r'(?<=[.!?؟])\s+')


class SegmentedQE:
    """Sentence-level scoring for long units.

    Segment-level QE models are trained on sentence-sized inputs; a 30-second paragraph gets a
    poor score even when every sentence is translated well. When the source has >= min_words
    words and source and translation split into the SAME number of sentences, each sentence pair
    is scored and the result is the source-length-weighted mean; otherwise the whole unit is scored."""

    def __init__(self, base, min_words=60):
        self.base, self.min_words = base, min_words
        self.name = f'{base.name}+seg{min_words}'
        self.segmented = 0

    def _split(self, src, hyp):
        if len(src.split()) < self.min_words:
            return None
        s = [x.strip() for x in _EN_SENT.split(src) if x.strip()]
        h = [x.strip() for x in _AR_SENT.split(hyp) if x.strip()]
        return list(zip(s, h)) if len(s) == len(h) > 1 else None

    def score(self, pairs):
        plan, flat = [], []
        for src, hyp in pairs:
            parts = self._split(src, hyp)
            if parts:
                plan.append((len(flat), len(parts), [len(p[0].split()) for p in parts]))
                flat += parts
            else:
                plan.append((len(flat), 1, None))
                flat.append((src, hyp))
        scores = self.base.score(flat) if flat else []
        out = []
        for start, n, weights in plan:
            chunk = scores[start:start + n]
            if weights:
                self.segmented += 1
                out.append(sum(c * w for c, w in zip(chunk, weights)) / sum(weights))
            else:
                out.append(chunk[0])
        return out


class MetricXQE:
    """MetricX-24 hybrid in QE mode (reference-free). 0 = perfect, 25 = worst.

    Same input format and output head as `metricx24.predict --qe`, through the stock MT5 class."""

    name = 'metricx-24-hybrid-large-v2p6-qe'

    def __init__(self, model_id='google/metricx-24-hybrid-large-v2p6', device=None):
        import torch
        from transformers import AutoTokenizer, MT5ForConditionalGeneration
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained('google/mt5-large')
        self.model = MT5ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32).to(self.device).eval()

    def score(self, pairs, batch_size=8):
        torch, scores = self.torch, []
        for start in range(0, len(pairs), batch_size):
            enc = [self.tok('source: ' + s + ' candidate: ' + h, max_length=1536, truncation=True)['input_ids'][:-1]
                   for s, h in pairs[start:start + batch_size]]
            ids = torch.full((len(enc), max(map(len, enc))), self.tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, e in enumerate(enc):
                ids[r, :len(e)] = torch.tensor(e)
                mask[r, :len(e)] = 1
            with torch.inference_mode():
                logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                    decoder_input_ids=torch.zeros((len(enc), 1), dtype=torch.long,
                                                                  device=self.device)).logits
            scores += torch.clamp(logits[:, 0, 250089], 0, 25).float().cpu().tolist()
        return scores

    def close(self):
        import gc
        del self.model
        gc.collect()
        if self.device == 'cuda':
            self.torch.cuda.empty_cache()


class Hunspell:
    """hunspell + hunspell-ar (Ayaspell). Installs the packages on Debian/Colab if missing."""

    name = 'hunspell-ar'

    def __init__(self, install=True):
        if not shutil.which('hunspell') and install:
            subprocess.run(['apt-get', 'update', '-qq'], check=True, capture_output=True)
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'hunspell', 'hunspell-ar'], check=True,
                           capture_output=True)
        if not shutil.which('hunspell'):
            raise RuntimeError('hunspell is not installed')

    def misspelled(self, words):
        words = sorted(set(words))
        if not words:
            return set()
        out = subprocess.run(['hunspell', '-i', 'utf-8', '-d', 'ar', '-l'], input='\n'.join(words),
                             capture_output=True, text=True, check=True).stdout
        return set(out.split())
