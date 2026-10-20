import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def clone(obj):
    return copy.deepcopy(obj)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')).hexdigest()


def read_json(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def save_new_json(path, obj):
    """Never replace an existing artifact; atomic initial write in destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(obj, ensure_ascii=False, indent=2) + '\n'
    with path.open('x', encoding='utf-8') as out:
        out.write(encoded)
    return path


def require_time(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Invalid finite timestamp: {label}={value!r}')
    return float(value)


def unit_times(unit):
    timing = unit.get('timing') or {}
    begin = timing.get('source_start', timing.get('start', unit.get('start')))
    end = timing.get('source_end', timing.get('end', unit.get('end')))
    a, b = require_time(begin, 'start'), require_time(end, 'end')
    if a < 0 or b <= a:
        raise ValueError(f'Invalid time range for {unit.get("id")}: {a}..{b}')
    return a, b


def unit_text(unit):
    return str((unit.get('source') or {}).get('text') or unit.get('source_text') or unit.get('text') or '').strip()


def approved_state(value):
    return str(value or '').lower() == 'approved'
