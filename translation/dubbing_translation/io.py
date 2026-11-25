"""Loading configuration, tasks and timing."""
import json
from pathlib import Path

from .common import read_json
from .timing import windows_from_slots, windows_from_timing

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11
    tomllib = None

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / 'configs' / 'translation_v1.toml'
V2_CONFIG = DEFAULT_CONFIG.with_name('translation_v2.toml')     # + language policy, auto glossary, polish


def load_config(path=DEFAULT_CONFIG, **overrides):
    path = Path(path)
    if path.suffix == '.json':
        cfg = json.loads(path.read_text(encoding='utf-8'))
    else:
        if tomllib is None:
            raise RuntimeError('TOML config needs Python 3.11+; use a .json config instead')
        cfg = tomllib.loads(path.read_text(encoding='utf-8'))
    for dotted, value in overrides.items():          # load_config(gate__qe_max=4.5)
        section, key = dotted.split('__', 1)
        cfg.setdefault(section, {})[key] = value
    return cfg


def load_tasks(path):
    """Tasks file: {'tasks': [{unit_id, source_text, context_before, context_after, source_duration_s}]}
    (the experiments_v1/tasks.json format) or a bare list of the same objects."""
    data = read_json(path)
    tasks = data['tasks'] if isinstance(data, dict) else data
    for t in tasks:
        for key in ('unit_id', 'source_text'):
            if not t.get(key):
                raise ValueError(f'task without {key}: {t}')
    return tasks


def _unit_times(unit):
    timing = unit.get('timing') or {}
    start = timing.get('source_start', timing.get('start', unit.get('start')))
    end = timing.get('source_end', timing.get('end', unit.get('end')))
    return (float(start), float(end)) if start is not None and end is not None else None


def load_source_timing(tasks, master_path=None):
    """{unit_id: {'start', 'end', 'words'}} from the source master (words = the unit's word-level
    alignment when the master has it, else None). Units without timing are left out."""
    out = {}
    if not master_path or not Path(master_path).exists():
        return out
    units = {u['id']: u for u in read_json(master_path)['translation_units']}
    for t in tasks:
        u = units.get(t['unit_id'], {})
        se = _unit_times(u)
        if se is None:
            continue
        timing = u.get('timing') or {}
        words = u.get('words') or timing.get('words') or u.get('word_timings')
        out[t['unit_id']] = {'start': se[0], 'end': se[1], 'words': words}
    return out


def load_windows(tasks, master_path=None, **kw):
    """Windows from the source master's unit timings when available, else slot = source_duration_s."""
    if master_path and Path(master_path).exists():
        units = {u['id']: u for u in read_json(master_path)['translation_units']}
        timings = []
        for t in tasks:
            se = _unit_times(units.get(t['unit_id'], {}))
            if se is None:
                break
            timings.append((t['unit_id'], *se))
        else:
            return windows_from_timing(timings, **kw)
    return windows_from_slots({t['unit_id']: t['source_duration_s'] for t in tasks})
