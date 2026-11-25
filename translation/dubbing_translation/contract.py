"""Data contract between the translation step and the TTS stage (schemas/translation_final.schema.json).

The output keeps every ORIGINAL source unit (id, text, start/end, word alignment) even when units were
merged, plus the written line, the accurate line, the spoken text and the provenance of the chosen line.
validate_final() is called before the file is written: an invalid output is a bug and stops the run."""
import json
from pathlib import Path

SCHEMA_VERSION = 'translation_final/2'
SCHEMA_PATH = Path(__file__).resolve().parent.parent / 'schemas' / 'translation_final.schema.json'


def source_reference(task, timing):
    """The original unit as the TTS stage needs it (timing may be unknown: start/end/words = None)."""
    tm = timing or {}
    return {'unit_id': task['unit_id'], 'source_text': task['source_text'],
            'start': tm.get('start'), 'end': tm.get('end'), 'words': tm.get('words')}


def validate_final(doc):
    """jsonschema when installed (Colab, CI); otherwise the required keys of every level."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
    try:
        import jsonschema
    except ImportError:
        jsonschema = None
    if jsonschema is not None:
        jsonschema.validate(doc, schema)
        return
    missing = [k for k in schema['required'] if k not in doc]
    unit_req = schema['$defs']['unit']['required']
    src_req = schema['$defs']['source_unit']['required']
    for u in doc.get('units', []):
        missing += [f"{u.get('unit_id')}.{k}" for k in unit_req if k not in u]
        for s in u.get('source_units') or []:
            missing += [f"{u.get('unit_id')}.source_units.{k}" for k in src_req if k not in s]
    if missing:
        raise ValueError('translation_final does not match its schema: missing ' + ', '.join(missing[:20]))


def check_coverage(doc, source_refs):
    """Every ORIGINAL unit appears exactly once in the output, in order, with its original text, start/end
    and word alignment unchanged; a derived (merged) unit spans exactly its parts. Raises ValueError."""
    order = [s['unit_id'] for u in doc['units'] for s in (u.get('source_units') or [])]
    problems = []
    dup = sorted({x for x in order if order.count(x) > 1})
    if dup:
        problems.append(f'appear more than once: {dup}')
    missing = [x for x in source_refs if x not in order]
    if missing:
        problems.append(f'missing: {missing}')
    extra = sorted({x for x in order if x not in source_refs})
    if extra:
        problems.append(f'not in the source: {extra}')
    if not problems and order != list(source_refs):
        problems.append('original order changed')
    for u in doc['units']:
        refs = u.get('source_units') or []
        for s in refs:
            ref = source_refs.get(s['unit_id'])
            if ref is not None and any(s.get(k) != ref.get(k) for k in ('source_text', 'start', 'end', 'words')):
                problems.append(f"{s['unit_id']}: text or timing changed")
        if refs and (u.get('start') != refs[0].get('start') or u.get('end') != refs[-1].get('end')):
            problems.append(f"{u['unit_id']}: start/end differ from its source units")
    if problems:
        raise ValueError('translation_final does not cover the source exactly: ' + '; '.join(problems[:10]))


def source_readiness(source_refs):
    """What the source data allows. Timings are never invented: a unit without start/end is ready for a
    translation test but not for audio synchronisation."""
    no_timing = [u for u, r in source_refs.items() if r.get('start') is None or r.get('end') is None]
    no_words = [u for u, r in source_refs.items() if not r.get('words')]
    return {'translation_test': True, 'audio_sync': not no_timing, 'word_alignment': not no_words,
            'units_without_timing': no_timing, 'units_without_word_alignment': no_words}
