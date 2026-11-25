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
