"""References stored by ID, adjacent text resolved on demand."""
from .common import unit_text


def link_units(units, window=2):
    """Generate context links without altering the source data."""
    if len({u['id'] for u in units}) != len(units):
        raise ValueError('Duplicate TU identifiers')
    return {u['id']:{'previous_unit_ids':[x['id'] for x in units[max(0,i-window):i]],
                     'next_unit_ids':[x['id'] for x in units[i+1:i+1+window]]}
            for i,u in enumerate(units)}


def build_context_package(master, unit_id, *, links=None, glossary=None, window=2):
    units = master.get('translation_units') or []
    byid = {u['id']:u for u in units}
    if unit_id not in byid: raise KeyError(unit_id)
    links = links or link_units(units,window)
    rel = links[unit_id]
    course = master.get('translation_v2_metadata', {}).get('course') or master.get('course') or {}
    approved = (glossary or {}).get('approved',[])
    current_text = unit_text(byid[unit_id])
    relevant = [x for x in approved if str(x.get('source_term') or x.get('source') or '').lower() in current_text.lower()]
    return {'unit_id':unit_id,'course':{k:course.get(k) for k in ('id','title','description','chapter','lesson','topic')},
            'source_text':current_text,
            'previous':[{'id':id,'source_text':unit_text(byid[id])} for id in rel['previous_unit_ids']],
            'next':[{'id':id,'source_text':unit_text(byid[id])} for id in rel['next_unit_ids']],
            'approved_terminology':relevant,
            'note':'Unknown visual context must not be invented; no asserted translation from adjacent data.'}
