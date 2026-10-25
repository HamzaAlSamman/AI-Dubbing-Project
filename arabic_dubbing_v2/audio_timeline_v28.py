"""Offline measurement-only TTS versus original lesson timeline.

Audits existing manifest records, never generates, re-times or approves audio.
An actual synchronized video still requires a separate validated mixer.
"""
import math
from .common import clone, unit_times


def audit_existing_tts_manifest(prepared, manifest, *, warning_ratio=1.15, severe_ratio=1.4):
    if not (1 < warning_ratio < severe_ratio):raise ValueError('Invalid timing limits')
    master_hash=(prepared.get('translation_v2_metadata') or {}).get('source_master_sha256')
    candidate_hash=manifest.get('master_sha256')
    if master_hash and candidate_hash and candidate_hash!=master_hash:
        raise ValueError('Audio manifest belongs to a different source Master')
    matching='sha256_matched' if master_hash and candidate_hash else 'unverified_missing_source_hash'
    unit_map={u['id']:u for u in prepared['translation_units']}
    units=manifest.get('units') or {}
    if not isinstance(units,dict):raise ValueError('TTS manifest units must be a dictionary')
    records=[]
    for uid,rec in units.items():
        if uid not in unit_map:raise ValueError(f'Audio unit not in source: {uid}')
        if rec.get('unit_id')!=uid:raise ValueError('Audio record ID differs from key')
        raw=rec.get('actual_seconds')
        if type(raw) not in (float,int) or not math.isfinite(raw) or raw<=0:
            raise ValueError('Invalid actual TTS duration')
        start,end=unit_times(unit_map[uid])
        duration=end-start
        next_start=min((unit_times(u)[0] for u in prepared['translation_units']
                        if unit_times(u)[0]>=end and u['id']!=uid),default=None)
        allowed=duration if next_start is None else next_start-start
        factor=raw/duration
        if factor>=severe_ratio:timing='major_overrun_needs_edit_or_retake'
        elif factor>warning_ratio:timing='overrun_needs_review'
        elif raw<duration/warning_ratio:timing='underrun_needs_review'
        else:timing='approximately_fits_source_span'
        records.append({'unit_id':uid,'source_start_seconds':round(start,4),
                        'source_end_seconds':round(end,4),'source_span_seconds':round(duration,4),
                        'actual_audio_seconds':round(raw,4),'difference_seconds':round(raw-duration,4),
                        'required_playback_speed_ratio_if_forced':round(factor,4),
                        'available_seconds_until_next_source_start':round(allowed,4),
                        'overlaps_next_source_if_placed_at_start':bool(next_start is not None and raw>allowed),
                        'timing_status':timing,
                        'human_listening_verified':bool(rec.get('audio_listened_and_verified') is True),
                        'audio_approved':bool(rec.get('tts_approved') is True),
                        'audio_sha256':rec.get('audio_sha256'),
                        'hash_verified_against_file':False,
                        'suggestion':'Review text length, pacing and pauses; avoid automatic speed-up or silent cuts.'})
    return {'schema_version':'tts-timeline-audit-v2.8','manifest_source_sha256':candidate_hash,
            'measured_units':len(records),'records':records,
            'manifest_source_match_status':matching,
            'units_exceeding_source_span':sum(x['difference_seconds']>0 for x in records),
            'units_with_potential_next_start_overlap':sum(x['overlaps_next_source_if_placed_at_start'] for x in records),
            'all_listened_and_approved':bool(records) and all(x['human_listening_verified'] and x['audio_approved'] for x in records),
            'final_sync_approved':False,'audio_generated_in_this_audit':0,
            'source_timings_modified':False,
            'notes':['Recorded durations are metadata, not proof that the WAV files are available or uncorrupted.',
                     'Hash was not checked against WAV bytes by this metadata-only audit.',
                     'Playback ratio is advisory; avoid unsupervised time-stretching.']}