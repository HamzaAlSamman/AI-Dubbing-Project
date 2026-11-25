"""The translation step: [glossary] -> candidates -> gate -> guarded refine -> selection -> [polish] -> artifacts.

V2 (config section [language]): before translation an LLM builds the glossary of the video from the
whole transcript (terminology agent, cached write-once); generators get the glossary terms of each unit;
the gate rejects candidates that miss an approved term or use dialect words; the ranker and refiner get
the simple-MSA rules; an accepted line with stiff translation patterns gets a guarded polish edit
(dubbing adaptation) that must keep the meaning (MetricX) and pass the whole gate again.

Layout (all under `work_dir`, normally the lesson folder on Drive):
  cache/candidates/<generator>/<unit>[__<budget>].json   generations, shared by every run (write-once)
  cache/qe_<metric>.jsonl                                 QE scores cache
  cache/refine/<refiner>/<unit>__<input hash>.json        refine attempts (write-once)
  cache/glossary/<version>__<transcript hash>.json        LLM-built glossary of the video (write-once)
  cache/polish/<editor>/<unit>__<input hash>.json         polish attempts (write-once)
  runs/<run_id>/manifest.json                             config, hashes, git commit, models
  runs/<run_id>/translation_final.json                    the hand-off to the TTS stage
  runs/<run_id>/candidates_scored.csv                     every candidate with every measurement
  runs/<run_id>/metrics.json                              run metrics (also appended to registry.jsonl)
"""
import collections
import csv
import json
import statistics
from pathlib import Path

from . import __version__
from .common import git_commit, now_iso, now_stamp, read_json, sha256_json, write_json_once
from .gate import evaluate, only_duration_failed, rank_key
from .generators import (AuthenticationError, InvalidLLMOutput, PermanentLLMError, QuotaExhausted,
                         TransientLLMError)

# an LLM role (glossary, refine, rank, polish) that fails is stopped for the run; the run is degraded
LLM_STOP = (QuotaExhausted, TransientLLMError, PermanentLLMError, AuthenticationError)
# a model answer that is not valid JSON: that unit's step is skipped, nothing is cached, the run is degraded
LLM_SKIP = (InvalidLLMOutput,)
from .language import CourseMemory, Glossary, Language, build_glossary, term_contexts
from .contract import SCHEMA_VERSION, check_coverage, source_readiness, source_reference, validate_final
from .segment import prepare_sources
from .prompts import (LEGACY_KEY_VERSIONS, POLISH_SYSTEM, POLISH_VERSION, RANK_SYSTEM, RANK_V2_SYSTEM,
                      RANK_V2_VERSION, RANK_VERSION, REFINE_SYSTEM, REFINE_V3_SYSTEM, REFINE_V3_VERSION,
                      REFINE_VERSION, describe_issues, prompt_sha)
from .text import ar_tokens, clean_translation, tts_text
from .timing import letter_budget


def _budget(window, cfg):
    rate, b = cfg['voice']['letters_per_second'], cfg['budget']
    lo, hi = letter_budget(window, rate, low=b['low'], high=b['high'])
    return {'min': lo, 'max': hi, 'short': max(lo, int(hi * b['short_factor']))}


# ------------------------------------------------------------------ 1. candidates
def _language_inputs(task, language):
    """What a language-aware generator sees for this unit: style rules + glossary lines, term pairs."""
    if language is None:
        return {}
    pairs = language.glossary.hymt_pairs(task['source_text']) if language.glossary else []
    return {'language_block': language.prompt_block(task['source_text'], task), 'term_pairs': pairs}


def _candidate_path(work_dir, gen, unit_id, budget, lang_inputs=None):
    suffix = f'__{sha256_json(budget)[:8]}' if getattr(gen, 'uses_budget', True) else ''
    if getattr(gen, 'language_aware', False) and lang_inputs:
        suffix += f'__L{sha256_json(lang_inputs)[:8]}'        # a new glossary never reuses old outputs
    return Path(work_dir) / 'cache' / 'candidates' / gen.name / f'{unit_id}{suffix}.json'


def failure_path(path):
    """Failure ledger of one cached result: kept next to it, never mistaken for a result."""
    path = Path(path)
    return path.with_name(path.stem + '.failure.json')


def record_failure(path, kind, exc, *, generator=None, unit_id=None):
    """Append one failed attempt to the unit's failure ledger (atomic replace). Results are never touched."""
    import os
    fp = failure_path(path)
    rec = read_json(fp) if fp.exists() else {'unit_id': unit_id, 'generator': generator, 'attempts': 0,
                                             'history': []}
    rec['attempts'] += 1
    rec['kind'], rec['last_error'], rec['resolved_at'] = kind, str(exc)[:300], None
    rec['history'].append({'at': now_iso(), 'kind': kind, 'error': str(exc)[:200]})
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_name(fp.name + '.tmp')
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, fp)
    return rec


def _resolve_failure(path):
    import os
    fp = failure_path(path)
    if fp.exists():
        rec = read_json(fp)
        rec['resolved_at'] = now_iso()
        tmp = fp.with_name(fp.name + '.tmp')
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, fp)


def generate_candidates(tasks, windows, generators, cfg, work_dir, log=print, language=None, report=None):
    """Resumable: units with a cached result are skipped.

    Only real model answers are cached as results: 'ok', or 'empty' when the model gave a valid answer
    without usable text. Failures go to a failure ledger next to the result path (failure_path), never
    to the result path:
      quota / request cap / authentication  the generator stops; the remaining units are deferred
      TransientLLMError   the unit is deferred; after `models.max_consecutive_failures` in a row the
                          generator stops for this run
      PermanentLLMError   blocked after 1 attempt          } until `models.retry_errors = true` (after the
      InvalidLLMOutput    blocked after `models.max_unit_attempts` } cause is fixed): then one more attempt
    A later success writes the result and marks the ledger resolved; earlier results are never deleted.
    `report` receives {'deferred', 'errors', 'invalid', 'blocked'}: {gen: [units]}."""
    import os
    stopped = {}
    report = report if report is not None else {}
    deferred, errors = report.setdefault('deferred', {}), report.setdefault('errors', {})
    invalid, blocked = report.setdefault('invalid', {}), report.setdefault('blocked', {})
    calls = report.setdefault('model_calls', {})
    mcfg = cfg.get('models', {})
    max_fail = int(mcfg.get('max_consecutive_failures', 3))
    max_unit = int(mcfg.get('max_unit_attempts', 3))
    retry_errors = bool(mcfg.get('retry_errors', False))
    limits = {'permanent': 1, 'invalid_output': max_unit}
    for gen in generators:
        todo = []
        for t in tasks:
            budget = _budget(windows[t['unit_id']], cfg)
            lang_inputs = _language_inputs(t, language)
            path = _candidate_path(work_dir, gen, t['unit_id'], budget, lang_inputs)
            if path.exists():
                rec = read_json(path)
                if rec.get('status') != 'error':
                    continue
                if not retry_errors:                     # an error record of an earlier version: kept as is
                    continue
                fp = failure_path(path)                  # moved to the ledger (not deleted), then retried
                if not fp.exists():
                    os.replace(path, fp)
                    led = read_json(fp)
                    led.update({'attempts': 1, 'kind': 'permanent', 'last_error': led.get('error'), 'history': []})
                    fp.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding='utf-8')
                else:
                    os.replace(path, fp.with_name(fp.stem + '.legacy.json'))
            fp = failure_path(path)
            if fp.exists() and not retry_errors:
                led = read_json(fp)
                lim = limits.get(led.get('kind'))
                if not led.get('resolved_at') and lim is not None and led.get('attempts', 0) >= lim:
                    key = errors if led['kind'] == 'permanent' else invalid
                    key.setdefault(gen.name, []).append(t['unit_id'])
                    blocked.setdefault(gen.name, []).append(t['unit_id'])
                    continue
            todo.append(({**t, **lang_inputs}, budget, path))
        log(f'[candidates] {gen.name}: {len(todo)} to generate')
        consecutive = 0
        for i, (t, budget, path) in enumerate(todo, 1):
            task = {**t, 'slot_s': windows[t['unit_id']].slot_s, 'window_s': windows[t['unit_id']].window_s}
            calls[gen.name] = calls.get(gen.name, 0) + 1
            try:
                items = gen.generate(task, budget)
            except (QuotaExhausted, AuthenticationError) as exc:
                kind = 'authentication' if isinstance(exc, AuthenticationError) else 'quota'
                stopped[gen.name] = f'{kind}: {str(exc)[:200]}'
                deferred.setdefault(gen.name, []).extend(x[0]['unit_id'] for x in todo[i - 1:])
                log(f'[candidates] {gen.name}: {kind} stop after {i - 1} units; fix the cause and rerun to resume')
                break
            except TransientLLMError as exc:
                record_failure(path, 'transient', exc, generator=gen.name, unit_id=t['unit_id'])
                deferred.setdefault(gen.name, []).append(t['unit_id'])
                consecutive += 1
                if consecutive >= max_fail:
                    stopped[gen.name] = f'{consecutive} consecutive service failures: {str(exc)[:150]}'
                    deferred[gen.name].extend(x[0]['unit_id'] for x in todo[i:])
                    log(f'[candidates] {gen.name}: service failing, stopped; {len(deferred[gen.name])} units deferred')
                    break
                continue
            except PermanentLLMError as exc:
                record_failure(path, 'permanent', exc, generator=gen.name, unit_id=t['unit_id'])
                errors.setdefault(gen.name, []).append(t['unit_id'])
                continue
            except InvalidLLMOutput as exc:
                record_failure(path, 'invalid_output', exc, generator=gen.name, unit_id=t['unit_id'])
                invalid.setdefault(gen.name, []).append(t['unit_id'])
                continue
            consecutive = 0
            write_json_once(path, {'unit_id': t['unit_id'], 'generator': gen.name, 'budget': budget,
                                   'items': items, 'status': 'ok' if items else 'empty',
                                   'prompt_sha': getattr(gen, 'prompt_sha', None),
                                   'generator_info': gen.describe() if hasattr(gen, 'describe') else None,
                                   'created_at': now_iso()})
            _resolve_failure(path)
            if i % 10 == 0 or i == len(todo):
                log(f'  {gen.name}: {i}/{len(todo)}')
        if hasattr(gen, 'close'):
            gen.close()
    return stopped


def load_candidates(tasks, windows, generators, cfg, work_dir, language=None, report=None):
    """{unit_id: [{'cand_id','generator','variant','text','raw_text'}]}; identical texts merged.
    `report['errors']` collects units whose cached record is a refused request ('error').
    A record written with another prompt text than the generator's current one is stale: not used,
    listed in `report['stale']`. Records without a prompt hash (V1 caches) are used and counted 'legacy'."""
    out = {}
    report = report if report is not None else {}
    errors, stale, legacy = report.setdefault('errors', {}), report.setdefault('stale', {}), report.setdefault('legacy', {})
    for t in tasks:
        budget = _budget(windows[t['unit_id']], cfg)
        seen, cands = {}, []
        for gen in generators:
            path = _candidate_path(work_dir, gen, t['unit_id'], budget, _language_inputs(t, language))
            if not path.exists():
                continue
            rec = read_json(path)
            current = getattr(gen, 'prompt_sha', None)
            if rec.get('prompt_sha') and current and rec['prompt_sha'] != current:
                stale.setdefault(gen.name, []).append(t['unit_id'])
                continue
            if not rec.get('prompt_sha'):
                legacy[gen.name] = legacy.get(gen.name, 0) + 1
            if rec.get('status') == 'error' and t['unit_id'] not in errors.get(gen.name, []):
                errors.setdefault(gen.name, []).append(t['unit_id'])
            for item in rec['items']:
                text = clean_translation(item['text'])
                if not text:
                    continue
                if text in seen:
                    seen[text]['also_from'].append(gen.name)
                    continue
                c = {'cand_id': f'{gen.name}:{item["variant"]}', 'generator': gen.name, 'variant': item['variant'],
                     'text': text, 'raw_text': item['text'], 'also_from': [],
                     'prompt_sha': rec.get('prompt_sha'), 'generator_info': rec.get('generator_info')}
                seen[text] = c
                cands.append(c)
        out[t['unit_id']] = cands
    return out


# ------------------------------------------------------------------ 2. scoring
def consensus_allowlist(cands_by_unit, misspelled, extra=()):
    """Words the spell checker rejects but that >= 2 different generators chose independently
    are treated as accepted technical vocabulary (e.g. بكسلات); plus the configured list."""
    users = collections.defaultdict(set)
    for cands in cands_by_unit.values():
        for c in cands:
            for gen in [c['generator'], *c['also_from']]:
                for w in set(ar_tokens(c['text'])) & misspelled:
                    users[w].add(gen.split('__')[0])          # provider/model family, not prompt variant
    return frozenset(w for w, g in users.items() if len(g) >= 2) | frozenset(extra)


def score_all(tasks, cands_by_unit, windows, cfg, qe, speller, language=None):
    src = {t['unit_id']: t['source_text'] for t in tasks}
    pairs = [(src[u], c['text']) for u, cs in cands_by_unit.items() for c in cs]
    scores = dict(zip(pairs, qe.score(pairs))) if pairs else {}
    words = {w for cs in cands_by_unit.values() for c in cs for w in ar_tokens(c['text'])}
    misspelled = speller.misspelled(sorted(words))
    allow = consensus_allowlist(cands_by_unit, misspelled, cfg['gate'].get('spelling_allowlist', ()))
    for u, cs in cands_by_unit.items():
        for c in cs:
            c['eval'] = evaluate(c['text'], src[u], windows[u], qe=scores[(src[u], c['text'])],
                                 misspelled_words=misspelled, cfg=cfg, allowlist=allow, language=language)
    return misspelled, allow


# ------------------------------------------------------------------ 3. guarded refine
def refine_unit(task, window, cands, cfg, client, qe, misspelled_fn, allow, work_dir, log=print, language=None):
    """Ask an LLM for one corrected translation when no candidate passed the gate.

    Guard: the edit is kept only if it passes the full gate AND, when the best draft failed on
    duration alone, its QE is not worse than that draft by more than `max_qe_worse`."""
    rcfg = cfg['refine']
    budget = _budget(window, cfg)
    valid = [c for c in cands if not any(r.startswith('invalid') for r in c['eval']['reasons'])] or cands
    drafts = sorted(valid, key=lambda c: (c['eval']['qe'] if c['eval']['qe'] is not None else 99))[:rcfg['drafts_shown']]
    if not drafts:
        return None, []
    best = drafts[0]
    payload = {'english_source': task['source_text'], 'slot_seconds': round(window.slot_s, 1),
               'target_letters_min': budget['min'], 'target_letters_max': budget['max'],
               'drafts': [{'arabic': c['text'], 'letters': c['eval']['letters'],
                           'problems': describe_issues(c['eval']['reasons'], c['eval']['misspelled'])} for c in drafts]}
    system, version = REFINE_SYSTEM, REFINE_VERSION
    if language is not None:
        payload.update(language.prompt_block(task['source_text'], task))
        system, version = REFINE_V3_SYSTEM, REFINE_V3_VERSION
    name = f'{client.provider}__{client.model.replace("/", "_").replace(":", "_")}__{version}'
    key = [payload, rcfg] if version in LEGACY_KEY_VERSIONS else [payload, rcfg, prompt_sha(system)]
    path = Path(work_dir) / 'cache' / 'refine' / name / f'{task["unit_id"]}__{sha256_json(key)[:10]}.json'
    if path.exists():
        rec = read_json(path)
    else:
        attempts, note = [], ''
        for attempt in range(rcfg['max_attempts']):
            obj = client.chat_json(system, json.dumps(payload, ensure_ascii=False) + note)  # invalid: not cached
            text = clean_translation(obj.get('natural_ar') if isinstance(obj.get('natural_ar'), str) else '')
            if not text:
                attempts.append({'attempt': attempt, 'text': '', 'verdict': 'no_text'})
                continue
            q = qe.score([(task['source_text'], text)])[0]
            ev = evaluate(text, task['source_text'], window, qe=q, misspelled_words=misspelled_fn(ar_tokens(text)),
                          cfg=cfg, allowlist=allow, language=language)
            guard_ok = not (only_duration_failed(best['eval']) and best['eval']['qe'] is not None
                            and q > best['eval']['qe'] + rcfg['max_qe_worse'])
            verdict = 'accepted' if ev['accepted'] and guard_ok else ('meaning_guard' if ev['accepted'] else 'rejected')
            attempts.append({'attempt': attempt, 'text': text, 'eval': ev, 'verdict': verdict})
            if verdict == 'accepted':
                break
            note = ('\nYour previous answer: ' + text + '\nIts problems: ' +
                    '; '.join(describe_issues(ev['reasons'], ev['misspelled']) or ['meaning changed too much']) +
                    '\nWrite a better one.')
        rec = {'unit_id': task['unit_id'], 'refiner': name, 'payload': payload, 'attempts': attempts,
               'created_at': now_iso()}
        write_json_once(path, rec)
    accepted = [a for a in rec['attempts'] if a.get('verdict') == 'accepted']
    if not accepted:
        return None, rec['attempts']
    a = accepted[0]
    refined = {'cand_id': f'{name}:refine', 'generator': name, 'variant': 'refine', 'text': a['text'],
               'raw_text': a['text'], 'also_from': [], 'eval': a['eval'], 'refined_from': best['cand_id'],
               'prompt_sha': prompt_sha(system),
               'generator_info': {'name': name, 'kind': 'llm', 'model': client.model, 'prompt_version': version}}
    return refined, rec['attempts']


# ------------------------------------------------------------------ 3b. LLM re-ranking of accepted candidates
def rank_unit(task, cands, cfg, client, work_dir, language=None):
    """Blind GEMBA-style scoring of the top accepted candidates; returns the chosen candidate.

    MetricX shortlists (gate + top_k by QE); the LLM judges sense errors MetricX misses (wrong word
    sense, agreement, dropped phrases). Candidate sources are hidden and the order is shuffled."""
    import random
    rcfg = cfg['rank']
    pool = sorted(cands, key=lambda c: rank_key(c['eval']))[:rcfg['top_k']]
    if len(pool) < 2:
        return pool[0] if pool else None, None
    rng = random.Random(sha256_json([task['unit_id'], [c['text'] for c in pool]]))
    shown = pool[:]
    rng.shuffle(shown)
    payload = {'english': task['source_text'],
               'candidates': [{'id': i + 1, 'arabic': c['text']} for i, c in enumerate(shown)]}
    system, version = RANK_SYSTEM, RANK_VERSION
    if language is not None:
        payload.update(language.prompt_block(task['source_text'], task))
        system, version = RANK_V2_SYSTEM, RANK_V2_VERSION
    name = f'{client.provider}__{client.model.replace("/", "_").replace(":", "_")}__{version}'
    key = payload if version in LEGACY_KEY_VERSIONS else [payload, prompt_sha(system)]
    path = Path(work_dir) / 'cache' / 'rank' / name / f'{task["unit_id"]}__{sha256_json(key)[:10]}.json'
    if path.exists():
        rec = read_json(path)
    else:
        reply = client.chat_json(system, json.dumps(payload, ensure_ascii=False))   # InvalidLLMOutput: not cached
        rec = {'unit_id': task['unit_id'], 'ranker': name, 'payload': payload, 'reply': reply, 'created_at': now_iso()}
        write_json_once(path, rec)
    scores = {}
    for s in (rec['reply'] or {}).get('scores') or []:
        try:
            scores[int(s['id'])] = float(s['adequacy']) * rcfg['adequacy_weight'] + float(s['fluency'])
        except (KeyError, TypeError, ValueError):
            continue
    if len(scores) < len(shown):
        return pool[0], rec                          # unusable reply: keep the MetricX choice
    best_i = max(range(len(shown)), key=lambda i: (scores[i + 1], -rank_key(shown[i]['eval'])[0]))
    chosen = shown[best_i]
    chosen['rank_scores'] = {shown[i]['cand_id']: scores[i + 1] for i in range(len(shown))}
    return chosen, rec


# ------------------------------------------------------------------ 3c. guarded polish (dubbing adaptation)
def polish_unit(task, window, chosen, cfg, client, qe, misspelled_fn, allow, work_dir, language):
    """Minimal language edit of an accepted line that still has style findings (stiff patterns).

    Kept only if the edit passes the whole gate, has fewer style findings, and its MetricX QE is not
    worse than the original line by more than `polish.max_qe_worse` (meaning guard)."""
    pcfg = cfg.get('polish', {})
    found = chosen['eval']['language']
    if not language.needs_polish(found):
        return None, None
    budget = _budget(window, cfg)
    payload = {'english_source': task['source_text'], 'arabic': chosen['text'],
               'target_letters_min': budget['min'], 'target_letters_max': budget['max'],
               'problems': language.issue_hints(task['source_text'], found),
               **language.prompt_block(task['source_text'], task)}
    name = f'{client.provider}__{client.model.replace("/", "_").replace(":", "_")}__{POLISH_VERSION}'
    key = [payload, pcfg, prompt_sha(POLISH_SYSTEM)]
    path = Path(work_dir) / 'cache' / 'polish' / name / f'{task["unit_id"]}__{sha256_json(key)[:10]}.json'
    if path.exists():
        rec = read_json(path)
    else:
        obj = client.chat_json(POLISH_SYSTEM, json.dumps(payload, ensure_ascii=False))  # invalid: not cached
        text = clean_translation(obj.get('natural_ar') if isinstance(obj.get('natural_ar'), str) else '')
        attempt = {'text': text, 'verdict': 'no_text'}
        if text and text != chosen['text']:
            q = qe.score([(task['source_text'], text)])[0]
            ev = evaluate(text, task['source_text'], window, qe=q, misspelled_words=misspelled_fn(ar_tokens(text)),
                          cfg=cfg, allowlist=allow, language=language)
            base_q = chosen['eval']['qe'] if chosen['eval']['qe'] is not None else 99.0
            if not ev['accepted']:
                verdict = 'rejected_by_gate'
            elif q > base_q + pcfg.get('max_qe_worse', 0.5):
                verdict = 'meaning_guard'
            elif ev['style_flags'] >= chosen['eval']['style_flags']:
                verdict = 'no_improvement'
            else:
                verdict = 'accepted'
            attempt = {'text': text, 'eval': ev, 'verdict': verdict}
        rec = {'unit_id': task['unit_id'], 'editor': name, 'payload': payload, 'attempt': attempt,
               'created_at': now_iso()}
        write_json_once(path, rec)
    a = rec['attempt']
    if a.get('verdict') != 'accepted':
        return None, rec
    return {'cand_id': f'{name}:polish', 'generator': name, 'variant': 'polish', 'text': a['text'],
            'raw_text': a['text'], 'also_from': [], 'eval': a['eval'], 'polished_from': chosen['cand_id'],
            'prompt_sha': prompt_sha(POLISH_SYSTEM),
            'generator_info': {'name': name, 'kind': 'llm', 'model': client.model, 'prompt_version': POLISH_VERSION},
            'rank_scores': chosen.get('rank_scores')}, rec


# ------------------------------------------------------------------ 4. run
def run(tasks, windows, cfg, work_dir, *, generators, qe, speller, refine_client=None, rank_client=None,
        glossary_client=None, course_memory=None, course_memory_snapshot=None, video_id=None, repo_dir=None,
        run_label='', environment=None, source_timing=None, log=print):
    """glossary_client: LLM that builds the glossary of the video when [language] glossary = "auto..."
    (defaults to the refine client).
    course_memory: path of the course's terminology memory (JSON shared by all videos of a course). The run
    uses a snapshot of its verified terms (saved in the run folder, hash in the manifest) and records this
    video's proposals only after a complete, non-degraded run. video_id names this video there.
    course_memory_snapshot: a saved snapshot file; replays exactly that prior and writes no memory.
    source_timing: {unit_id: {'start', 'end', 'words'}} of the ORIGINAL units (io.load_source_timing); copied
    unchanged into every output record (schemas/translation_final.schema.json)."""
    work_dir = Path(work_dir)
    source_tasks_hash = sha256_json([{k: t[k] for k in ('unit_id', 'source_text')} for t in tasks])
    source_units_count = len(tasks)
    source_timing = source_timing or {}
    source_refs = {t['unit_id']: source_reference(t, source_timing.get(t['unit_id'])) for t in tasks}
    tasks, windows, source_prep = prepare_sources(
        tasks, windows, cfg, timings={u: (v['start'], v['end']) for u, v in source_timing.items()
                                      if v.get('start') is not None and v.get('end') is not None})
    auto_glossary, glossary_path, subject = None, None, None
    stopped, gen_report = {}, {}
    memory = CourseMemory(course_memory) if course_memory else None
    snapshot = None
    if course_memory_snapshot:
        snapshot = read_json(course_memory_snapshot)                   # replay: read-only
        memory = None
    elif memory is not None:
        snapshot = memory.snapshot()
    lc = cfg.get('language') or {}
    if str(lc.get('glossary', '')).startswith('auto'):
        builder = glossary_client or refine_client
        if builder is not None:
            try:
                auto_glossary, glossary_path, subject = build_glossary(
                    tasks, builder, work_dir / 'cache', prior_terms=snapshot['verified'] if snapshot else (),
                    rejected_terms=(snapshot or {}).get('rejected', ()), log=log)
            except LLM_STOP + LLM_SKIP as exc:
                stopped['glossary'] = str(exc)[:200]
                log(f'[glossary] service stopped, no automatic glossary this run: {str(exc)[:120]}')
        else:
            log('[glossary] no LLM client: the automatic glossary is skipped')
        if auto_glossary is None and snapshot and snapshot['verified']:
            auto_glossary = Glossary.from_terms(snapshot['verified'], 'course_memory')   # earlier decisions
    language = Language.from_config(cfg, auto_glossary=auto_glossary)
    if language is not None:
        language.subject = subject
    cfg_hash = sha256_json(cfg)
    run_id = f'tr_{now_stamp()}_{cfg_hash[:8]}' + (f'_{run_label}' if run_label else '')
    run_dir = work_dir / 'runs' / run_id
    snapshot_sha = None
    if snapshot is not None:                     # the exact memory this run used, for replay
        import hashlib
        body = json.dumps(snapshot, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
        snapshot_sha = hashlib.sha256(body).hexdigest()
        snap_path = run_dir / 'course_memory_snapshot.json'
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        if not snap_path.exists():
            snap_path.write_bytes(body)
    tasks_hash = sha256_json([{k: t[k] for k in ('unit_id', 'source_text')} for t in tasks])
    manifest = {'run_id': run_id, 'created_at': now_iso(), 'package_version': __version__,
                'git_commit': git_commit(repo_dir) if repo_dir else None, 'config': cfg, 'config_sha256': cfg_hash,
                'tasks_sha256': tasks_hash, 'source_tasks_sha256': source_tasks_hash,
                'prepared_tasks_sha256': tasks_hash, 'source_units': source_units_count,
                'units': len(tasks), 'generators': [g.name for g in generators],
                'generator_details': [g.describe() if hasattr(g, 'describe') else {'name': g.name} for g in generators],
                'qe_metric': qe.name, 'spell_checker': speller.name,
                'refiner': None if refine_client is None else f'{refine_client.provider}:{refine_client.model}',
                'ranker': None if rank_client is None else f'{rank_client.provider}:{rank_client.model}',
                'windows': {u: w.as_dict() for u, w in windows.items()}, 'human_review': 'none',
                'source_preparation': source_prep,
                'environment': environment,             # preflight report: package versions, GPU, python
                'language': None if language is None else {
                    'name': language.name, 'modes': language.modes,
                    'glossary_terms': len(language.glossary.terms) if language.glossary else 0,
                    'glossary_file': str(glossary_path) if glossary_path else None,
                    'video_subject': subject,
                    'course_memory': str(memory.path) if memory else None,
                    'course_memory_snapshot_sha256': snapshot_sha,
                    'course_memory_revision': snapshot.get('revision') if snapshot else None,
                    'course_memory_terms': len(snapshot['verified']) if snapshot else None}}
    write_json_once(run_dir / 'run_started.json', manifest)       # the complete manifest is written at the end
    if language is not None and language.glossary is not None:
        write_json_once(run_dir / 'glossary.json', language.glossary.as_dict())

    stopped.update(generate_candidates(tasks, windows, generators, cfg, work_dir, log, language=language,
                                       report=gen_report))
    cands = load_candidates(tasks, windows, generators, cfg, work_dir, language=language, report=gen_report)
    misspelled, allow = score_all(tasks, cands, windows, cfg, qe, speller, language=language)
    cache_words = dict.fromkeys(misspelled)

    def misspelled_fn(words):
        new = [w for w in words if w not in cache_words and w not in allow]
        found = speller.misspelled(new) if new else set()
        for w in new:
            cache_words[w] = None
        return set(misspelled) | found

    final, refine_log, rank_changed = [], {}, 0
    role_invalid = gen_report.setdefault('role_invalid', {})          # LLM roles: answers that were not JSON
    spoken_version = cfg['voice'].get('spoken_version', 'v1')
    suspect_units = {}
    for sus in source_prep.get('suspected_fragments', []):
        for uid in sus['units']:
            suspect_units.setdefault(uid, []).append(sus['reason'])

    def contract_fields(t, w):
        """Fields of the TTS contract: original units with timings, derived-unit mapping, source flags."""
        parts = t.get('merged_from') or [t['unit_id']]
        refs = [source_refs[p] for p in parts]
        return {'derived': bool(t.get('merged_from')), 'merged_from': t.get('merged_from'), 'source_units': refs,
                'source_flags': {'spelled_digits': [s for p in parts
                                                    for s in source_prep.get('spelled_digit_flags', {}).get(p, [])],
                                 'suspected_fragment': [r for p in parts for r in suspect_units.get(p, [])]},
                'start': refs[0]['start'], 'end': refs[-1]['end'], 'spoken_version': spoken_version}

    def provenance(c):
        return {'cand_id': c.get('cand_id'), 'generator': c.get('generator'), 'prompt_sha': c.get('prompt_sha'),
                'generator_info': c.get('generator_info'), 'refined_from': c.get('refined_from'),
                'polished_from': c.get('polished_from')}
    mt_generators = {g.name for g in generators if not getattr(g, 'uses_budget', True)}

    def accurate_of(cs):
        """Best MetricX line among the plain MT candidates (no length target): the accurate reference
        version kept next to the dubbing line (FigJam: translation.accurate / dub / final)."""
        mt = [c for c in cs if c['generator'] in mt_generators and c['eval']['qe'] is not None
              and not any(r.startswith('invalid') for r in c['eval']['reasons'])]
        return min(mt, key=lambda c: c['eval']['qe'])['text'] if mt else None
    use_rank = rank_client is not None and cfg.get('rank', {}).get('enabled')
    for t in tasks:
        u, w = t['unit_id'], windows[t['unit_id']]
        cs = cands[u]
        accepted = sorted((c for c in cs if c['eval']['accepted']), key=lambda c: rank_key(c['eval']))
        status, chosen = 'accepted', (accepted[0] if accepted else None)
        if use_rank and len(accepted) >= 2:
            try:
                ranked, _ = rank_unit(t, accepted, cfg, rank_client, work_dir, language)
            except LLM_STOP as exc:
                stopped['rank'] = str(exc)[:200]
                use_rank, ranked = False, None
            except LLM_SKIP:
                role_invalid.setdefault('rank', []).append(u)
                ranked = None
            if ranked is not None:
                rank_changed += ranked is not chosen
                chosen = ranked
        if chosen is None and refine_client is not None and cfg['refine']['enabled'] and cs:
            try:
                chosen, refine_log[u] = refine_unit(t, w, cs, cfg, refine_client, qe, misspelled_fn, allow,
                                                    work_dir, log, language)
            except LLM_STOP as exc:
                stopped['refine'] = str(exc)[:200]
                chosen = None
            except LLM_SKIP:
                role_invalid.setdefault('refine', []).append(u)
                chosen = None
            if chosen is not None:
                cs.append(chosen)
                status = 'accepted_after_refine'
        if chosen is None:
            valid = [c for c in cs if not any(r.startswith('invalid') for r in c['eval']['reasons'])] or cs
            if not valid:
                final.append({'unit_id': u, 'source_text': t['source_text'], 'status': 'missing', 'natural_ar': None,
                              'accurate_ar': None, 'tts_text': None, 'slot_s': w.slot_s, 'window_s': w.window_s,
                              'predicted_s': None, 'flags': ['no_candidate'], **contract_fields(t, w),
                              'provenance': provenance({})})
                continue
            chosen = min(valid, key=lambda c: (len(c['eval']['reasons']), rank_key(c['eval'])))
            status = 'needs_review'
            # still hand the reviewer the best draft: blind LLM ranking among valid, non-overlong drafts
            fit = [c for c in valid if not any(r.startswith('invalid') or r == 'duration_too_long'
                                               for r in c['eval']['reasons'])]
            if use_rank and len(fit) >= 2:
                try:
                    ranked, _ = rank_unit(t, fit, cfg, rank_client, work_dir, language)
                    chosen = ranked or chosen
                except LLM_STOP as exc:
                    stopped['rank'] = str(exc)[:200]
                    use_rank = False
                except LLM_SKIP:
                    role_invalid.setdefault('rank', []).append(u)
        use_polish = (language is not None and refine_client is not None and cfg.get('polish', {}).get('enabled')
                      and 'polish' not in stopped)
        if use_polish and status != 'needs_review':
            try:
                polished, _ = polish_unit(t, w, chosen, cfg, refine_client, qe, misspelled_fn, allow, work_dir,
                                          language)
            except LLM_STOP as exc:
                stopped['polish'] = str(exc)[:200]
                polished = None
            except LLM_SKIP:
                role_invalid.setdefault('polish', []).append(u)
                polished = None
            if polished is not None:
                cs.append(polished)
                chosen = polished
        ev = chosen['eval']
        flags = []
        if w.slot_s < cfg['gate']['tiny_slot_s'] and ev['zone'] in ('too_long', 'speedup'):
            flags.append('tiny_slot_consider_merging_with_neighbour')
        if ev['zone'] == 'speedup':
            flags.append(f'needs_speedup_x{ev["ratio_window"]:.2f}')
        final.append({'unit_id': u, 'source_text': t['source_text'], 'status': status,
                      'natural_ar': chosen['text'],
                      'tts_text': tts_text(chosen['text'], version=spoken_version),
                      'selected_from': chosen['cand_id'], 'refined_from': chosen.get('refined_from'),
                      'polished_from': chosen.get('polished_from'), 'language': ev.get('language'),
                      'accurate_ar': accurate_of(cs), **contract_fields(t, w), 'provenance': provenance(chosen),
                      'qe': ev['qe'], 'spelling_rate': ev['spelling_rate'], 'letters': ev['letters'],
                      'predicted_s': ev['predicted_s'], 'slot_s': w.slot_s, 'window_s': w.window_s,
                      'ratio_slot': ev['ratio_slot'], 'ratio_window': ev['ratio_window'], 'zone': ev['zone'],
                      'reasons': ev['reasons'], 'flags': flags, 'n_candidates': len(cs),
                      'rank_scores': chosen.get('rank_scores')})

    metrics = summarize(final, cands, stopped, refine_client, allow)
    status = run_status(stopped, gen_report, cfg, language_requested=bool(lc), glossary_built=glossary_path is not None,
                        final=final)
    metrics.update({'degraded': status['degraded'], 'infrastructure': status['infrastructure'],
                    'quality_gate': status['quality'], 'model_calls': status['model_calls'],
                    'degraded_reasons': status['reasons'],
                    'deferred_units': status['deferred_units'], 'error_units': status['error_units']})
    manifest['status'] = status
    if memory is not None and glossary_path is not None:
        if status['degraded']:
            log('[glossary] run degraded: the course memory is NOT updated')
            manifest['language']['course_memory_update'] = 'skipped: degraded run'
        else:
            proposed = read_json(glossary_path)['terms']
            upd = memory.record_video(video_id or work_dir.parent.name, proposed,
                                      contexts=term_contexts(proposed, tasks),
                                      meta={'glossary_version': read_json(glossary_path)['version'],
                                            'builder': read_json(glossary_path).get('builder'),
                                            'run_id': run_id, 'subject': subject})
            manifest['language']['course_memory_update'] = upd
            log(f'[glossary] course memory {memory.path}: {upd}')
    metrics['rank_changed_choice'] = rank_changed
    metrics['llm_calls_rank'] = getattr(rank_client, 'calls', 0) if rank_client else 0
    manifest['source_readiness'] = source_readiness(source_refs)
    write_json_once(run_dir / 'manifest.json', manifest)
    _write_outputs(run_dir, final, cands, metrics, tasks, manifest, refine_log, source_refs)
    _append_registry(work_dir / 'registry.jsonl', manifest, metrics)
    if cfg.get('tracking', {}).get('mlflow'):
        _log_mlflow(cfg, manifest, metrics, run_dir, log)
    log(f'[run] {run_id}: accepted {metrics["accepted"]}/{metrics["units"]} ({metrics["accepted_pct"]}%) | '
        f'QE mean {metrics["qe_mean"]} | refine accepted {metrics["accepted_after_refine"]} | '
        f'needs review {metrics["needs_review_units"]}')
    return run_dir, final, metrics


def summarize(final, cands, stopped, refine_client, allow):
    n = len(final)
    ok = [f for f in final if f['status'] in ('accepted', 'accepted_after_refine')]
    qes = [f['qe'] for f in final if f.get('qe') is not None]
    return {
        'units': n, 'accepted': len(ok), 'accepted_pct': round(100 * len(ok) / max(1, n), 1),
        'accepted_without_refine': sum(f['status'] == 'accepted' for f in final),
        'accepted_after_refine': sum(f['status'] == 'accepted_after_refine' for f in final),
        'needs_review_units': [f['unit_id'] for f in final if f['status'] in ('needs_review', 'missing')],
        'qe_mean': round(statistics.mean(qes), 3) if qes else None,
        'qe_median': round(statistics.median(qes), 3) if qes else None,
        'zones': dict(collections.Counter(f.get('zone') for f in final)),
        'picked_from': dict(collections.Counter((f.get('selected_from') or 'none').split(':')[0] for f in final)),
        'candidates_per_unit': round(statistics.mean(len(c) for c in cands.values()), 2) if cands else 0,
        'llm_calls_refine': getattr(refine_client, 'calls', 0) if refine_client else 0,
        'spelling_allowlist_size': len(allow), 'stopped_early': stopped,
        'flags': dict(collections.Counter(fl.split('_x')[0] for f in final for fl in f.get('flags', []))),
        **language_metrics(final),
        'merged_units': sum(bool(f.get('merged_from')) for f in final),
    }


def run_status(stopped, gen_report, cfg, *, language_requested, glossary_built, final=None):
    """Two separate questions.
    infrastructure: 'degraded' when a component did not do its work for every unit (a generator or an LLM
      role stopped: quota, request cap, authentication, service failure; units deferred, refused, blocked or
      answered with invalid JSON; stale cache; the automatic glossary of a V2 config not built). Degraded
      runs are not full runs: they are not used in comparisons as such.
    quality: how many lines passed the quality gate. A complete run with lines that failed the gate is
      NOT degraded and stays in the evaluation; its rejected lines are part of its result."""
    deferred = {g: sorted(set(u)) for g, u in (gen_report.get('deferred') or {}).items() if u}
    errors = {g: sorted(set(u)) for g, u in (gen_report.get('errors') or {}).items() if u}
    invalid = {g: sorted(set(u)) for g, u in (gen_report.get('invalid') or {}).items() if u}
    role_invalid = {g: sorted(set(u)) for g, u in (gen_report.get('role_invalid') or {}).items() if u}
    reasons = [f'stopped:{k}: {v[:120]}' for k, v in stopped.items()]
    reasons += [f'deferred:{g}:{len(u)} units' for g, u in deferred.items()]
    reasons += [f'refused:{g}:{len(u)} units' for g, u in errors.items()]
    reasons += [f'invalid_output:{g}:{len(u)} units' for g, u in invalid.items()]
    reasons += [f'invalid_output:{g}:{len(u)} units' for g, u in role_invalid.items()]
    stale = {g: u for g, u in (gen_report.get('stale') or {}).items() if u}
    reasons += [f'stale cache:{g}:{len(u)} units' for g, u in stale.items()]
    lc = cfg.get('language') or {}
    if language_requested and str(lc.get('glossary', '')).startswith('auto') and not glossary_built:
        reasons.append('glossary:not built')
    quality = None
    if final is not None:
        quality = {'units': len(final),
                   'passed_gate': sum(f['status'] in ('accepted', 'accepted_after_refine') for f in final),
                   'failed_gate': sum(f['status'] == 'needs_review' for f in final),
                   'no_candidate': sum(f['status'] == 'missing' for f in final)}
    return {'degraded': bool(reasons), 'infrastructure': 'degraded' if reasons else 'complete',
            'reasons': reasons, 'deferred_units': deferred, 'error_units': errors, 'invalid_units': invalid,
            'role_invalid_units': role_invalid, 'blocked_units': gen_report.get('blocked') or {},
            'model_calls': gen_report.get('model_calls') or {}, 'quality': quality, 'finished_at': now_iso()}


def language_metrics(final):
    """Language findings of the delivered lines (all zero / absent for V1 runs)."""
    langs = [f.get('language') for f in final if f.get('language')]
    if not langs:
        return {}
    return {'units_term_missing': sum(bool(x['terms']) for x in langs),
            'units_dialect': sum(bool(x['dialect']) for x in langs),
            'units_stiff_patterns': sum(bool(x['calques']) for x in langs),
            'stiff_patterns_total': sum(len(x['calques']) for x in langs),
            'units_hard_to_follow': sum(bool(x.get('simplicity')) for x in langs),
            'polished_units': sum(bool(f.get('polished_from')) for f in final)}


def _write_outputs(run_dir, final, cands, metrics, tasks, manifest, refine_log, source_refs=None):
    doc = {'schema_version': SCHEMA_VERSION, 'run_id': manifest['run_id'],
           'config_sha256': manifest['config_sha256'], 'tasks_sha256': manifest['tasks_sha256'],
           'source_tasks_sha256': manifest.get('source_tasks_sha256', manifest['tasks_sha256']),
           'git_commit': manifest['git_commit'], 'created_at': now_iso(), 'human_approved': False,
           'degraded': bool((manifest.get('status') or {}).get('degraded')), 'units': final}
    if source_refs is not None:
        doc['source_readiness'] = source_readiness(source_refs)
    validate_final(doc)                                        # an invalid hand-off is a bug: stop here
    if source_refs is not None:
        check_coverage(doc, source_refs)                       # each original unit exactly once, unchanged
    write_json_once(run_dir / 'translation_final.json', doc)
    write_json_once(run_dir / 'metrics.json', metrics)
    if refine_log:
        write_json_once(run_dir / 'refine_log.json', refine_log)
    rows = []
    for t in tasks:
        for c in cands[t['unit_id']]:
            e = c['eval']
            rows.append({'unit_id': t['unit_id'], 'cand_id': c['cand_id'], 'also_from': '|'.join(c['also_from']),
                         'accepted': e['accepted'], 'reasons': '|'.join(e['reasons']), 'qe': e['qe'],
                         'spelling_rate': e['spelling_rate'], 'letters': e['letters'],
                         'predicted_s': e['predicted_s'], 'ratio_slot': e['ratio_slot'],
                         'ratio_window': e['ratio_window'], 'zone': e['zone'],
                         'style_flags': '|'.join((e.get('language') or {}).get('calques', []) +
                                                 (e.get('language') or {}).get('simplicity', [])), 'text': c['text']})
    path = run_dir / 'candidates_scored.csv'
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['unit_id'])
        writer.writeheader()
        writer.writerows(rows)


def _log_mlflow(cfg, manifest, metrics, run_dir, log):
    """Optional MLflow mirror of the run (the JSON registry stays the source of truth)."""
    try:
        import mlflow
    except ImportError:
        log('[tracking] mlflow not installed; registry.jsonl only')
        return
    mlflow.set_tracking_uri(cfg['tracking'].get('mlflow_uri') or f'file:{run_dir.parent.parent / "mlruns"}')
    mlflow.set_experiment(cfg['tracking'].get('experiment', 'translation_step'))
    with mlflow.start_run(run_name=manifest['run_id']):
        mlflow.log_params({'config_sha256': manifest['config_sha256'], 'git_commit': manifest['git_commit'],
                           'generators': ','.join(manifest['generators']), 'refiner': manifest['refiner'],
                           **{f'gate.{k}': v for k, v in cfg['gate'].items() if not isinstance(v, (list, tuple))}})
        mlflow.log_metrics({k: metrics[k] for k in ('accepted_pct', 'accepted', 'accepted_after_refine')
                            if metrics.get(k) is not None})
        if metrics.get('qe_mean') is not None:
            mlflow.log_metric('qe_mean', metrics['qe_mean'])
        mlflow.log_artifacts(str(run_dir))


def _append_registry(path, manifest, metrics):
    rec = {k: manifest[k] for k in ('run_id', 'created_at', 'package_version', 'git_commit', 'config_sha256',
                                    'tasks_sha256', 'generators', 'qe_metric', 'refiner')}
    rec['metrics'] = {k: metrics[k] for k in ('units', 'accepted', 'accepted_pct', 'accepted_after_refine',
                                              'qe_mean', 'zones', 'picked_from', 'needs_review_units')}
    rec['degraded'] = metrics.get('degraded', False)
    rec['degraded_reasons'] = metrics.get('degraded_reasons', [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
