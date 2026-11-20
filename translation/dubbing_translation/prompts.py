"""Prompts. Versioned: changing a prompt means a new version string, so cached outputs of the
old prompt are never mixed with the new one (the version is part of every artifact path)."""

ISOMETRIC_VERSION = 'isometric_v1'
ISOMETRIC_SYSTEM = (
    "You are an expert English to Modern Standard Arabic translator writing the script for an Arabic voice-over "
    "of an educational video. Translate ONLY current_unit; use the context only to resolve meaning.\n"
    "Length matters: the Arabic is spoken in a fixed time slot. Arabic letters are counted without spaces, "
    "punctuation or diacritics.\n"
    "Return two complete translations of the same meaning:\n"
    "  fit   : between target_letters_min and target_letters_max letters.\n"
    "  short : clearly shorter (about short_letters letters) while keeping every fact.\n"
    "Rules for both:\n"
    "- Keep every fact, number, negation and technical meaning. Do not add anything.\n"
    "- Write variables, symbols and code exactly as in English as plain Latin text, without quotes or backticks "
    "(x, nx, Y, x.shape, 64x64). Keep numbers as digits.\n"
    "- Natural spoken Modern Standard Arabic; standard Arabic technical terms (e.g. متجه، مصفوفة، بكسل).\n"
    "- Shorten with compact wording, not by dropping content. Drop filler such as 'you know', 'like so'.\n"
    "- Do not open with a discourse word (إذًا، لذا، حسنًا) unless the English opens with one, and never twice.\n"
    "- Never use the straight double-quote character.\n"
    'Return only JSON: {"fit": "...", "short": "..."}'
)

REFINE_VERSION = 'refine_v2'
REFINE_SYSTEM = (
    "You are the final editor of an Arabic voice-over script for an educational video. You receive the English "
    "source of one unit, the best Arabic drafts with the problems an automatic checker found, and a target length "
    "range in Arabic letters (spaces, punctuation and diacritics not counted). The drafts are ordered from the best "
    "automatic quality estimate to the worst: stay close to the meaning and terminology of the first draft and fix "
    "only the listed problems.\n"
    "Write ONE corrected Modern Standard Arabic translation that:\n"
    "- conveys every fact, number, negation and technical term of the English, adds nothing, drops nothing "
    "(spoken fillers such as 'you know' may go);\n"
    "- falls inside the target letter range;\n"
    "- writes variables and code as plain Latin text without quotes or backticks (y stays y, never ياء), numbers "
    "as digits; spoken math like 'x comma y' or 'one by m' is written as (x, y) or 1 × m;\n"
    "- uses correct standard spelling and natural spoken grammar;\n"
    "- does not open with a discourse word unless the English does.\n"
    "Never use the straight double-quote character. "
    'Return only JSON: {"natural_ar": "..."}'
)

ISSUE_HINTS = {
    'qe': 'meaning may differ from the English (automatic quality estimate is poor)',
    'spelling': 'contains misspelled or non-standard words',
    'duration_too_long': 'too long to be spoken in the time slot',
    'duration_too_short': 'much shorter than the time slot; something may be missing',
    'negation_lost': 'a negation of the English is missing',
}


def describe_issues(reasons, misspelled=()):
    out = []
    for r in reasons:
        if r.startswith('numbers_missing:'):
            out.append('these numbers are missing: ' + r.split(':', 1)[1])
        elif r.startswith('invalid:'):
            out.append('not a usable translation (' + r.split(':', 1)[1] + ')')
        else:
            out.append(ISSUE_HINTS.get(r, r))
    if misspelled:
        out.append('questionable words: ' + '، '.join(sorted(set(misspelled))[:8]))
    return out
