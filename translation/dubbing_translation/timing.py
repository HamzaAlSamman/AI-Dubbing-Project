"""Time windows, letter budgets and predicted spoken duration.

A unit's *slot* is its source speech duration. Its *window* is the time the dubbed clip may
really occupy on the timeline: the slot, plus the silence before the next unit minus a guard,
plus a small early start into the silence after the previous unit (same rules as the final
audio timeline check). Predicted duration = Arabic letters / speaking rate of the voice.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Window:
    unit_id: str
    slot_s: float                 # source speech duration
    window_s: float               # time available on the timeline (>= slot_s)

    def as_dict(self):
        return asdict(self)


def windows_from_timing(timings, *, next_guard_s=0.10, max_lead_s=0.50):
    """timings: list of (unit_id, start_s, end_s) in timeline order."""
    ordered = sorted(timings, key=lambda t: t[1])
    out = {}
    for i, (uid, start, end) in enumerate(ordered):
        if end <= start:
            raise ValueError(f'invalid timing for {uid}: {start}..{end}')
        gap_next = (ordered[i + 1][1] - end) if i + 1 < len(ordered) else 1.0
        gap_prev = (start - ordered[i - 1][2]) if i else 0.0
        extra = max(0.0, gap_next - next_guard_s) + min(max_lead_s, max(0.0, gap_prev))
        out[uid] = Window(uid, round(end - start, 3), round(end - start + extra, 3))
    return out


def windows_from_slots(slots):
    """Fallback when only durations are known: the window equals the slot."""
    return {uid: Window(uid, float(s), float(s)) for uid, s in slots.items()}


def calibrate_rate(clips, count):
    """Speaking rate from real clips: clips = [(text, seconds)], count = letters-of-text function.

    Returns rate plus the predictor error on the same clips, so a new voice (or a new way of
    counting) is measured before it is used by the gate."""
    letters = sum(count(t) for t, _ in clips)
    seconds = sum(s for _, s in clips)
    rate = letters / seconds
    errs = sorted(abs(count(t) / rate - s) / s for t, s in clips if s > 0)
    return {'letters_per_second': round(rate, 3), 'n': len(clips),
            'median_abs_error': round(errs[len(errs) // 2], 4) if errs else None}


def predicted_seconds(letters, rate):
    return letters / rate


def letter_budget(window, rate, *, low=0.90, high=1.05):
    """Target Arabic letter range for a unit (what the LLM is asked to write)."""
    return max(4, int(window.slot_s * rate * low)), max(6, int(window.window_s * rate * high))


def duration_verdict(letters, window, rate, *, min_ratio=0.60, max_speedup=1.25):
    """Classify a text length against its window.

    ratio_slot   = predicted / slot           (how full the source slot is)
    ratio_window = predicted / window         (needs speed-up above 1.0)
    fits      : natural speech ends inside the window
    speedup   : fits only with bounded speed-up (<= max_speedup)
    too_long  : overruns even with the speed-up
    too_short : below min_ratio of the slot (long silence / unnatural slow-down)
    """
    pred = predicted_seconds(letters, rate)
    ratio_slot = pred / window.slot_s
    ratio_window = pred / window.window_s
    if ratio_window > max_speedup:
        zone = 'too_long'
    elif ratio_window > 1.0:
        zone = 'speedup'
    elif ratio_slot < min_ratio:
        zone = 'too_short'
    else:
        zone = 'fits'
    return {'predicted_s': round(pred, 3), 'ratio_slot': round(ratio_slot, 3),
            'ratio_window': round(ratio_window, 3), 'zone': zone}
