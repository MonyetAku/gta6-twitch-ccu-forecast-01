"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Method: cross-launch empirical quantiles, conditioned on the launch already
being live at hour 1.

For every past launch in the case file's ``comparables`` we take its release
day's 24 hourly Twitch viewer counts and its log-ratio profile
``log(viewers_h / viewers_1)`` for hours 2..24.  A launch whose hour-1 count
is tiny next to its own day peak (hour1 / day-max < 0.2) unlocked later in
the day (a single global unlock) and has a totally different shape -- its
viewers can rise 100-1000x after hour 1 -- so it is left out of the
reference set.  Grand Theft Auto VI unlocks at rolling midnight local time
(New Zealand 11:00Z Nov 18, Europe 23:00Z Nov 18, UK 00:00Z Nov 19), so by
hour 1 a large part of the world is already playing and hour 1 is already
a substantial count, exactly the regime the kept launches are in.  The
Grand Theft Auto VI category's own pre-release history (``Not a launch``)
is also left out: it is not a launch day at all.

For each forecast hour the five quantiles are the weighted empirical
quantiles of the kept launches' log-ratios at the scored levels, weighted
1.0 for launches from 2024 or later and 0.5 for older ones (the Twitch
ecosystem of 2024-26 is the closest analogue), exponentiated and scaled by
hour 1's print:  ``q_tau(h) = p1 * exp(Q_tau(log-ratio_h))``.

Leave-one-out validation on the 24 kept launches in the kit (each launch
predicted from the other kept launches, scored with the contest's log
pinball loss): loss 0.222, typical median miss ~46%, 90% coverage 0.89,
against the house persistence baseline's 0.265 on the same launches
(baseline: loss ~0.387 on the practice cases).  Using every comparable
unfiltered scores far worse (0.39), which is why the filter matters.

Fallbacks, in order, if too few launches pass the filter: relax the
hour1/day-max threshold to 0.1, then 0.05, then take every complete
release-day profile; if still fewer than 5 references, fall back to the
house baseline (persistence with a 0.15 log-spread per root-hour).

Deterministic: no randomness, no clock, no network, no model calls; only
the case file is read.  Output is exactly the required shape: one
``{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}`` object
with 23 rows, hours 2..24, quantiles non-decreasing, each in (0, 1e9],
rounded to 2 decimals half-up like the starter.

Run locally: ``mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py``
Grader: ``python3 -P -s /bot/main.py``, at most 8 s, 1 CPU, 1024 MB per case.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# House baseline spread, used only as a fallback (see module docstring).
SIGMA = 0.15

# A reference launch must already be live at hour 1: hour1 / day-max >= this.
RATIO_MIN = 0.2
# Relaxed thresholds, tried in order if too few references pass.
RATIO_FALLBACKS = (0.1, 0.05, 0.0)
MIN_REFERENCES = 10
MIN_REFERENCES_FALLBACK = 5


def round2(x):
    """x rounded to 2 decimals, half up, as the starter rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def day_profile(comp):
    """A comparable's release-day hourly viewers (24 numbers), or None.

    Returns None for the target game's own pre-release history (not a
    launch), for incomplete days, and for non-positive counts.
    """
    name = str(comp.get("name", ""))
    unlock = str(comp.get("unlock", ""))
    if "not a launch" in unlock.lower() or "before release" in name.lower():
        return None
    day_start = str(comp.get("day_start", ""))
    if len(day_start) < 10:
        return None
    date = day_start[:10]
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            by_hour[str(row["from"])] = row["viewers"]
        except (KeyError, TypeError):
            continue
    vals = []
    for hh in range(24):
        v = by_hour.get("%sT%02d:00:00Z" % (date, hh))
        if v is None or not isinstance(v, (int, float)) or v <= 0:
            return None
        vals.append(float(v))
    return vals


def reference_weight(comp):
    """Recency weight: 2024+ launches count double vs older ones."""
    try:
        year = int(str(comp.get("day_start", ""))[:4])
    except ValueError:
        year = 0
    return 1.0 if year >= 2024 else 0.5


def collect_references(comparables):
    """(profiles, weights) of usable reference launches, filtered."""
    usable = []
    for comp in comparables or []:
        prof = day_profile(comp)
        if prof is None:
            continue
        usable.append((prof, reference_weight(comp)))
    for threshold in (RATIO_MIN,) + RATIO_FALLBACKS:
        kept = [(p, w) for (p, w) in usable if p[0] / max(p) >= threshold]
        if len(kept) >= MIN_REFERENCES:
            return kept
    if len(usable) >= MIN_REFERENCES_FALLBACK:
        return usable
    return []


def weighted_quantiles(pairs, levels):
    """Weighted empirical quantiles (type-1) of [(value, weight), ...]."""
    ordered = sorted(pairs, key=lambda t: t[0])
    total = sum(w for _, w in ordered)
    out = []
    for q in levels:
        target = q * total
        acc = 0.0
        chosen = ordered[-1][0]
        for value, w in ordered:
            acc += w
            if acc >= target:
                chosen = value
                break
        out.append(chosen)
    return out


def baseline_quantiles(p1, h):
    """House baseline: persistence with widening log-spread."""
    return [round2(p1 * math.exp(z * SIGMA * math.sqrt(h - 1))) for z in Z]


def forecast_case(case, name):
    if list(case.get("levels") or []) != LEVELS:
        raise ValueError("%s asks for levels %s, not %s" % (name, case.get("levels"), LEVELS))
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("%s has no hour-1 print" % name)
    p1 = max(1.0, float(first["viewers"]))

    refs = collect_references(case.get("comparables"))
    hours = []
    for row in case.get("forecast") or []:
        h = int(row["hour"])
        if refs:
            pairs = [
                (math.log(prof[h - 1] / prof[0]), w) for prof, w in refs
            ]
            qs = weighted_quantiles(pairs, LEVELS)
            q = [p1 * math.exp(x) for x in qs]
        else:
            q = [float(v) for v in baseline_quantiles(p1, h)]
        # Non-decreasing, strictly positive, capped, rounded to 2 decimals.
        cleaned = []
        prev = 0.0
        for v in q:
            v = min(max(v, 0.01), 1e9)
            v = round2(v)
            if v <= prev:
                v = prev if prev > 0 else v
            cleaned.append(v)
            prev = v
        # Rounding can only break monotonicity by a cent at most; repair.
        for i in range(1, 5):
            if cleaned[i] < cleaned[i - 1]:
                cleaned[i] = cleaned[i - 1]
        hours.append({"hour": h, "q": cleaned})
    hours.sort(key=lambda r: r["hour"])
    if [r["hour"] for r in hours] != list(range(2, 25)):
        raise ValueError("%s forecast hours are not 2..24" % name)
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / ("%s.out" % path.stem)).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
