"""GTA VI trophy earn-rate forecaster for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy, five quantiles of its Exophase earn rate (%) at the
case target, as JSON {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.

Model (structural, per trophy, in logit space)
-----------------------------------------------
An earn rate is a ratio:  r_i(t) = e_i(t) / N(t),  where N(t) is the players
Exophase tracks and e_i(t) the tracked players holding trophy i.  Both move
against each other over a launch week:

  * N(t) grows as more players sync profiles (dilution: pushes rates DOWN,
    hard for rare trophies, because a new cohort holds almost nothing yet);
  * e_i(t) grows as players progress (pushes rates UP, fastest for easy and
    early trophies).

So the program forecasts N and e_i separately and divides, instead of
extrapolating the ratio directly.  Both are fitted in log-log time
(log value against log(1 + hours since the release day began)), which is the
shape a launch curve takes: fast early growth that keeps slowing.

Two regimes, decided by the history itself:

  * RICH history (>= 2 distinct snapshots): ordinary least squares of
    log N and log e_i on log(1 + t), per-trophy slopes shrunk toward their
    tier-group mean (a trophy seen twice cannot carry its own slope), earners
    monotone (slopes floored at 0), medians capped by tier (platinum can
    barely move in a week; an early bronze can).  Spread comes from the fit
    residuals plus an extrapolation term that grows with log-time distance
    to the target, plus a refresh-risk term: Exophase only refreshes a
    game's counts every few hours, so the answering read is the last refresh
    before the target, not the target instant itself.

  * FLAT history (one distinct snapshot, e.g. the practice cases: every
    launch-night read identical, 638 players, rates frozen): no trend is
    estimable, and persistence is exactly what the refresh breaks.  The
    program therefore assumes the next refresh lands before the target
    (the page had already been flat for 3+ hours at the checkpoint, and the
    published launch-night record shows refreshes hours apart) and models
    it: N multiplies by ~6x (the published first-refresh jump, 638 -> 3,868
    tracked players), the new cohort holds the opening trophy (the one at
    100%) and almost nothing else -- the published totals pin this down:
    after the refresh 4,276 trophies were earned by 3,868 players, i.e. the
    3,230 newcomers hold 3,333 trophies between them, so ~3,230 of those
    are the opening trophy and only ~100 spread over the other 54, tilted
    toward the easiest trophies.  The median is the diluted rate
    (old earners + small newcomer share) / new N; the spread is wide
    (sigma ~ 0.45 logit) because the exact split of those ~100 trophies
    across trophies is unknown and a refresh may or may not have landed.

Everything is deterministic (no randomness, no network, stdlib only) and
runs in milliseconds: the two executions of a case write identical bytes.

Usage: python3 main.py, reading $MAKEFUN_INPUT_DIR (else /input/) *.in and
writing $MAKEFUN_OUTPUT_DIR (else /output/) <id>.out per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# Release day began 2026-11-19T00:00:00Z; log-time is measured from here.
T0 = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- flat-history (refresh) priors, from the published launch-night record ---
REFRESH_MULT = 6.0        # first refresh: tracked players 638 -> 3,868 (~6x)
NEWCOMER_RATE = 0.07      # newcomers' earn rate on a non-opening trophy,
                          # as a fraction of the old rate (totals-calibrated:
                          # ~100 extra trophies over 54 trophies, see docstring)
NEWCOMER_TILT = 0.25      # tilt toward easier (higher-rate) trophies
FLAT_SIGMA = 0.45         # logit spread when history is flat (refresh risk);
                          # grid-searched on the refresh reconstruction: 0.35-0.45
                          # minimises mean pinball over the plausible allocations
# --- rich-history fit ---
TIER_CAP = {"platinum": 5.0, "gold": 30.0, "silver": 65.0, "bronze": 92.0}
MIN_SIGMA = 0.30
MAX_SIGMA = 1.10
REFRESH_RISK = 0.20       # extra logit spread: answer is last refresh <= target


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_t0(s):
    return (parse_time(s) - T0).total_seconds() / 3600.0


def log_time(t_hours):
    """log(1 + hours since release-day start); guards pre-midnight reads."""
    return math.log1p(max(t_hours, -0.9))


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return (a, b, resid)


def distinct_snapshots(case):
    """Collapse consecutive identical reads; keep first-seen time each."""
    snaps = []
    prev = None
    for h in case["history"]:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != prev:
            snaps.append(h)
            prev = key
    return snaps


def forecast_flat(case, snaps):
    """One distinct snapshot: model the next Exophase refresh before target."""
    last = snaps[-1]
    n_last = last["players_tracked"]
    n_new = n_last * REFRESH_MULT
    n_gain = n_new - n_last
    rates = last["rates"]
    ntr = len(case["trophies"])

    # Newcomer earners per trophy, tilted toward easier trophies, then
    # renormalised so the total matches NEWCOMER_RATE * sum(old rates).
    raw = []
    total_raw = 0.0
    for i in range(ntr):
        r = rates[i]
        if r is None or r >= 99.99:
            raw.append(0.0)
            continue
        k = NEWCOMER_RATE * (max(r, 0.05) / 0.9) ** NEWCOMER_TILT
        k = min(max(k, 0.02), 0.30)
        raw.append(k * r)
        total_raw += k * r
    target_total = NEWCOMER_RATE * sum(
        r for r in rates if r is not None and r < 99.99)
    scale = (target_total / total_raw) if total_raw > 0 else 0.0

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = rates[i]
        if r is None:
            r = 0.0
        if r >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        e_old = r * n_last / 100.0
        e_new = e_old + scale * raw[i] * n_gain / 100.0
        med = 100.0 * e_new / n_new
        med = min(max(med, 0.01), 99.99)
        lg = logit(med)
        q = [round4(inv_logit(lg + z * FLAT_SIGMA)) for z in Z]
        q = [max(0.01, min(100.0, v)) for v in q]
        q.sort()
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def forecast_rich(case, snaps):
    """ >= 2 distinct snapshots: log-log fits of N and per-trophy earners. """
    xs = [log_time(hours_since_t0(h["at"])) for h in snaps]
    x_target = log_time(hours_since_t0(case["target"]))
    x_last = xs[-1]
    extrap = max(0.0, x_target - x_last)
    ntr = len(case["trophies"])

    # --- players tracked ---
    log_n = [math.log(max(h["players_tracked"], 1)) for h in snaps]
    fit_n = ols(xs, log_n)
    if fit_n is None:
        b_n, a_n = 0.0, log_n[-1]
    else:
        a_n, b_n, _ = fit_n
        b_n = min(max(b_n, 0.0), 1.2)
    log_n_pred = a_n + b_n * x_target
    # never forecast fewer tracked players than the last read saw
    log_n_pred = max(log_n_pred, log_n[-1])
    n_pred = math.exp(log_n_pred)

    # --- per-trophy earners: first pass fits, then tier-group slopes ---
    fits = []
    for i in range(ntr):
        pts = []
        for h, x in zip(snaps, xs):
            r = h["rates"][i]
            if r is None:
                continue
            e = r * h["players_tracked"] / 100.0
            pts.append((x, math.log(e + 0.5)))
        if len(pts) >= 3:
            f = ols([p[0] for p in pts], [p[1] for p in pts])
        else:
            f = None
        fits.append((pts, f))

    group_slope = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [f[1] for (pts, f), t in zip(fits, case["trophies"])
                  if f is not None and t["tier"] == tier]
        if slopes:
            group_slope[tier] = sum(slopes) / len(slopes)
    all_slopes = [f[1] for _, f in fits if f is not None]
    overall = sum(all_slopes) / len(all_slopes) if all_slopes else 0.5
    for tier in ("platinum", "gold", "silver", "bronze"):
        group_slope.setdefault(tier, overall)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        pts, f = fits[i]
        tier = trophy["tier"]
        r_last = snaps[-1]["rates"][i]
        if r_last is None:
            r_last = 0.0
        if r_last >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        e_last = r_last * snaps[-1]["players_tracked"] / 100.0
        if f is not None:
            a_i, b_i, resid = f
            # shrink toward the tier mean; weight grows with # distinct points
            w = len(pts) / (len(pts) + 3.0)
            b_use = w * b_i + (1.0 - w) * group_slope[tier]
            b_use = max(b_use, 0.0)
            # re-anchor the intercept at the last point so the level is exact
            a_use = math.log(e_last + 0.5) - b_use * x_last
            resid = resid if resid is not None else 0.15
        else:
            b_use = max(group_slope[tier], 0.0)
            a_use = math.log(e_last + 0.5) - b_use * x_last
            resid = 0.35
        log_e_pred = a_use + b_use * x_target
        e_pred = math.exp(log_e_pred) - 0.5
        e_pred = max(e_pred, e_last)  # earners never decrease
        med = 100.0 * e_pred / n_pred
        med = min(max(med, 0.01), TIER_CAP.get(tier, 92.0))
        lg = logit(med)
        sigma = math.sqrt(max(resid, 0.15) ** 2
                          + (0.15 * math.sqrt(extrap)) ** 2
                          + REFRESH_RISK ** 2)
        sigma = min(max(sigma, MIN_SIGMA), MAX_SIGMA)
        if len(pts) < 3:
            sigma = max(sigma, 0.70)
        q = [round4(inv_logit(lg + z * sigma)) for z in Z]
        q = [max(0.01, min(100.0, v)) for v in q]
        q.sort()
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    snaps = distinct_snapshots(case)
    if len(snaps) < 2:
        return forecast_flat(case, snaps)
    return forecast_rich(case, snaps)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
