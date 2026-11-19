[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/haidildariell/gta6-rarity-forecast-v1/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/haidildariell/gta6-rarity-forecast-v1/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
Built for the [GTA VI trophy earn-rate forecast bounty on make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

# GTA VI trophy earn-rate forecast

One program, `main.py` (Python 3.13, standard library only), forecasts every
GTA VI PlayStation 5 trophy's Exophase earn rate (%) at the case target as five
quantiles (5/25/50/75/95%), scored by mean pinball loss on the logit scale.

## Model

An earn rate is a ratio, `r = earners / players_tracked`, and the two sides
move against each other over launch week: the tracked base grows as more
players sync profiles (dilution — pushes rates down, hardest for rare
trophies, since newcomers hold almost nothing yet) while earners grow as
players progress (pushes rates up, fastest for easy/early trophies). The
program therefore forecasts `N` and per-trophy earners separately, each as a
log-log fit in time since the release day began, and divides.

Two regimes, picked from the history itself:

- **Rich history (2+ distinct snapshots):** OLS of log `N` and log earners on
  log(1 + t), per-trophy slopes shrunk toward their tier-group mean, earners
  monotone (slopes floored at 0), medians capped by tier (platinum 5%, gold
  30%, silver 65%, bronze 92%). Spread from fit residuals + an extrapolation
  term growing with log-time distance to the target + a refresh-risk term
  (Exophase refreshes a game's counts every few hours, so the answer is the
  last refresh before the target).
- **Flat history (one distinct snapshot — the practice cases):** no trend is
  estimable, and persistence is exactly what the refresh breaks. The program
  models the next refresh: tracked players ×~6 (the published first-refresh
  jump, 638 → 3,868), newcomers holding the opening trophy (the one at 100%)
  and almost nothing else. The published totals pin this down: after the
  refresh 4,276 trophies were earned by 3,868 players, so the 3,230 newcomers
  hold 3,333 trophies — ~3,230 of them the opening trophy, only ~100 spread
  over the other 54, tilted toward the easiest trophies. Median = diluted
  rate; spread σ = 0.45 logit (grid-searched, see below).

Deterministic, no network, no randomness; runs in ~0.03 s per case.

## Backtest (honest numbers)

- **In-history backtest** (fit on reads ≤ 02:00, predict the 02:30/03:00 reads):
  mine 1.15 vs house baseline 0.02. This window is degenerate — every read is
  identical, no refresh happens inside it — so persistence wins it trivially
  and it cannot test refresh handling. It is reported, not hidden.
- **Refresh reconstruction** (practice answer rebuilt from the published
  totals — 3,868 players, 4,276 trophies earned — under three allocations of
  the ~100 non-opening trophies: pure dilution / proportional to old rate /
  concentrated on easiest): mine **0.153 / 0.093 / 0.099** vs house baseline
  **1.69 / 1.40 / 1.43** (house's own published practice loss is ~1.51, inside
  that bracket). Median vs last read: e.g. a trophy last at 1.10% forecasts
  0.25% median; the 100% opening trophy stays 100%.
- **Synthetic final-like worlds** (2 days of reads with refreshes, 4.5-day
  horizon, three player-growth shapes): mine 0.208 / 0.085 / 0.077 vs house
  0.278 / 0.272 / 0.392.
- Practice runs: shape valid (55 rows, quantiles ordered, 2,771 bytes),
  deterministic byte-for-byte across runs, ~0.03 s per case (limit 8 s).

## How to run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input dir, writes `<id>.out` per case.
