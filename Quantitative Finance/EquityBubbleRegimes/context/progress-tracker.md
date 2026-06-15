# Progress Tracker — EquityBubbleRegimes

Update this file after every meaningful implementation change.

## Current Phase

- S&P 500 point-in-time constituents pipeline complete (2026-06-15). LPPL
  fitting prototype complete (2026-06-15). Next: sentiment/hype index (see
  `ai-workflow-rules.md` for the intended order).

## Current Goal

- Identify a sentiment/hype index data source, per `ai-workflow-rules.md`
  step 3.

## Completed

- **S&P 500 point-in-time constituents** (2026-06-15) — `sp500_constituents.ipynb`
  fetches `fja05680/sp500`'s "S&P 500 Historical Components & Changes
  (Updated).csv", caches it locally (`sp500_historical_components_raw.csv`),
  and transforms it into a long-format membership table
  (`EquityBubbleRegimes_SP500Constituents.csv`: `ticker, start_date, end_date`).
  Validated against known events (TSLA added 2020-12-21, FB→META rename
  2022-06-09, ENRNQ removed 2001-11-30, GOOG added 2014-04-03). 1255 membership
  intervals, 1202 unique tickers, 503 currently active, covering
  1996-01-02 to present.

- **LPPL fitting prototype** (2026-06-15) — `lppl_fitting.ipynb` implements
  `fit_lppl` (Filimonov-Sornette two-step calibration: `differential_evolution`
  over `(tc, m, omega)`, linear least-squares for `A, B, C1, C2`),
  `lppl_features` (Sornette qualifying conditions: `0.1<=m<=0.9`,
  `2<=omega<=25`, `B<0`, `0<horizon<=0.5*window_len`, `damping>0.5`), and
  `rolling_lppl` (fits across a range of window-end dates `t2`). Validated on
  three cases, each saved to `EquityBubbleRegimes_LPPL_Features_<case>.csv`:
  - **NASDAQ dot-com** (`^IXIC`, 700-day window, `t2` Sep 1999-Feb 2000):
    19/19 fits qualified; `tc` estimates for `t2` around 1999-12-10/12-20
    land near 2000-03-12, close to the actual 2000-03-10 peak.
  - **S&P 500 2008 GFC** (`^GSPC`, 700-day window, `t2` Apr-Sep 2007): 19/19
    qualified; `t2=2007-07-10` -> `tc≈2007-10-04`, `t2=2007-07-20` ->
    `tc≈2007-10-09` — essentially exact agreement with the actual
    2007-10-09 peak.
  - **GME 2021** (120-day window, `t2` Jan 4-24 2021): 0/5 qualified — all
    fail `damping>0.5` (≈0.25-0.27) despite `tc` estimates landing within
    1-4 days of the actual 2021-01-27/28 peak (`t2=2021-01-19` ->
    `tc≈2021-01-21`, `t2=2021-01-24` -> `tc≈2021-01-23`). Flagged as a
    discussion point: the damping threshold, calibrated for slow macro
    bubbles, may be too strict for fast short-squeeze dynamics — a
    case-type signal rather than a fit to discard.

## In Progress

- None.

## Next Up

1. Sentiment/hype index data source.
2. Combine LPPL features with sentiment/hype into bubble labels.
3. Dual-stream transformer + Bubble Score.
4. Run the rolling LPPL pipeline across the full S&P 500 point-in-time
   universe for backtesting (currently validated on single
   tickers/indices only).

## Open Questions

- Sentiment/hype index data source not yet identified.
- Whether the `damping>0.5` qualifying condition should be relaxed or made
  case-type-dependent (see GME finding above) once more cases/tickers are
  tested.
- Whether/how this integrates with MacroRegimes2's macro regime labels —
  deferred until both projects are further along.

## Architecture Decisions

- **S&P 500 constituents source**: `fja05680/sp500` (MIT licensed,
  Wikipedia + manual research), covering 1996-present. 1990-1996 gap accepted
  as a scope limitation; tickers active on 1996-01-02 have left-censored
  `start_date`. Raw CSV cached locally to avoid dependence on the upstream
  repo staying available.
- **LPPL fitting approach**: Filimonov-Sornette two-step calibration
  (`scipy.optimize.differential_evolution` for `(tc, m, omega)`, linear
  least-squares for `A, B, C1, C2`), with Sornette qualifying conditions and
  a rolling-window ("confidence indicator") feature extraction. Validated
  against three known historical bubbles before proceeding to sentiment/hype
  work.
