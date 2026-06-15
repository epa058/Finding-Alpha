# Progress Tracker — EquityBubbleRegimes

Update this file after every meaningful implementation change.

## Current Phase

- S&P 500 point-in-time constituents pipeline complete (2026-06-15). LPPL
  fitting prototype complete (2026-06-15). Hype/Sentiment index prototype
  complete (2026-06-15), using GDELT (see "Hype/Sentiment index prototype"
  below). Next: combine LPPL residuals + hype/sentiment into Bubble Score
  labels.

## Current Goal

- Combine LPPL residuals (`ε_norm(t)` from `lppl_fitting.ipynb`) with the
  hype/sentiment features (`H`, `CapH`, `S` from `hype_sentiment.ipynb`) into
  Bubble Score labels, per `ai-workflow-rules.md` step 4.

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

- **Hype/Sentiment index prototype** (2026-06-15) — `hype_sentiment.ipynb`
  computes `H_i,t`, `CapH_i,t` (HLPPL Hype Index / cap-adjusted Hype Index)
  and a tone-based `S_i,t` (Sentiment Score proxy) from GDELT, for GME
  against a small reference universe (AAPL, MSFT, TSLA, GME), covering the
  85-day window 2026-03-23 to 2026-06-15 (a recent/live case study — see
  "GDELT data source decision" for why a historical case wasn't possible).
  Saved to `EquityBubbleRegimes_HypeSentiment_GME_recent90d.csv`
  (`date, N_i, N_mkt, H, CapH, S`). Results:
  - `H_GME`: mean 0.0134, std 0.0182, range [0, 0.1232].
  - `CapH_GME`: mean 12.16, std 16.55, range [0, 112.0] — GME's news-attention
    share is on average ~12x its market-cap weight, spiking to ~112x. A
    strong, expected meme-stock over-hype signal that sanity-checks the
    pipeline.
  - `S_GME`: mean -0.0248, std 0.194, range [-1.0, 0.424] — within `[-1,1]`
    as designed.
  - Market-cap weights used (current, static): AAPL 0.4907, MSFT 0.3344,
    TSLA 0.1738, GME 0.0011.
  - All 8 GDELT cache files (4 tickers x 2 modes) fetched successfully into
    `gdelt_cache/`.

## HLPPL hype/sentiment methodology (research notes, 2026-06-15)

Reviewed the full HLPPL paper (arXiv:2510.10878) to inform the sentiment/hype
index step:

- **Hype Index** `H_i,t = N_i,t / N_mkt,t` — share of financial news articles
  mentioning stock `i` on day `t`, out of total articles covering a reference
  universe (e.g. S&P 100) that day. Measures attention *volume*, not tone.
  Also define a capitalization-adjusted version `CapH_i,t = H_i,t / W_cap_i,t`
  (`W_cap` = stock's share of total market cap): `CapH > 1` = excess hype
  relative to economic size, `CapH < 1` = under-hyped. The Hype Index
  methodology itself is from a separate prior paper: Cao, Wunkaew & Geman,
  "The Hype Index: an NLP-driven Measure of Market News Attentions"
  (arXiv:2506.06329, 2025).
- **Sentiment Score** `S_i,t` — FinBERT-based polarity score per article
  (`s_i,t,k ∈ [-1,1]`), aggregated as a confidence-weighted daily average.
  Captures tone, complementing Hype's volume measure.
- **Pipeline (paper's Section 4.2.1)**: WSJ news corpus (2018-2024) ->
  BERTopic to filter/cluster articles by sector -> FinBERT sentiment scoring
  -> daily aggregation into `[S_pos, S_neu, S_neg]` one-hot-weighted vectors.
- **Bubble Score** combines LPPL residual `ε_norm(t)` with hype/sentiment:
  `BubbleScore = ε_norm(t) ± α1*H_i,t + α2*S_i,t` (sign depends on
  bubble vs. negative-bubble regime per `ε_norm(t)`'s sign). Hype amplifies
  the extreme in either direction; sentiment is a corrective/dampening term.
- **Caveat**: the paper's news data is WSJ (likely paid/institutional access)
  and market-level fundamentals come from WRDS/CRSP. A free replication needs
  a substitute news corpus (e.g. a free financial news API/RSS) feeding a
  FinBERT pipeline — this is the key open question for our sentiment/hype
  step (see Open Questions).

## Next Up

1. Combine LPPL features + hype/sentiment into bubble labels (Bubble Score:
   `BubbleScore = ε_norm(t) ± α1*H_i,t + α2*S_i,t`).
2. Dual-stream transformer + Bubble Score.
3. Run the rolling LPPL pipeline across the full S&P 500 point-in-time
   universe for backtesting (currently validated on single
   tickers/indices only).
4. Future: expand the hype/sentiment reference universe toward S&P 100/500,
   and/or move to GDELT BigQuery for full historical (2015+) coverage so
   hype/sentiment features can be computed for the dot-com/GFC/GME 2021
   case studies (see "GDELT data source decision" below).

## Open Questions

- Whether the `damping>0.5` qualifying condition should be relaxed or made
  case-type-dependent (see GME finding above) once more cases/tickers are
  tested.
- Whether/how this integrates with MacroRegimes2's macro regime labels —
  deferred until both projects are further along.
- GDELT's free DOC API rate limit was partially resolved in practice: all
  8/8 cache files for the 4-ticker universe eventually fetched successfully
  (~3 hours total across retries, with most files taking several minutes
  each but the final retry succeeding quickly). Still unclear if this
  sandbox-specific slowness would recur for a larger universe or in other
  environments — if scaling up the reference universe, budget significant
  wall-clock time or move to BigQuery.

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
- **GDELT data source decision** (2026-06-15): chosen as the free,
  no-API-key source for both Hype Index and Sentiment Score, via the GDELT
  2.0 DOC API's `timelinevol` (daily article-volume intensity) and
  `timelinetone` (daily average tone) modes. Key findings:
  - `H_i,t = N_i,t/N_mkt,t` can be computed directly from GDELT's "Volume
    Intensity" values without knowing raw article counts, because
    `intensity_i,t = N_i,t/N_total,t` and the unknown `N_total,t` cancels
    when normalizing across a reference universe.
  - GDELT's average tone substitutes for the paper's FinBERT `S_i,t`
    (rescaled from roughly `[-10,10]` to `[-1,1]`) — avoids running any NLP
    model, directly addressing Simon's "the NLP part is too complicated"
    concern.
  - **Coverage limitation**: with no `startdatetime`/`enddatetime`, the free
    DOC API returns only a rolling ~90-day window ending "today". Explicit
    historical date params (tested for Jan-Feb 2021 and even May 2026)
    returned persistent `429` errors — the free DOC API does not serve
    arbitrary historical ranges. Historical hype/sentiment (dot-com, GFC,
    GME 2021) would require GDELT's BigQuery GKG archive (free GCP tier,
    2015-present) — deferred as future work.
  - **Rate-limit finding**: the free DOC API enforces "one request per ~5
    seconds" per its own error message, but in this dev sandbox actual
    successful requests took far longer (~4 minutes per attempt observed,
    mostly `429` retries) — far slower than the documented limit suggests.
    Unclear if this is sandbox-specific (shared/throttled egress IP) or a
    general free-tier reality; flagged as an open question. Because of this,
    the prototype reference universe was kept small (AAPL, MSFT, TSLA, GME)
    rather than the originally-planned ~15-20 tickers.
  - All GDELT responses are cached to `gdelt_cache/` (per query+mode) so
    reruns are instant and don't re-hit the rate limit.
