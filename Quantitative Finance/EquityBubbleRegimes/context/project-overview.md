# Project Overview — EquityBubbleRegimes

## What this is

A new (not yet started) companion project to `MacroRegimes2`, applying
regime-style thinking to **individual equities / equity markets** to detect
financial bubbles and anticipate crashes a few months in advance — the
equity-market analogue of MacroRegimes2's recession-anticipation goal.

## Primary reference

> *Identifying and Quantifying Financial Bubbles with the Hyped Log-Periodic
> Power Law Model (HLPPL)* — arXiv:2510.10878 (2026).

The HLPPL model:
- Generates "bubble labels" from a Log-Periodic Power Law (LPPL) fit, a
  sentiment score, and a "hype index" (from the authors' prior NLP work on
  volatility forecasting).
- Trains a dual-stream transformer on market data + the labels/sentiment to
  produce a continuous **Bubble Score** time series, capturing both
  overpricing (positive bubbles) and underpricing (negative bubbles) in one
  framework.
- Reported ~34.1% average annualized return backtesting U.S. equities
  2018–2024, with a conservative bias (few false positives) that suits it for
  market-timing/signaling use.

The initial goal of this project is to **replicate the HLPPL approach** (LPPL
fitting, hype/sentiment index, dual-stream transformer, Bubble Score) and
backtest it, before deciding how/whether to combine it with MacroRegimes2's
macro regime signals.

## Historical S&P 500 constituents — resolved

Backtesting equity-level models requires knowing **which tickers were in the
S&P 500 at each point in time** (to avoid survivorship bias). Instead of
scraping Wikipedia's revision history directly (free, but incomplete and
error-prone) or paying for vendor data (CRSP, Compustat/WRDS, Norgate,
Bloomberg — ruled out by preference), this project uses the
community-maintained, MIT-licensed dataset from
[`fja05680/sp500`](https://github.com/fja05680/sp500), which already performs
this reconstruction from Wikipedia + supplementary research.

- **Coverage: 1996-01-02 to present.** No free source (including this one)
  covers 1990–1996 — that ~6-year gap is an **accepted scope limitation**.
  Backtests effectively start in 1996, which still covers the dot-com bubble
  (1996–2002) onward.
- For tickers present in the dataset's first row (1996-01-02), the true entry
  date is unknown — `start_date` is left-censored to 1996-01-02.
- See `sp500_constituents.ipynb` for the fetch/transform/validation pipeline,
  and `EquityBubbleRegimes_SP500Constituents.csv` for the resulting
  point-in-time membership table (`ticker, start_date, end_date`).

## Relationship to MacroRegimes2

- Independent methodology (LPPL/transformer vs. trees/RF/HMM) and independent
  data needs (equity-level price/sentiment vs. macro series).
- May eventually combine: MacroRegimes2's macro regime labels could become an
  input feature to the HLPPL-style model, or vice versa. Not decided — don't
  assume integration until `project-overview.md` is updated to reflect that
  decision.

## LPPL fitting prototype — complete

`lppl_fitting.ipynb` implements the LPPL fitting and rolling-window feature
extraction pipeline (Filimonov-Sornette two-step calibration + Sornette
qualifying conditions), validated against three known historical bubbles
(NASDAQ dot-com, S&P 500 2008 GFC, GME 2021 short squeeze). Outputs:
`EquityBubbleRegimes_LPPL_Features_{IXIC_dotcom,GSPC_gfc,GME_2021}.csv`. See
`context/progress-tracker.md` for results and findings (including an open
question about the damping qualifying condition for fast bubbles like GME).

## Hype/Sentiment index prototype — complete

The HLPPL paper's Bubble Score combines the LPPL residual with a **Hype
Index** (`H_i,t = N_i,t/N_mkt,t`, a stock's share of news attention vs. a
reference universe; also a cap-adjusted `CapH_i,t`) and a **Sentiment Score**
(`S_i,t`, FinBERT-based tone). The paper's own data (WSJ corpus + WRDS/CRSP)
and a reference code repo
([chirindaopensource/...hyped_log_period_power_law](https://github.com/chirindaopensource/identifying_quantifying_financial_bubbles_hyped_log_period_power_law))
both require paid News API / WRDS access, so neither was directly usable.

`hype_sentiment.ipynb` instead uses the free, no-API-key
[GDELT Project](https://www.gdeltproject.org/) DOC 2.0 API:
- `H_i,t`/`CapH_i,t` are computed from GDELT's daily "Volume Intensity"
  (article-count share of all worldwide news), normalized across a small
  reference universe (AAPL, MSFT, TSLA, GME).
- `S_i,t` is GDELT's daily average article tone, rescaled to `[-1,1]` — a
  substitute for FinBERT that requires no NLP model.

**Key limitation**: the free DOC API only serves a rolling ~90-day window
(no historical date ranges), so this is currently a *live/forward-looking*
signal, not a historical backtest — it cannot yet produce hype/sentiment
features for the dot-com, GFC, or GME 2021 case studies. Full historical
coverage (2015+) would require GDELT's BigQuery archive (free GCP tier).
GDELT's free DOC API also turned out to be far more rate-limited in practice
than documented. See `context/progress-tracker.md` ("GDELT data source
decision") for details.

The notebook ran end-to-end over the 85-day window 2026-03-23 to 2026-06-15,
producing `EquityBubbleRegimes_HypeSentiment_GME_recent90d.csv`
(`date, N_i, N_mkt, H, CapH, S`). Headline result: `CapH_GME` averages ~12x
(peaking at ~112x) GME's market-cap weight — a strong, expected meme-stock
over-hype signal that sanity-checks the pipeline. `S_GME` stays within
`[-1, 1]` as designed. See `context/progress-tracker.md` for full stats.

## Status

S&P 500 point-in-time constituents pipeline, LPPL fitting prototype, and
Hype/Sentiment index prototype (GDELT-based `hype_sentiment.ipynb`, see
above) are all complete. Next: combine LPPL residuals + hype/sentiment into
Bubble Score labels. Dual-stream transformer and Bubble Score combination not
yet started.
