# Architecture Context — EquityBubbleRegimes

## Status

Nothing built yet — this file describes the *expected* shape based on
MacroRegimes2's conventions and the HLPPL paper, to guide the first
implementation. Update it as real decisions are made.

## Expected format

Likely a Jupyter notebook (matching `MacroRegimes2.ipynb`'s style), in this
folder, e.g. `EquityBubbleRegimes.ipynb` — Colab-friendly, Google
Drive-mounted `project_folder`, same as MacroRegimes2 (see that project's
`architecture.md` for the pattern).

## Expected pipeline (per HLPPL paper)

```
S&P 500 point-in-time constituents  ─┐
                                      │
Price/volume data (per ticker, yfinance or similar) ──┐
                                                        ├─→ LPPL fits per window
News/text data (for sentiment + hype index) ──────────┘         │
                                                                  ▼
                                                     Bubble labels (LPPL + sentiment + hype)
                                                                  │
                                                                  ▼
                                              Dual-stream transformer (market data + labels/sentiment)
                                                                  │
                                                                  ▼
                                                        Bubble Score time series
                                                                  │
                                                                  ▼
                                                      Backtest (signal → returns)
```

## Dependencies / blockers

1. **Point-in-time S&P 500 constituents (1990–present)** — blocks any
   multi-ticker backtest. See `project-overview.md`. Until resolved, initial
   work could prototype on a small fixed set of large, long-lived tickers
   (e.g. current mega-caps with full history) to validate the LPPL/transformer
   pipeline mechanics without claiming a realistic backtest.
2. **Sentiment / hype index** — the HLPPL paper builds this from prior NLP
   work on volatility forecasting; needs a news/text data source. Not yet
   sourced.
3. **LPPL fitting implementation** — needs a nonlinear fitting routine
   (existing open-source LPPL fitters could be a starting point rather than
   implementing from scratch).

## Storage model (proposed, pending first implementation)

- Follow MacroRegimes2's convention: local CSV/Excel inputs + cached raw pulls
  in this folder, output artifacts prefixed `EquityBubbleRegimes_*.csv`.
