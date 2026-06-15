# Project Overview — MacroRegimes2

## What this is

A research notebook (`MacroRegimes2.ipynb`) that classifies the U.S. economy into
discrete **macroeconomic regimes** month-by-month, with the goal of identifying
regime shifts that precede recessions by a few months.

Based on:
> Diebold, Bie, Li, He. *Machine Learning and the Yield Curve: Tree-Based
> Macroeconomic Regime Switching* (2024), with the proxy-variable factor list
> from Zhao et al. (2025).

This is the second iteration. `MacroRegimes1.ipynb` had significant look-ahead
bias from careless preprocessing decisions; MacroRegimes2 fixes those issues
(see `code-standards.md`) and extends the methodology with two additional
classifiers.

## Goal

- Build a monthly dataset of ~60 macro/financial variables (FRED, yfinance,
  supplementary CSVs/Excel).
- Classify history into 3 regimes using three independent methods, and check
  how well each lines up with NBER recession dates (`USREC`).
- End state for this notebook: **complete and validated** — all three models
  run end-to-end and their outputs (regime time series, transition matrices,
  emission means, summaries) are saved as CSVs in this folder.

## The three classification approaches

1. **Bayesian sequential tree** (sections 4–10) — greedy 2-split tree using
   Bayesian marginal likelihood (Normal-Inverse-Gamma) as the splitting
   criterion, targeting Svensson-reconstructed zero-coupon yields.
2. **Proxy-Variable Random Forest** (sections 11–15) — dimensionality
   reduction via proxy variables (benchmarked against PCA / RMT-PCA), then
   out-of-sample RF regime classification and a 1-month-ahead regime
   forecaster.
3. **Gaussian HMM** (final section) — Hidden Markov Model over the same
   proxy-variable space, with Viterbi decoding, transition matrix, emission
   means, and a confusion matrix vs. the Bayesian tree labels.

All three are cross-checked against actual NBER recession months to gauge
predictive value.

## Data sources

- **FRED** (via `fredapi`) — ~50 macro/bond/sentiment series (industrial
  output, labour, housing, credit, inflation, yields, FX, uncertainty
  indices).
- **yfinance** — major equity indices (`^GSPC`, `^IXIC`, `^DJI`, `^N225`,
  `^FCHI`, `^AEX`, `^BFX`, `^GSPTSE`, `^HSI`, `^FTSE`, `^GDAXI`).
- **Supplementary CSV/Excel files** in this folder — series FRED has removed
  or that need manual sourcing (e.g. `BAMLH0A0HYM2.csv`, `gold.csv`,
  `fear-greed_sentiment.xls`, `data_gpr_export.xls`, `feds200628.csv` for NSS
  yield parameters).
- Designed to run in **Google Colab** with the project folder mounted from
  Google Drive (`project_folder = '/content/drive/MyDrive/MacroRegimes2'`).

## Where this fits in the wider repo

- `Quantitative Finance/MacroRegimes1.ipynb` — earlier, superseded version
  (kept for reference / lessons learned).
- `Quantitative Finance/EquitiesRegimes1.ipynb`, `ML1-3.ipynb`,
  `GraphTheory1.ipynb` — other exploratory notebooks, not part of this
  project's scope.
- A separate, newer project (`EquityBubbleRegimes/`, sibling folder) applies
  similar regime-detection thinking to **equity bubbles / crash detection**
  (LPPL/HLPPL-based), independent of this notebook's macro focus.

## Status

Complete and validated for the current scope. Output artifacts already
present in this folder:
`MacroRegimes2_HMM_TimeSeries.csv`, `MacroRegimes2_HMM_TransitionMatrix.csv`,
`MacroRegimes2_HMM_EmissionMeans.csv`, `MacroRegimes2_HMM_RegimeSummary.csv`,
`MacroRegimes2_HMM_Model_Stats.csv`, `MacroRegimes2_TreeResults.csv`,
`MacroRegimes2_PV-RF-FCAST-Results.csv`, `MacroRegimes2_PV-RF-OOS-Results.csv`.

Future work on this notebook (if any) should extend/improve these models, not
restart them — see `progress-tracker.md` for open questions.
