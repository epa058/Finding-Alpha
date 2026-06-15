# Architecture Context — MacroRegimes2

## Format

A single Jupyter notebook (`MacroRegimes2.ipynb`), intended to run top-to-bottom
in Google Colab. There is no app, server, or package structure — "architecture"
here means the notebook's section pipeline and how data/files flow between
sections.

## Notebook section map

| Section | Purpose |
| --- | --- |
| 0. Standard Imports | numpy/pandas/yfinance/matplotlib/seaborn/scipy/pandas_datareader/fredapi |
| 1. Load & Merge Category Factors | Mount Drive, pull FRED + yfinance + CSV/Excel series, fix discontinued series, merge into `raw` |
| 2. Compute Year-over-Year Changes | YoY transforms for level series; rates/spreads/indices kept as-is |
| 3. Rolling 5-Year Quantile Standardization | Per-variable rolling quantile transform → `data` |
| 4. Bayesian Marginal Likelihood | Splitting criterion (Normal-Inverse-Gamma) used by the tree |
| 5. Reconstruct Yields from NSS Parameters | Builds the 13-maturity zero-coupon yield target from `feds200628.csv` |
| 6. Target Variable Setup | Assembles the tree's likelihood target |
| 7. Sequential Tree Growing | 2 splits → 3 regimes (Bayesian tree) |
| 8–10. Results / Visualizations / Stats | Tree regime outputs, plots, summary stats |
| 11. Feature Importance via RF | Random Forest importance over proxy variables |
| 12. Dimensionality Reduction | Proxy variables, benchmarked vs PCA and RMT-PCA |
| 13. OOS Proxy-Variable RF Classification | Full-sample → local canonical ground truth → actual OOS classification |
| 14. PV-RF Forecaster | 1-month-ahead regime forecast |
| 15. Compiled Results | RF results summary |
| (final) Gaussian HMM | HMM fit on proxy-variable space, Viterbi decoding, transition matrix, emission means, recession overlap, agreement vs Bayesian tree |

## Data flow

```
FRED API ──┐
yfinance ──┼─→ raw (merged monthly DataFrame, ~60 vars)
local CSV/Excel ─┘
        │
        ▼
YoY transforms + rolling 5yr quantile standardization → data
        │
        ├─→ Bayesian sequential tree → 3 regimes  → MacroRegimes2_TreeResults.csv
        ├─→ Proxy-Variable RF (OOS + forecaster) → MacroRegimes2_PV-RF-*.csv
        └─→ Gaussian HMM (proxy-variable space) → MacroRegimes2_HMM_*.csv
                       │
                       └─→ cross-checked against NBER `USREC` recession dates
```

## Storage model

- **Inputs**: local CSV/Excel files in this folder (`BAMLH0A0HYM2.csv`,
  `gold.csv`, `feds200628.csv`, `fear-greed_sentiment.xls`,
  `data_gpr_export.xls`) plus live pulls from FRED/yfinance each run.
- **Intermediate**: `fred_raw.csv` is saved as a cache/backup of the raw FRED
  pull (in case FRED removes a series later).
- **Outputs**: each model writes its own CSV(s) prefixed `MacroRegimes2_*`
  directly into this folder — these are the artifacts other notebooks/tools
  would consume.
- No database; everything is file-based, designed for a Colab + Google Drive
  workflow (`project_folder` variable points at the Drive folder mirroring
  this directory).

## Invariants

1. All transformations must be **point-in-time / no look-ahead** — rolling
   windows, quantile standardization, and regime labels must only use data
   available up to that date. This was the main bug class in `MacroRegimes1`.
2. The 3-regime structure (Bayesian tree, PV-RF, HMM) is canonically ordered
   by mean term-spread so regime labels are comparable across methods.
3. Output CSV naming convention `MacroRegimes2_<Model>_<Artifact>.csv` must be
   preserved if regenerated, since downstream work may reference these names.
4. Series that FRED has discontinued/altered (e.g. credit spreads, FX
   indices) are patched via local CSVs or bridging functions — do not silently
   drop them without an equivalent replacement.
