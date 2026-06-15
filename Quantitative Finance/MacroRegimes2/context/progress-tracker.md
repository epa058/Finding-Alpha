# Progress Tracker — MacroRegimes2

Update this file after every meaningful implementation change.

## Current Phase

- Complete & validated. All three regime-classification approaches
  (Bayesian sequential tree, Proxy-Variable Random Forest, Gaussian HMM) run
  end-to-end and their outputs are saved as CSVs in this folder.

## Completed

- Built unified monthly dataset (~60 vars) from FRED, yfinance, and local
  CSV/Excel patches; cached raw FRED pull to `fred_raw.csv`.
- YoY transforms + rolling 5-year quantile standardization (no look-ahead).
- Bayesian sequential tree (2 splits, 3 regimes) targeting NSS-reconstructed
  yields → `MacroRegimes2_TreeResults.csv`.
- Proxy-variable dimensionality reduction (benchmarked vs PCA / RMT-PCA),
  OOS RF regime classification + 1-month-ahead forecaster →
  `MacroRegimes2_PV-RF-OOS-Results.csv`, `MacroRegimes2_PV-RF-FCAST-Results.csv`.
- Gaussian HMM over proxy-variable space, canonical regime ordering,
  transition matrix, emission means, recession overlap, agreement vs
  Bayesian tree → `MacroRegimes2_HMM_*.csv`.
- Context docs (this folder) rewritten 2026-06-15 to actually describe this
  project (previously contained unrelated content from an Astro website
  project, copy-pasted by mistake).

## In Progress

- None.

## Next Up

- No active work planned on this notebook itself. Future ideas (not started):
  - Investigate fixing `PAYEMS`/`UNRATE` revision issues noted in
    `project-overview.md`'s labour data section.
  - Possible reuse of the regime-classification methodology (canonical
    ordering, HMM/RF approach) in `EquityBubbleRegimes/` for equity
    crash/bubble detection — see that project's own context for status.

## Open Questions

- None blocking. If the equity-bubble project ends up needing macro regime
  labels as a feature, decide then whether MacroRegimes2 outputs should be
  exposed in a more reusable format (currently just CSVs in this folder).

## Architecture Decisions

- Notebook-based, Colab + Google Drive workflow (`project_folder`).
- File-based storage (input CSVs/Excel + output `MacroRegimes2_*.csv`), no
  database.
- 3-regime structure, canonically ordered by mean term-spread across all
  models for comparability.
