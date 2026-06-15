# Code Standards — MacroRegimes2

## General

- This is research code in a single notebook — prioritize correctness and
  readability over abstraction. Don't extract modules/packages unless the
  notebook becomes genuinely unmanageable.
- Keep the section structure (`## N. Title`) intact. New work should be added
  as a new numbered section (or subsection) at the end, or inline where it
  extends an existing section — don't renumber/reorganize existing sections
  without good reason, since outputs and notes reference section numbers.

## No look-ahead bias (critical)

This is the #1 rule, and the reason MacroRegimes2 exists (MacroRegimes1 had
multiple violations of this). When adding or modifying any transform:

- Rolling windows (e.g. the 5-year quantile standardization) must only use
  data up to and including the current row — never `center=True`, never
  `.shift(-n)` on inputs to a model.
- When fitting a model "out-of-sample", make sure the train window strictly
  precedes the test point in time.
- If a series is revised after the fact (e.g. `PAYEMS`), be aware that using
  the *current* vintage for historical dates is itself a form of look-ahead —
  prefer series that aren't heavily revised (this is why `ICSA`/`CCSA` are
  used instead of `UNRATE`/`PAYEMS`).
- When in doubt, write a quick sanity check (e.g. compare a rolling stat
  computed two ways) before trusting a new transform.

## Data fetching

- FRED series go in `fred_series`, yfinance tickers in `yf_series` — add new
  series to these lists rather than fetching ad hoc inline.
- If FRED discontinues or changes a series, prefer a bridging/patch function
  (see `sigmoid_bridge`, `ecu_euro_series`) over dropping the series, unless
  there's no reasonable substitute.
- Cache raw pulls that are slow or at risk of disappearing (pattern used for
  `fred_raw.csv`) — write a CSV backup after fetching.
- Local CSV/Excel files used as data patches live in this folder
  (`MacroRegimes2/`), referenced via `project_folder`.

## Modeling

- New regime-classification approaches should follow the existing pattern:
  fit → canonical-order labels by mean term-spread (`sort_by_spread`) →
  compare against `USREC` recession overlap → save results as
  `MacroRegimes2_<Model>_<Artifact>.csv`.
- Keep model comparison code (confusion matrices, agreement %) when adding a
  new model, so it's directly comparable to the Bayesian tree / PV-RF / HMM
  results already in the notebook.

## Environment

- Designed for Google Colab with Google Drive mounted
  (`project_folder = '/content/drive/MyDrive/MacroRegimes2'`). If working
  locally, keep `project_folder` pointing at this directory so relative file
  references (CSV/Excel inputs and outputs) still resolve.
- Handle optional dependencies (e.g. `fredapi`) with the existing
  try/`pip install`/import pattern rather than assuming they're preinstalled.
