# Visualization & Output Conventions — MacroRegimes2

There is no UI for this project (it's a notebook). This file replaces the
typical "UI context" with the conventions used for plots and exported
artifacts, so new charts/outputs stay consistent with the existing ones.

## Plotting (matplotlib / seaborn)

- Figures use `plt.subplots(figsize=(14, 4))`-style wide panels for time
  series (one variable per row when stacking multiple series).
- Regime/segment shading uses a fixed color map per regime index, e.g.
  `{0: 'coral', 1: 'steelblue', 2: 'mediumseagreen', 3: 'orchid', 4: 'goldenrod'}`
  — reuse these so regime colors mean the same thing across all charts.
- Heatmaps (emission means, confusion matrices) use `cmap='RdYlGn'` for
  0–1 normalized quantities (centered at 0.5) and `cmap='Blues'` for counts
  (e.g. confusion matrices), with `annot=True`.
- Always call `plt.tight_layout()` before `plt.show()`.

## Regime labeling

- Regimes are always labeled `Regime 0`, `Regime 1`, `Regime 2`, ... and
  ordered canonically by mean term-spread (`sort_by_spread`) — never by raw
  cluster/state index from the model. Apply the same canonical ordering to
  any new model's output before comparing it to existing regime labels.

## Output CSVs

- Prefix: `MacroRegimes2_<Model>_<Artifact>.csv`, written to this folder
  (mirrors the Colab Drive `project_folder`).
- Time-series outputs are indexed by `date` (monthly, `MS` frequency).
- Keep one CSV per logical artifact (time series, transition matrix,
  emission means, summary stats) rather than combining into one wide file —
  matches the existing `MacroRegimes2_HMM_*` set.
