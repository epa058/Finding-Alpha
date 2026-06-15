# Visualization & Output Conventions — EquityBubbleRegimes

No UI for this project (notebook-based, like MacroRegimes2). Once
implementation starts, follow MacroRegimes2's `ui-context.md` conventions
(figure sizing, heatmap colormaps, `tight_layout()`) for consistency:

- Bubble Score time series should be plotted alongside the underlying price
  series (e.g. price on one axis, Bubble Score on a secondary axis or as a
  shaded band), similar in spirit to MacroRegimes2's regime-shaded macro
  variable plots.
- If/when this project produces regime-like labels (positive bubble / neutral
  / negative bubble), use the same canonical-ordering and color-mapping
  approach as MacroRegimes2 for consistency if the two are ever shown
  together.
- Output CSVs: prefix `EquityBubbleRegimes_<Artifact>.csv`, one file per
  logical artifact (matches MacroRegimes2's pattern).

Update this file once real plots/outputs exist and conventions are settled.
