# Workflow Rules — MacroRegimes2

## Status

This notebook is **complete and validated** for its original scope (3 regime
classifiers, all cross-checked against NBER recessions, outputs saved). Treat
changes here as *extensions/improvements to a working baseline*, not a
greenfield build.

## Scoping rules

- Work on one section (or one model/approach) at a time. Don't refactor
  earlier sections as a side effect of changes to a later one.
- Before changing a transform that earlier sections/models depend on (e.g.
  the quantile standardization in section 3, or `sort_by_spread`), check
  whether the Bayesian tree / PV-RF / HMM outputs would need to be
  regenerated — and say so explicitly rather than silently leaving them
  stale.
- Prefer small, verifiable increments: run the affected cells and sanity
  check outputs (recession overlap %, regime counts, transition matrix) match
  expectations before moving on.

## No look-ahead bias

Re-check `code-standards.md`'s look-ahead section before touching any
rolling/transform/forecast code — this is the most common way new bugs get
introduced into this kind of notebook, and was the explicit reason
MacroRegimes2 superseded MacroRegimes1.

## Do not infer scope

Implement against what's described in `project-overview.md` and
`architecture.md`. If a request would change the regime structure (e.g.
number of regimes, canonical ordering), the data sources, or the output file
naming convention, flag that explicitly rather than changing it silently —
these are referenced by other parts of the workflow (and potentially by
`EquityBubbleRegimes`, which may reuse this methodology).

## Updating context

If implementation changes the architecture, scope, data sources, or
conventions documented in these context files, update the relevant file
*before* continuing (per the root pointer in `CLAUDE.md`). Always update
`progress-tracker.md` after any meaningful change.
