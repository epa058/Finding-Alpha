# Workflow Rules — EquityBubbleRegimes

## Status

Greenfield. Nothing built yet. Unlike MacroRegimes2 (complete, validated
baseline), changes here are net-new — but still build incrementally.

## Scoping rules

- Build incrementally and in this order unless directed otherwise:
  1. Solve or work around the point-in-time S&P 500 constituents problem
     (or explicitly scope-limit to a fixed ticker set as a stopgap — see
     `architecture.md`).
  2. Implement LPPL fitting and validate against known historical bubbles.
  3. Add sentiment/hype index.
  4. Build the dual-stream transformer + Bubble Score.
  5. Backtest.
- Don't jump ahead to later steps before earlier ones are verified working —
  each step's output feeds the next.
- One feature/step per implementation pass; verify before moving on.

## No look-ahead bias

See `code-standards.md` — re-check this for every new transform, label, or
model input, especially anything involving "future" information about a
bubble's outcome (e.g. a crash date) leaking into earlier feature values.

## Do not infer scope

This project's scope is still being defined. If a request implies a design
decision not yet documented here (e.g. choice of constituents data source,
transformer architecture details, integration with MacroRegimes2), flag it
and update `project-overview.md`/`architecture.md` with the decision rather
than silently assuming.

## Updating context

Update `progress-tracker.md` after any meaningful change — this project will
change shape quickly early on, so keep it current. If a design decision is
made (constituents source, fitting library, etc.), record it under
"Architecture Decisions" in `progress-tracker.md` and reflect it in
`project-overview.md`/`architecture.md`.
