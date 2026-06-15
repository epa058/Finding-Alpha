# Code Standards — EquityBubbleRegimes

## General

- Match MacroRegimes2's conventions where applicable (notebook-based,
  numbered sections, Colab + Drive `project_folder`, file-based I/O) — see
  `../MacroRegimes2/context/code-standards.md`.
- This is research/prototyping code. Don't build abstractions, packages, or
  config systems ahead of need.

## No look-ahead bias (critical)

Same rule as MacroRegimes2, arguably even more important here:

- LPPL fits, sentiment/hype indices, and the Bubble Score must only use data
  available as of the date they're attributed to. A bubble label fit using
  the full historical window (including the future crash) and then applied
  retroactively as a "feature" at earlier dates is look-ahead bias.
- When backtesting, the dual-stream transformer must be trained only on data
  prior to the test period (proper walk-forward / OOS split), matching
  MacroRegimes2's PV-RF OOS approach.
- Point-in-time S&P 500 constituents matter for the same reason — using
  today's constituent list for a 1995 backtest introduces survivorship bias.

## Data sourcing

- Prefer free/already-available sources (FRED, yfinance — same as
  MacroRegimes2) before introducing new paid dependencies.
- If a data source is found for point-in-time S&P 500 constituents, document
  it (method, coverage, known gaps) in `progress-tracker.md` and
  `project-overview.md` immediately — this has been a long-standing blocker.

## Modeling

- Start simple: validate the LPPL fitting + bubble labeling on a handful of
  well-known historical bubbles (dot-com, 2008, GME 2021, etc.) before
  building the full dual-stream transformer.
- Keep the eventual output schema (`EquityBubbleRegimes_*.csv`) consistent
  with one row per `(date, ticker)` for the Bubble Score, so it can later be
  joined with MacroRegimes2's monthly macro regime labels if that integration
  happens.
