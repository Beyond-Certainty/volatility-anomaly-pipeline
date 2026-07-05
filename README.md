# Volatility Anomaly Pipeline

A small daily pipeline that watches three instruments — **XOM**, **CVX**, and **SPY** — and flags unusual moves. XOM and CVX are a historically correlated energy pair; SPY is there for market context.

The point of interest isn't the single-name moves — it's the **pair-spread signal**. XOM and CVX normally move together, so the more informative event is when they *stop* moving together. The pipeline flags when the XOM/CVX return spread diverges beyond its recent norm, which a single-stock monitor would miss.

I'm teaching myself to build, and this is a learning project — not a desk-grade tool. I wrote it to understand a data pipeline end to end and to be able to reason about every stage. Feedback on where the logic is naive is welcome.

## What it does

Four stages, each its own function so each can be read and reasoned about on its own:

1. **Fetch** — pulls ~120 trading days of daily close + volume per ticker. Tries `yfinance` first, falls back to Stooq's CSV export if `yfinance` returns empty or errors (it's a scraper, so it breaks periodically).
2. **Compute** — daily returns, a 20-day rolling volatility baseline, a z-score for today's move, and a flag when a move exceeds 2.5σ. Separately, the XOM/CVX return spread, z-scored the same way — the "correlated pair diverging" signal.
3. **Format** — turns the numbers into a short, readable morning brief.
4. **Output** — prints the brief and saves it to a timestamped `.txt` file.

## Sample output

```
============================================================
DAILY VOLATILITY & ANOMALY BRIEF - 2026-07-05
============================================================

XOM  (source: yfinance)
  Latest close : 137.09
  Today's move : +0.59%   (+0.69 sigma)   normal range

CVX  (source: yfinance)
  Latest close : 169.20
  Today's move : +2.12%   (+1.92 sigma)   normal range

SPY  (source: yfinance)
  Latest close : 744.78
  Today's move : -0.13%   (-0.05 sigma)   normal range

------------------------------------------------------------
XOM / CVX PAIR-SPREAD SIGNAL  (is the correlated pair diverging?)
  Return spread today: -1.52%   (-2.32 sigma)   normal range
------------------------------------------------------------

SUMMARY: No anomalies detected today. Everything in normal range.
```

## Running it

Requires Python 3. From the project folder:

```bash
pip install pandas requests yfinance
python volatility_pipeline.py
```

Briefs are saved to a `briefs/` subfolder, one timestamped file per run.

## Known limitations

I'd rather be upfront about where this is rough than pretend it isn't:

- **The z-score assumes roughly normal returns, which markets aren't.** Returns have fat tails, so 2.5σ moves happen more often than a normal distribution implies. The z-score here is a heuristic, not a real probability. A more honest version would use thresholds that account for the actual distribution.
- **Simple returns, not log returns.** Fine for a daily MVP; log returns would be cleaner for anything real.
- **The rolling baseline is ~20 points** — a small, noisy estimate of volatility. The baseline itself wobbles.
- **The pair spread is unweighted** (XOM − CVX). A real pairs signal would beta-weight the two legs rather than assume a 1:1 relationship.

One thing I was deliberate about: the rolling mean/std are **shifted by one day**, so today's move isn't included in the baseline it's measured against — avoiding lookahead bias muting the signal.

## Why these three tickers

XOM and CVX are chosen as a correlated pair so the signal is *relative* — the interesting event is the two of them diverging, not just one name moving. SPY provides broad market context to sanity-check whether a move is idiosyncratic or market-wide.
