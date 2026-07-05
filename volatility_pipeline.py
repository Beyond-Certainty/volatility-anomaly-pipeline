"""
Volatility Anomaly Pipeline
============================
Tracks a volatility-based anomaly signal for XOM, CVX, and SPY.

Pipeline stages (each is its own function, on purpose, so each piece
can be read, tested, and explained on its own):

  1. FETCH    - pull ~120 trading days of price/volume per ticker.
                Tries yfinance first, falls back to Stooq (direct CSV
                pull) if yfinance is empty or errors.
  2. COMPUTE  - daily returns, rolling volatility, z-scores, and
                anomaly flags (per ticker + the XOM/CVX pair spread).
  3. FORMAT   - turns those numbers into a short, readable daily brief.
  4. OUTPUT   - prints the brief and saves it to a timestamped .txt file.

Run with:  python volatility_pipeline.py
"""

import io
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

# ------------------------------------------------------------------------
# CONFIG - the knobs you're most likely to want to tweak live here,
# so you never have to go hunting inside the functions below.
# ------------------------------------------------------------------------
TICKERS = ["XOM", "CVX", "SPY"]
LOOKBACK_DAYS = 120       # ~ trading days of history to pull per ticker
ROLL_WINDOW = 20          # trading days used for the rolling volatility baseline
Z_FLAG_THRESHOLD = 2.5    # sigma threshold that counts as "anomalous"
OUTPUT_DIR = "briefs"     # sub-folder where each day's .txt brief gets saved


# ------------------------------------------------------------------------
# STAGE 1: FETCH
# ------------------------------------------------------------------------
def fetch_ticker_data(ticker, lookback_days=LOOKBACK_DAYS):
    """
    Get ~lookback_days of daily Close price + Volume for ONE ticker.

    Tries yfinance first. If that raises an error OR comes back empty
    (both happen -- Yahoo's endpoints shift around and yfinance is a
    scraper, not an official API, so it periodically breaks), we fall
    back to Stooq's own CSV export.

    Returns:
        (dataframe, source_name)
        dataframe has columns ['Close', 'Volume'], indexed by date,
        oldest-to-newest. source_name is "yfinance", "stooq", or
        None if both sources failed.
    """
    # Ask for extra calendar days beyond lookback_days, since weekends
    # and holidays mean trading days are fewer than calendar days.
    calendar_buffer_days = int(lookback_days * 1.7) + 15
    end = datetime.today()
    start = end - timedelta(days=calendar_buffer_days)

    # ---- attempt 1: yfinance ----
    try:
        import yfinance as yf
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,       # gives split/dividend-adjusted Close
            multi_level_index=False,  # keep columns flat: 'Close', 'Volume'...
            progress=False,
        )
        if raw is not None and not raw.empty:
            df = raw[["Close", "Volume"]].dropna().tail(lookback_days)
            if len(df) > 0:
                return df, "yfinance"
        print(f"  [fetch] yfinance returned no data for {ticker}, trying Stooq...")
    except Exception as e:
        print(f"  [fetch] yfinance failed for {ticker}: {e}, trying Stooq...")

    # ---- attempt 2: Stooq fallback ----
    # NOTE: pandas-datareader's built-in Stooq reader was removed upstream
    # (the maintainers pulled it along with several other readers that
    # depended on "defunct or heavily broken upstream APIs" - see their
    # changelog). So instead of pandas-datareader, we pull Stooq's own
    # public CSV export directly. Same data source, one less dependency
    # that can go stale on us.
    try:
        url = (
            "https://stooq.com/q/d/l/"
            f"?s={ticker.lower()}&d1={start.strftime('%Y%m%d')}"
            f"&d2={end.strftime('%Y%m%d')}&i=d"
        )
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text))
        if raw is not None and not raw.empty and "Close" in raw.columns:
            raw["Date"] = pd.to_datetime(raw["Date"])
            raw = raw.set_index("Date").sort_index()
            df = raw[["Close", "Volume"]].dropna().tail(lookback_days)
            if len(df) > 0:
                return df, "stooq"
    except Exception as e:
        print(f"  [fetch] stooq failed for {ticker}: {e}")

    return None, None


def fetch_all_data(tickers, lookback_days=LOOKBACK_DAYS):
    """
    Loop fetch_ticker_data() over every ticker we care about, and keep
    track of which source worked for each one (or whether it failed
    entirely). This is what lets a single bad ticker degrade gracefully
    instead of taking down the whole run.

    Returns:
        data    : dict[ticker -> DataFrame]   (only successful fetches)
        sources : dict[ticker -> "yfinance"/"stooq"/None]
        failed  : list of tickers where BOTH sources failed
    """
    data, sources, failed = {}, {}, []

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df, source = fetch_ticker_data(ticker, lookback_days)
        sources[ticker] = source
        if df is not None:
            data[ticker] = df
            print(f"  -> success via {source} ({len(df)} rows)")
        else:
            failed.append(ticker)
            print("  -> FAILED (both yfinance and Stooq were unavailable)")

    return data, sources, failed


# ------------------------------------------------------------------------
# STAGE 2: COMPUTE
# ------------------------------------------------------------------------
def compute_signals(data, roll_window=ROLL_WINDOW, z_threshold=Z_FLAG_THRESHOLD):
    """
    For each ticker we have data for:
      - daily returns (% change in Close, day over day)
      - a rolling 20-day mean/std of those returns, computed on the
        PRIOR 20 days (not including today) -- so a big move today
        doesn't dilute its own baseline and mute its own alarm
      - a z-score for today: (today's return - recent mean) / recent std
      - a flag if |z| > z_threshold

    Also computes the XOM/CVX return-spread signal (if both are
    present): the daily difference between XOM's and CVX's returns,
    z-scored the same way. This is the "correlated pair diverging"
    signal.

    Returns a dict keyed by ticker, plus a "pair_spread" key.
    """
    signals = {}
    returns_lookup = {}

    for ticker, df in data.items():
        returns = df["Close"].pct_change().dropna()
        returns_lookup[ticker] = returns

        rolling_mean = returns.rolling(roll_window).mean().shift(1)
        rolling_std = returns.rolling(roll_window).std().shift(1)
        z_series = (returns - rolling_mean) / rolling_std

        latest_return = returns.iloc[-1]
        latest_z = z_series.iloc[-1]
        latest_std = rolling_std.iloc[-1]
        is_valid = pd.notna(latest_z) and pd.notna(latest_std) and latest_std > 0

        signals[ticker] = {
            "latest_close": df["Close"].iloc[-1],
            "latest_return_pct": latest_return * 100,
            "z_score": latest_z if is_valid else None,
            "flagged": bool(is_valid and abs(latest_z) > z_threshold),
            "valid": is_valid,
        }

    # ---- XOM / CVX pair-spread signal ----
    if "XOM" in returns_lookup and "CVX" in returns_lookup:
        # Build from a dict of Series so pandas aligns by date automatically;
        # dropna() then keeps only days both tickers actually have data for.
        aligned = pd.DataFrame({
            "XOM": returns_lookup["XOM"],
            "CVX": returns_lookup["CVX"],
        }).dropna()

        spread = aligned["XOM"] - aligned["CVX"]
        spread_mean = spread.rolling(roll_window).mean().shift(1)
        spread_std = spread.rolling(roll_window).std().shift(1)
        spread_z_series = (spread - spread_mean) / spread_std

        latest_spread = spread.iloc[-1]
        latest_spread_z = spread_z_series.iloc[-1]
        latest_spread_std = spread_std.iloc[-1]
        is_valid = (
            pd.notna(latest_spread_z)
            and pd.notna(latest_spread_std)
            and latest_spread_std > 0
        )

        signals["pair_spread"] = {
            "available": True,
            "latest_spread_pct": latest_spread * 100,
            "z_score": latest_spread_z if is_valid else None,
            "flagged": bool(is_valid and abs(latest_spread_z) > z_threshold),
            "valid": is_valid,
        }
    else:
        signals["pair_spread"] = {"available": False}

    return signals


# ------------------------------------------------------------------------
# STAGE 3: FORMAT
# ------------------------------------------------------------------------
def format_brief(date_str, signals, sources, failed):
    """
    Turn the numbers in `signals` into a short, readable morning note:
    a couple of lines per ticker, a line for the pair signal, and a
    one-line summary up top. Plain ASCII only (no special symbols),
    so it displays correctly in any terminal or text editor.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"DAILY VOLATILITY & ANOMALY BRIEF - {date_str}")
    lines.append("=" * 60)
    lines.append("")

    any_flag = False

    for ticker in TICKERS:
        if ticker in failed:
            lines.append(f"{ticker}: DATA UNAVAILABLE (both yfinance and Stooq failed)")
            lines.append("")
            continue

        s = signals[ticker]
        src = sources.get(ticker, "unknown")

        if not s["valid"]:
            lines.append(f"{ticker}: not enough history yet for a reliable z-score (source: {src})")
            lines.append("")
            continue

        if s["flagged"]:
            any_flag = True
            status = "*** ANOMALY FLAGGED ***"
        else:
            status = "normal range"

        lines.append(f"{ticker}  (source: {src})")
        lines.append(f"  Latest close : {s['latest_close']:.2f}")
        lines.append(f"  Today's move : {s['latest_return_pct']:+.2f}%   ({s['z_score']:+.2f} sigma)   {status}")
        lines.append("")

    lines.append("-" * 60)
    lines.append("XOM / CVX PAIR-SPREAD SIGNAL  (is the correlated pair diverging?)")
    pair = signals.get("pair_spread", {"available": False})

    if not pair.get("available"):
        lines.append("  Not available - need both XOM and CVX data to compute this.")
    elif not pair["valid"]:
        lines.append("  Not enough history yet for a reliable spread z-score.")
    else:
        if pair["flagged"]:
            any_flag = True
            status = "*** ANOMALY FLAGGED - pair may be diverging ***"
        else:
            status = "normal range"
        lines.append(f"  Return spread today: {pair['latest_spread_pct']:+.2f}%   ({pair['z_score']:+.2f} sigma)   {status}")
    lines.append("-" * 60)
    lines.append("")

    if any_flag:
        lines.append("SUMMARY: One or more anomalies flagged today - see above.")
    elif len(failed) == len(TICKERS):
        lines.append("SUMMARY: Could not fetch data for ANY ticker today. Check your internet connection.")
    elif failed:
        lines.append(f"SUMMARY: No anomalies in the data we could fetch, but {', '.join(failed)} failed to load.")
    else:
        lines.append("SUMMARY: No anomalies detected today. Everything in normal range.")

    return "\n".join(lines)


# ------------------------------------------------------------------------
# STAGE 4: OUTPUT
# ------------------------------------------------------------------------
def safe_print(text):
    """
    Print text, falling back to plain ASCII if the console can't
    handle it. This mostly guards against older Windows terminals
    that don't default to UTF-8 -- without this, a display quirk on
    your machine could crash a run that otherwise worked perfectly.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def save_and_print_brief(brief_text, output_dir=OUTPUT_DIR):
    """
    Print the brief to the console AND save it to a timestamped .txt
    file, so every day's run leaves a permanent, dated record.
    Returns the path of the file that was written.
    """
    safe_print("\n" + brief_text + "\n")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"brief_{timestamp}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(brief_text)

    return filepath


# ------------------------------------------------------------------------
# ORCHESTRATION
# ------------------------------------------------------------------------
def main():
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"Running volatility anomaly pipeline for {today_str}...\n")

    data, sources, failed = fetch_all_data(TICKERS, LOOKBACK_DAYS)
    signals = compute_signals(data)
    brief = format_brief(today_str, signals, sources, failed)
    filepath = save_and_print_brief(brief)

    print(f"Brief saved to: {filepath}")


if __name__ == "__main__":
    main()
