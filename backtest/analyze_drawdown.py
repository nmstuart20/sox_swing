"""One-off diagnostic: explain a backtest's max drawdown vs. the daily stop.

A long backtest can post a peak-to-trough max drawdown far larger than the
per-day stop-loss limit, and that is *not* a contradiction:

  * the daily stop is re-anchored every session (it caps a single day's loss
    against that day's *starting* equity), while max drawdown is measured once,
    peak-to-trough, across the whole equity curve;
  * a run of modest down-days compounds into a deep cumulative drawdown even
    though every individual day respected the limit ("multi-day grind");
  * the limit is checked at bar boundaries and only force-flattens on the *next*
    cycle, so a single bar can overshoot it — especially on 3x ETFs ("overshoot").

This script reads a saved equity curve (the CSV written by
``python -m backtest --equity-curve curve.csv``) and splits the drawdown into
those two causes, so you can tell which one is driving the number.

Usage::

    python -m backtest.analyze_drawdown curve.csv
    python -m backtest.analyze_drawdown curve.csv --daily-limit 0.05 --top 15
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from backtest.metrics import max_drawdown


def _load_curve(path: str) -> pd.Series:
    """Load a (time, equity) CSV into a tz-aware, time-indexed Series.

    Tolerant of column naming: takes the first datetime-like column as the index
    and the first numeric column as equity, so it also reads a curve someone
    saved by hand.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("equity curve is empty")

    time_col = next(
        (c for c in df.columns if c.lower() in ("time", "timestamp", "date", "datetime")),
        df.columns[0],
    )
    equity_col = next(
        (c for c in df.columns if c.lower() in ("equity", "value", "nav")),
        next((c for c in df.columns if c != time_col), df.columns[-1]),
    )
    idx = pd.DatetimeIndex(pd.to_datetime(df[time_col], utc=True))
    series = pd.Series(df[equity_col].astype(float).values, index=idx).sort_index()
    return series


def _daily_table(equity: pd.Series) -> pd.DataFrame:
    """Per-session start/low/end and the two drawdown measures of interest.

    ``intraday_dd`` is the worst mark within the day vs. that day's *open*
    equity — the quantity the daily stop limit governs. ``day_return`` is the
    day's open-to-close, the per-day building block of the cumulative curve.
    """
    by_day = equity.groupby(equity.index.normalize())
    table = pd.DataFrame(
        {
            "start": by_day.first(),
            "low": by_day.min(),
            "end": by_day.last(),
        }
    )
    table["intraday_dd"] = table["low"] / table["start"] - 1.0
    table["day_return"] = table["end"] / table["start"] - 1.0
    return table


def _drawdown_span(equity: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """The (peak_time, trough_time, depth) of the largest peak-to-trough move."""
    running_peak = equity.cummax()
    dd = equity / running_peak - 1.0
    trough_t = dd.idxmin()
    depth = float(dd.min())
    # The peak is the last all-time high at or before the trough.
    peak_t = equity.loc[:trough_t].idxmax()
    return peak_t, trough_t, depth


def analyze(path: str, daily_limit: float, top: int) -> str:
    equity = _load_curve(path)
    if len(equity) < 2:
        return "Need at least two equity points to analyze."

    overall_dd = max_drawdown(equity)
    peak_t, trough_t, depth = _drawdown_span(equity)
    table = _daily_table(equity)

    # Days where the within-day drawdown breached the daily stop limit. With a
    # working stop these should be rare and barely past the line; a day well
    # past it points to single-bar / leverage overshoot.
    breaches = table[table["intraday_dd"] <= -daily_limit].sort_values("intraday_dd")
    worst_day = table["intraday_dd"].min()

    # Days inside the max-drawdown span: this is the "grind" — how many sessions
    # the worst peak-to-trough decline took to play out.
    span = table.loc[peak_t.normalize() : trough_t.normalize()]
    down_days = int((span["day_return"] < 0).sum())

    lines = [
        "=" * 64,
        " Drawdown attribution",
        "=" * 64,
        f" Equity points     : {len(equity)} "
        f"({equity.index[0]:%Y-%m-%d} → {equity.index[-1]:%Y-%m-%d})",
        f" Daily stop limit  : {daily_limit:.1%} (per-session, vs. day open)",
        "-" * 64,
        f" Max drawdown      : {overall_dd:.2%}  (peak-to-trough, whole curve)",
        f"   peak            : ${equity.loc[peak_t]:,.2f} at {peak_t:%Y-%m-%d %H:%M}",
        f"   trough          : ${equity.loc[trough_t]:,.2f} at {trough_t:%Y-%m-%d %H:%M}",
        f"   span            : {span.shape[0]} sessions, {down_days} of them down",
        "-" * 64,
        " Reason #1 — multi-day grind",
        f"   The worst drawdown unfolded over {span.shape[0]} session(s); compounding",
        f"   modest down-days alone can reach {overall_dd:.1%} with every day in-limit.",
        "-" * 64,
        " Reason #2 — single-day overshoot of the daily stop",
        f"   Worst single-session intraday drawdown : {worst_day:.2%}",
        f"   Sessions breaching the {daily_limit:.0%} limit          : {len(breaches)}",
    ]

    if not breaches.empty:
        show = breaches.head(top)
        lines.append("   " + "-" * 58)
        lines.append(f"   {'date':<12} {'start':>12} {'low':>12} {'intraday_dd':>12}")
        for day, row in show.iterrows():
            date_str = f"{day:%Y-%m-%d}"
            lines.append(
                f"   {date_str:<12} {row['start']:>12,.2f} "
                f"{row['low']:>12,.2f} {row['intraday_dd']:>11.2%}"
            )
        if len(breaches) > top:
            lines.append(f"   … {len(breaches) - top} more breaching session(s)")

    # A one-line verdict so the reader doesn't have to interpret the tables.
    lines.append("-" * 64)
    if breaches.empty:
        verdict = (
            "Every session stayed within the daily stop — the drawdown is a "
            "multi-day grind (reason #1), a strategy-quality issue, not a stop bug."
        )
    elif worst_day <= -(daily_limit + 0.02):
        verdict = (
            f"At least one session overshot the stop by a wide margin "
            f"(worst {worst_day:.1%}); investigate intra-bar / leverage gaps (reason #2)."
        )
    else:
        verdict = (
            "Breaching sessions only just cross the limit (expected bar-boundary "
            "slop); the bulk of the drawdown is the multi-day grind (reason #1)."
        )
    lines.append(f" Verdict: {verdict}")
    lines.append("=" * 64)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m backtest.analyze_drawdown",
        description="Explain a backtest's max drawdown relative to the daily stop limit.",
    )
    p.add_argument("curve", help="Equity-curve CSV (from `python -m backtest --equity-curve`).")
    p.add_argument(
        "--daily-limit", type=float, default=None,
        help="Daily stop limit as a fraction (default: read max_daily_loss_pct from settings).",
    )
    p.add_argument("--top", type=int, default=10, help="Max breaching sessions to list.")
    args = p.parse_args(argv)

    daily_limit = args.daily_limit
    if daily_limit is None:
        try:
            from config import load_settings

            daily_limit = load_settings().risk.max_daily_loss_pct
        except Exception:  # noqa: BLE001 - fall back to the spec default if config is unavailable
            daily_limit = 0.05

    try:
        print(analyze(args.curve, daily_limit=daily_limit, top=args.top))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not analyze {args.curve}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
