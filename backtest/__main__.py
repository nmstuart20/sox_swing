"""CLI for the backtest harness.

Examples::

    # Replay the last two weeks of SOXL/SOXS through the live strategy:
    python -m backtest --start 2026-05-20 --end 2026-06-03

    # Bigger account, wider slippage, a per-share commission, write the log:
    python -m backtest --start 2026-05-01 --capital 250000 \\
        --slippage-bps 5 --commission-per-share 0.005 --trade-log trades.csv

    # Technicals only (skip the Finnhub news pull):
    python -m backtest --start 2026-05-20 --no-news

Configuration (API keys, symbols, risk/strategy params) is read from the same
``.env``/environment the live bot uses, so the backtest is wired exactly like
production — only the broker and news feed are simulated.
"""

from __future__ import annotations

import argparse
import sys

from backtest.broker import CostModel
from backtest.data import load_bars, load_news, parse_window
from backtest.engine import build_backtest_engine
from backtest.metrics import format_report, trades_to_frame
from config import load_settings, setup_logging
from config.logging_setup import get_logger

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backtest",
        description="Replay historical Alpaca bars + Finnhub news through the live strategy.",
    )
    p.add_argument("--start", required=True, help="Window start (YYYY-MM-DD or ISO, UTC).")
    p.add_argument("--end", default=None, help="Window end (default: now).")
    p.add_argument("--capital", type=float, default=100_000.0, help="Starting capital.")
    p.add_argument("--slippage-bps", type=float, default=2.0, help="Per-fill slippage in bps.")
    p.add_argument(
        "--commission-per-share", type=float, default=0.0,
        help="Commission $/share (Alpaca equities are free; default 0).",
    )
    p.add_argument("--commission-pct", type=float, default=0.0, help="Commission as a fraction of notional.")
    p.add_argument("--commission-min", type=float, default=0.0, help="Per-order commission floor.")
    p.add_argument("--no-news", action="store_true", help="Skip Finnhub; run technicals-only.")
    p.add_argument("--trade-log", default=None, help="Write the trade log to this CSV path.")
    p.add_argument("--max-trades-shown", type=int, default=50, help="Trades to print (0 = all).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    setup_logging(settings.logging)

    try:
        start, end = parse_window(args.start, args.end)
    except ValueError as exc:
        print(f"Invalid window: {exc}", file=sys.stderr)
        return 1

    logger.info("Loading history %s → %s", start, end)
    try:
        bars = load_bars(settings, start, end)
    except Exception as exc:  # noqa: BLE001 - surface a clean message on data failure
        print(f"Failed to load bars: {exc}", file=sys.stderr)
        return 1

    news = None
    if not args.no_news and settings.strategy.sentiment_weight > 0:
        news = load_news(settings, start, end)

    cost_model = CostModel(
        slippage_bps=args.slippage_bps,
        commission_per_share=args.commission_per_share,
        commission_pct=args.commission_pct,
        commission_min=args.commission_min,
    )

    try:
        engine = build_backtest_engine(
            settings, bars, initial_capital=args.capital, cost_model=cost_model, news=news
        )
        result = engine.run()
    except ValueError as exc:
        print(f"Backtest could not run: {exc}", file=sys.stderr)
        return 1

    max_shown = None if args.max_trades_shown <= 0 else args.max_trades_shown
    print(format_report(result.metrics, result.trades, max_trades=max_shown))

    if args.trade_log:
        trades_to_frame(result.trades).to_csv(args.trade_log, index=False)
        print(f"\nTrade log written to {args.trade_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
