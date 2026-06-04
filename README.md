# SOXL/SOXS Trading Bot

An automated trading bot that trades **only SOXL and SOXS** (shares or options)
using technical analysis and Finnhub news data, executing through **Alpaca**.

> ⚠️ SOXL and SOXS are 3x leveraged ETFs. Run everything in Alpaca **paper mode**
> until backtest and live behavior reconcile. Trade at your own risk.

## Project layout

```
config/      # settings loader + logging setup
data/        # Finnhub news/sentiment + Alpaca market data ingestion
strategy/    # technical indicators + combined signal engine
execution/   # Alpaca client + order/position management
risk/        # position sizing, loss limits, signal vetoes
monitoring/  # P&L tracking, per-cycle status summary, Discord alerts
logs/        # rotating log files (gitignored)
tests/       # unit + end-to-end paper-mode tests (pytest)
main.py      # orchestration entry point
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

## Run

```bash
python main.py
```

## Backtesting

The `backtest/` harness replays historical Alpaca bars and Finnhub news through
the **same** signal engine, risk manager, and order manager the live bot uses,
so backtest and live behavior are driven by identical code. Only two seams are
simulated:

* a **simulated broker** (`backtest/broker.py`) that fills orders at market with
  configurable **slippage** and **commission**, holds the attached OTO
  stop-loss and triggers it intrabar (modelling gap-throughs), and marks the
  account to market each bar; and
* a **point-in-time news feed** (`backtest/feeds.py`) that only reveals articles
  dated at or before the current bar — no look-ahead — so the engine's real
  sentiment-blending path runs unchanged.

The driver steps a simulated clock over the bars using the same cycle structure
as the live loop (reconcile → forced exits → end-of-day flat → signal →
execute), passing the simulated time explicitly so daily resets and EOD
handling track the replayed session.

```bash
# Replay a window through the live strategy (keys read from .env):
python -m backtest --start 2026-05-20 --end 2026-06-03

# Bigger account, wider slippage, a per-share commission, write the trade log:
python -m backtest --start 2026-05-01 --capital 250000 \
    --slippage-bps 5 --commission-per-share 0.005 --trade-log trades.csv

# Technicals only (skip the Finnhub news pull):
python -m backtest --start 2026-05-20 --no-news
```

It reports **total return, annualized Sharpe, max drawdown, win rate** (plus
profit factor and average win/loss), and prints a per-trade **trade log**
(entry/exit, net P&L, return, bars held, and exit reason: stop-loss / flip /
end-of-day / forced). `--trade-log PATH` also writes the full log to CSV.

## Tests

The suite is fully offline — Alpaca and Finnhub are replaced with in-memory
fakes, so no API keys, network, or live orders are involved. `pytest` is
included in `requirements.txt`.

```bash
pytest                        # run everything
pytest tests/test_risk_manager.py   # a single module
pytest -v                     # verbose, per-test output
```

It covers unit tests for the indicator calculations, signal engine, and risk
manager, plus an end-to-end test that runs the full trading loop in paper mode
and verifies the bot never holds SOXL and SOXS at once and never exceeds its
risk limits.

## Monitoring & alerts

Every cycle the bot logs its decisions and trades, tracks running P&L
(session-to-date and intraday), and prints a compact status line. It also pushes
**Discord alerts** on entries, exits (forced / flip / end-of-day), errors, and
risk-limit breaches.

Alerts are opt-in via environment variables (all optional; defaults shown):

| Variable | Default | Purpose |
| --- | --- | --- |
| `ALERTS_ENABLED` | `true` | Master switch for Discord alerts |
| `DISCORD_WEBHOOK_URL` | _(empty)_ | Incoming-webhook URL; empty = alerts disabled |
| `ALERT_MIN_LEVEL` | `INFO` | Minimum severity to send: `INFO`/`SUCCESS`/`WARNING`/`ERROR` |
| `BOT_NAME` | `trade_bot` | Display name on webhook messages |

Delivery is non-blocking (a background worker thread) and best-effort: a slow or
failing webhook never stalls or crashes the trading loop. With no webhook URL the
notifier is a silent no-op, so the bot runs fine without Discord configured.
