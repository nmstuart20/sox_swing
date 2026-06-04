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
