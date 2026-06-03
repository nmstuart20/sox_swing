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

## Build progress

This repo is built incrementally per `soxl_soxs_trading_bot_prompts.md`.
Step 1 (project scaffolding) is complete: directory structure, dependencies,
`.env.example`, a validated config loader, and console + rotating-file logging.
