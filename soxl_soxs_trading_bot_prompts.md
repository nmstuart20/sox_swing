# SOXL/SOXS Trading Bot — LLM Build Prompts

A sequenced list of prompts to feed an LLM to build an automated trading app that trades **only SOXL and SOXS** (shares or options) using **technical analysis** and **Finnhub news data**, executing through **Alpaca**.

The prompts are ordered so each builds on the last, with clean module boundaries.

---


## 3. Market data ingestion

> Write `data/market_data.py` that pulls historical and real-time OHLCV bars for SOXL and SOXS from Alpaca (1-min, 5-min, daily). Return clean pandas DataFrames indexed by timestamp, handle gaps, and cache recent bars. Include a method to get the latest quote and current price.

## 4. Finnhub news + sentiment

> Write `data/finnhub_data.py` using the Finnhub API to fetch: company news for SOXL/SOXS and the underlying semiconductor sector (e.g. SMH, NVDA, AMD), news sentiment scores, and any relevant economic data. Return a normalized DataFrame with timestamp, headline, source, and a sentiment score. Add simple keyword-based sentiment fallback if Finnhub sentiment is unavailable, and rate-limit handling.

## 5. Technical indicators

> Write `strategy/indicators.py` that computes technical indicators on the OHLCV DataFrames using the `ta` library: RSI, MACD, EMA(9/21/50), Bollinger Bands, ATR, and VWAP. Add functions that return current values plus boolean signal flags (e.g. RSI oversold, MACD bullish cross, price above/below key EMA).

## 6. Signal engine

> Write `strategy/signal_engine.py` that combines technical signals and Finnhub news sentiment into a single trade decision for the SOXL/SOXS pair. Since these are inverse leveraged ETFs, the logic should go long SOXL (or short SOXS) on bullish setups and long SOXS on bearish setups, never holding both. Output a structured signal object: direction, confidence score, and the reasons behind it. Make the weighting between technical and sentiment configurable.

## 7. Risk management

> Write `risk/risk_manager.py` that enforces: max position size as a % of equity, max daily loss limit, max number of trades per day, stop-loss and take-profit levels based on ATR, and a check that prevents holding SOXL and SOXS simultaneously. It should validate or veto any signal before execution and handle forced exits when limits are hit. Note that these are 3x leveraged ETFs, so size conservatively.

## 8. Order execution + position management

> Write `execution/order_manager.py` that takes an approved signal, sizes the position via the risk manager, and places the order through the Alpaca client with bracket stop-loss/take-profit. Handle flipping positions (close SOXS before opening SOXL and vice versa), track open orders, and reconcile fills.

## 9. Options support (optional module)

> Extend the execution layer to optionally trade options on SOXL/SOXS instead of shares. Add a module that fetches the Alpaca options chain, selects contracts by delta/expiration/liquidity, and places single-leg call/put orders mapped from the same bullish/bearish signals. Keep it toggleable via config.

## 10. Main orchestration loop

> Write `main.py` that runs the trading loop: check market hours, pull latest market data and Finnhub news, compute indicators, generate a signal, pass it through risk management, and execute. Include a configurable poll interval, graceful shutdown, and a daily reset of counters. Make it run end-of-day flat (close all positions before market close) as a config option.

## 11. Backtesting

> Write a backtesting harness that replays historical Alpaca bars and Finnhub news through the same signal engine and risk manager, simulating fills with slippage and commission. Report total return, Sharpe, max drawdown, win rate, and a trade log. Reuse the production strategy code so backtest and live behavior match.

## 12. Monitoring + alerts

> Add a monitoring module that logs every decision and trade, computes running P&L, and sends alerts (Slack/email/Discord) on entries, exits, errors, and risk-limit breaches. Include a simple status summary printed each cycle.

## 13. Tests + safety

> Write unit tests for the signal engine, risk manager, and indicator calculations using mocked Alpaca/Finnhub responses. Add an integration test that runs the full loop end-to-end in Alpaca paper mode without placing live orders. Verify the bot never holds SOXL and SOXS at once and never exceeds risk limits.

---

## Notes to feed in alongside these prompts

- **Leverage matters more than entries.** SOXL and SOXS are both 3x leveraged (one long semis, one inverse), so decay and volatility make position sizing and tight risk controls more important than the entry logic itself.
- **Paper first.** Run everything in Alpaca paper mode until the backtest and live behavior reconcile.
- **Options are account-gated.** Confirm Alpaca's options availability and approval level for your account before committing to the options module.
