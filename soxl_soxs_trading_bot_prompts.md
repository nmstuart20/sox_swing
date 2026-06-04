# SOXL/SOXS Trading Bot — LLM Build Prompts

A sequenced list of prompts to feed an LLM to build an automated trading app that trades **only SOXL and SOXS** (shares or options) using **technical analysis** and **Finnhub news data**, executing through **Alpaca**.

The prompts are ordered so each builds on the last, with clean module boundaries.

---

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
