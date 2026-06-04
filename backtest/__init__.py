"""Historical backtesting harness for the SOXL/SOXS bot.

Replays historical Alpaca bars and Finnhub news through the *exact* production
strategy pipeline — :class:`~strategy.signal_engine.SignalEngine`,
:class:`~risk.risk_manager.RiskManager`, and
:class:`~execution.order_manager.OrderManager` — so backtest and live behavior
are driven by the same code. Only two seams are simulated:

  * :class:`~backtest.broker.SimBroker` — an in-memory stand-in for
    :class:`~execution.alpaca_client.AlpacaClient` that fills orders with
    slippage and commission, holds the bracket (take-profit + stop) and triggers
    a leg intrabar,
    and marks the account to market each bar; and
  * :class:`~backtest.feeds.BacktestFinnhub` — a news/sentiment feed that only
    reveals articles dated at or before the current simulated bar (no
    look-ahead), so :meth:`SignalEngine.evaluate` runs unchanged.

:class:`~backtest.engine.BacktestEngine` drives the bars through the same cycle
structure as :meth:`main.TradingEngine._run_cycle` (reconcile → forced exits →
end-of-day flat → signal → execute), passing the simulated clock explicitly so
daily resets and EOD handling track the replayed session rather than wall time.
"""

from backtest.broker import CostModel, SimBroker, Trade
from backtest.engine import BacktestEngine, BacktestResult, build_backtest_engine
from backtest.feeds import BacktestFinnhub
from backtest.metrics import PerformanceMetrics, compute_metrics, format_report

__all__ = [
    "CostModel",
    "SimBroker",
    "Trade",
    "BacktestEngine",
    "BacktestResult",
    "build_backtest_engine",
    "BacktestFinnhub",
    "PerformanceMetrics",
    "compute_metrics",
    "format_report",
]
