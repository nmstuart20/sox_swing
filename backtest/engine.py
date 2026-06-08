"""The backtest driver: replays bars through the production strategy pipeline.

:class:`BacktestEngine` steps a simulated clock over historical 5-minute bars
and, at each bar, runs the *same* decision/execution path the live bot runs in
:meth:`main.TradingEngine._run_cycle`:

    re-mark & trigger stops  →  reconcile orders  →  forced exits
      →  end-of-day flat  →  build signal  →  risk-gate & execute

The only differences from live are the two simulated seams (the
:class:`~backtest.broker.SimBroker` and :class:`~backtest.feeds.BacktestFinnhub`)
and that the simulated time is threaded *explicitly* into the risk manager and
order manager (``now=bar_time``) instead of read from the wall clock — so daily
resets and EOD handling track the replayed session. The signal engine, risk
manager, order manager, and indicators are the production classes, unmodified,
so a setup that trades in the backtest trades the same way live.

Indicators are computed once over the full series (recursive/rolling indicators
give identical per-bar values whether or not later bars exist), then the latest
bar is distilled per step via :func:`strategy.indicators.latest_snapshot` — the
same call ``main`` makes — keeping the replay linear in the number of bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

import pandas as pd

from backtest.broker import CostModel, SimBroker, SimClock, Trade
from backtest.feeds import BacktestFinnhub
from backtest.metrics import PerformanceMetrics, compute_metrics
from config.logging_setup import get_logger
from config.settings import Settings
from execution.order_manager import OrderManager
from risk.risk_manager import RiskManager
from strategy.indicators import (
    IndicatorError,
    IndicatorParams,
    IndicatorSnapshot,
    compute_indicators,
    latest_snapshot,
)
from strategy.signal_engine import SignalEngine, SignalParams

logger = get_logger(__name__)

# Regular US equity session in UTC (ignoring the half-hour DST wobble, which the
# bar timestamps themselves already encode). Used only to drive the simulated
# clock's is_open / next_close for end-of-day flattening.
_SESSION_OPEN_UTC = time(13, 30)
_SESSION_CLOSE_UTC = time(20, 0)


@dataclass(frozen=True)
class BacktestResult:
    """Everything a run produces: metrics, the trade log, and the equity curve."""

    metrics: PerformanceMetrics
    trades: list[Trade]
    equity_curve: list[tuple[datetime, float]]


class BacktestEngine:
    """Replays aligned OHLCV frames through the production strategy pipeline.

    Args:
        broker: the simulated execution venue (holds cash, positions, trades).
        signal_engine / risk_manager / order_manager: the production strategy
            components, with the order manager bound to ``broker``.
        bars: ``{symbol: DataFrame}`` of OHLCV for the long and short legs.
        finnhub: point-in-time news feed, or ``None`` for technicals-only.
        symbol_long / symbol_short: the pair.
        indicator_params: indicator tunables (defaults to the production set).
        eod_flat_buffer_minutes / close_at_eod: end-of-day flattening policy.
    """

    def __init__(
        self,
        broker: SimBroker,
        signal_engine: SignalEngine,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        bars: dict[str, pd.DataFrame],
        *,
        finnhub: BacktestFinnhub | None = None,
        symbol_long: str = "SOXL",
        symbol_short: str = "SOXS",
        indicator_params: IndicatorParams | None = None,
        eod_flat_buffer_minutes: int = 15,
        close_at_eod: bool = True,
    ) -> None:
        self._broker = broker
        self._signals = signal_engine
        self._risk = risk_manager
        self._orders = order_manager
        self._finnhub = finnhub
        self._symbol_long = symbol_long
        self._symbol_short = symbol_short
        self._params = indicator_params or IndicatorParams()
        self._eod_buffer = timedelta(minutes=eod_flat_buffer_minutes)
        self._close_at_eod = close_at_eod

        # Align both legs onto a common timeline and pre-enrich with indicators.
        self._index, self._enriched = self._prepare(bars)

        self._eod_flattened = False
        self._cur_date = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _prepare(
        self, bars: dict[str, pd.DataFrame]
    ) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
        """Intersect both legs' timelines and precompute indicators once."""
        long_df = bars[self._symbol_long]
        short_df = bars[self._symbol_short]
        common = long_df.index.intersection(short_df.index)
        if len(common) < self._params.min_bars + 1:
            raise ValueError(
                f"Need > {self._params.min_bars} overlapping bars to backtest, "
                f"got {len(common)}."
            )
        enriched = {
            self._symbol_long: compute_indicators(long_df.loc[common], self._params),
            self._symbol_short: compute_indicators(short_df.loc[common], self._params),
        }
        return common, enriched

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        """Replay every bar and return the performance summary."""
        logger.info(
            "Backtest starting over %d bars (%s → %s)",
            len(self._index), self._index[0], self._index[-1],
        )
        for i, ts in enumerate(self._index):
            self._step(i, ts.to_pydatetime())

        # Flatten anything still open at the final close so the trade log and
        # realized P&L are complete (open exposure was already marked into the
        # equity curve, so this doesn't move final equity).
        self._broker.force_close_all(reason="backtest-end")

        metrics = compute_metrics(
            self._broker.equity_curve, self._broker.trades, self._broker.total_commission
        )
        logger.info(
            "Backtest done: return %.2f%%, Sharpe %.2f, maxDD %.2f%%, %d trades",
            metrics.total_return * 100, metrics.sharpe,
            metrics.max_drawdown * 100, metrics.num_trades,
        )
        return BacktestResult(metrics, list(self._broker.trades), list(self._broker.equity_curve))

    def _step(self, i: int, now: datetime) -> None:
        """One simulated cycle at bar ``i`` / time ``now`` (mirrors ``_run_cycle``)."""
        ohlc = self._bar_ohlc(i)
        clock = self._clock_for(now)

        # 1. Advance the broker: stops fire against this bar, then re-mark equity.
        #    (Stops rest at the broker and can trigger even outside the session,
        #    exactly as a real resting stop would.)
        self._broker.set_marks({s: ohlc[s]["close"] for s in ohlc})
        self._broker.process_bar(now, ohlc, bar_index=i, clock=clock)

        # Market closed: don't trade (mirrors _run_cycle's clock gate). Clear the
        # once-per-session EOD guard so the next session can flatten again.
        if not clock.is_open:
            self._eod_flattened = False
            return

        # New trading day: clear the once-per-session EOD guard. The risk
        # manager re-anchors the day's start equity / trade count below. (Real
        # intraday data has no overnight bars, so the close gate above may never
        # fire — this date check is what resets across consecutive sessions.)
        if self._cur_date != now.date():
            self._eod_flattened = False
            self._cur_date = now.date()

        equity = self._broker.get_equity()
        self._risk.maybe_reset_for_day(equity, now=now)

        # 2. Settle fills, then run safety-net forced exits (invariant / loss stop).
        self._orders.reconcile()
        self._broker.default_close_reason = "forced-exit"
        self._orders.handle_forced_exits(equity, now=now)

        # 3. End-of-day flat: stop opening new exposure and close out before the bell.
        if self._should_flatten_for_eod(clock, now):
            if not self._eod_flattened:
                self._broker.default_close_reason = "eod-flat"
                self._orders.close_all(reason="end-of-day flat")
                self._eod_flattened = True
            return

        # 4. Build the signal on the long leg (same pipeline as live).
        signal = self._build_signal(i, now)
        if signal is None or not signal.is_actionable:
            return

        # 5. Size, gate, and execute on the target leg.
        target = signal.target_symbol
        assert target is not None
        target_snap = self._snapshot(target, i)
        if target_snap is None:
            return
        entry_price = ohlc[target]["close"]
        self._broker.default_close_reason = "flip"
        # Fills are synchronous in the sim, so skip the (real-time) fill wait and
        # settle the entry against the daily trade counter right away — the live
        # bot's wait_for_fill=True achieves the same in-cycle registration.
        self._orders.execute_signal(
            signal, equity=equity, entry_price=entry_price, atr=target_snap.atr,
            now=now, wait_for_fill=False,
        )
        self._orders.reconcile()

    # ------------------------------------------------------------------
    # Signal pipeline (parallels TradingEngine._build_signal / _snapshot_for)
    # ------------------------------------------------------------------
    def _build_signal(self, i: int, now: datetime):
        snapshot = self._snapshot(self._symbol_long, i)
        if snapshot is None:
            return None
        if self._finnhub is not None:
            self._finnhub.set_time(now)
        return self._signals.evaluate(snapshot, self._finnhub)

    def _snapshot(self, symbol: str, i: int) -> IndicatorSnapshot | None:
        """Distill the latest snapshot from pre-enriched bars up to index ``i``."""
        frame = self._enriched[symbol].iloc[: i + 1]
        try:
            return latest_snapshot(frame, self._params, precomputed=True)
        except IndicatorError:
            # Warmup: the slow indicators haven't filled yet — no decision.
            return None

    # ------------------------------------------------------------------
    # Bar / clock helpers
    # ------------------------------------------------------------------
    def _bar_ohlc(self, i: int) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for symbol, frame in self._enriched.items():
            row = frame.iloc[i]
            out[symbol] = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        return out

    def _clock_for(self, now: datetime) -> SimClock:
        """A simulated market clock for ``now`` based on regular session hours."""
        t = now.timetz().replace(tzinfo=None)
        is_open = _SESSION_OPEN_UTC <= t < _SESSION_CLOSE_UTC
        close_dt = datetime.combine(now.date(), _SESSION_CLOSE_UTC, tzinfo=timezone.utc)
        open_dt = datetime.combine(now.date(), _SESSION_OPEN_UTC, tzinfo=timezone.utc)
        next_open = open_dt if now < open_dt else open_dt + timedelta(days=1)
        return SimClock(is_open=is_open, next_open=next_open, next_close=close_dt)

    def _should_flatten_for_eod(self, clock: SimClock, now: datetime) -> bool:
        """True when EOD flattening is on and we're inside the close buffer."""
        if not self._close_at_eod:
            return False
        return (clock.next_close - now) <= self._eod_buffer


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def build_backtest_engine(
    settings: Settings,
    bars: dict[str, pd.DataFrame],
    *,
    initial_capital: float = 100_000.0,
    cost_model: CostModel | None = None,
    news: dict[str, pd.DataFrame] | None = None,
    indicator_params: IndicatorParams | None = None,
) -> BacktestEngine:
    """Assemble a backtest from ``settings`` — the analogue of ``main.build_engine``.

    Uses the same component wiring the live bot uses (signal engine, risk
    manager, order manager), but over the :class:`SimBroker` and the point-in-time
    news feed. Sentiment is replayed only when ``news`` is supplied *and* the
    strategy's sentiment weight is positive, mirroring ``build_engine``'s gate.
    """
    broker = SimBroker(
        initial_capital=initial_capital,
        cost_model=cost_model,
        trailing_stop=settings.risk.trailing_stop,
    )

    finnhub: BacktestFinnhub | None = None
    if news is not None and settings.strategy.sentiment_weight > 0:
        from data.finnhub_data import DEFAULT_SECTOR_SYMBOLS

        finnhub = BacktestFinnhub(
            news,
            sector_symbols=DEFAULT_SECTOR_SYMBOLS,
            sentiment_method=settings.strategy.sentiment_method,
        )

    signal_engine = SignalEngine(
        settings.strategy,
        symbol_long=settings.symbol_long,
        symbol_short=settings.symbol_short,
        params=SignalParams(entry_threshold=settings.strategy.entry_threshold),
    )
    risk_manager = RiskManager(
        settings.risk,
        symbol_long=settings.symbol_long,
        symbol_short=settings.symbol_short,
    )
    # Synchronous fills: a tiny (but non-zero) timeout lets the reused
    # order-manager wait loops observe the already-flat / already-filled state on
    # their first check and return without ever sleeping.
    order_manager = OrderManager(
        broker,  # type: ignore[arg-type]  # SimBroker is AlpacaClient-shaped
        risk_manager,
        symbol_long=settings.symbol_long,
        symbol_short=settings.symbol_short,
        flip_timeout=1.0,
        fill_timeout=1.0,
        poll_interval=0.0,
    )
    return BacktestEngine(
        broker,
        signal_engine,
        risk_manager,
        order_manager,
        bars,
        finnhub=finnhub,
        symbol_long=settings.symbol_long,
        symbol_short=settings.symbol_short,
        indicator_params=indicator_params,
        eod_flat_buffer_minutes=settings.engine.eod_flat_buffer_minutes,
        close_at_eod=settings.engine.close_at_eod,
    )


__all__ = ["BacktestEngine", "BacktestResult", "build_backtest_engine"]
