"""Entry point and orchestration loop for the SOXL/SOXS trading bot.

This wires every module built in the earlier steps into a single trading loop:

  1. **Market hours** — gate the whole cycle on Alpaca's clock; nothing trades
     outside the session.
  2. **Market data + news** — pull the latest bars for the pair and (best-effort)
     Finnhub sentiment.
  3. **Indicators + signal** — distill the long leg's bars into an
     :class:`~strategy.indicators.IndicatorSnapshot` and fold in sentiment to get
     a single :class:`~strategy.signal_engine.TradeSignal` for the pair.
  4. **Risk + execution** — pass the signal through the
     :class:`~risk.risk_manager.RiskManager` and let the
     :class:`~execution.order_manager.OrderManager` size, flip, and place the
     bracket.

Around that pipeline the loop also:

  * **reconciles** open orders each cycle so fills register against the daily cap,
  * runs **forced exits** (both-legs invariant breach / daily-loss stop),
  * flattens **end-of-day** before the close when ``CLOSE_AT_EOD`` is set,
  * **resets daily counters** when the trading day rolls over,
  * sleeps a **configurable poll interval** between cycles, interruptibly, and
  * shuts down **gracefully** on SIGINT/SIGTERM.

The loop is deliberately defensive: a transient failure in any one cycle is
logged and swallowed so a single bad poll never kills the bot.
"""

from __future__ import annotations

import signal
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from config import load_settings, setup_logging
from config.logging_setup import get_logger
from config.settings import Settings
from data.finnhub_data import FinnhubData
from data.market_data import MarketData, MarketDataError, Timeframe
from execution.alpaca_client import AlpacaClient, AlpacaClientError
from execution.order_manager import OrderManager
from risk.risk_manager import RiskManager
from strategy.indicators import IndicatorError, IndicatorParams, IndicatorSnapshot, latest_snapshot
from strategy.signal_engine import SignalEngine, TradeSignal

logger = get_logger(__name__)

# Bar resolution the signal pipeline runs on. 5-minute bars give enough history
# (the default lookback is ~5 days) for the slow 50-period EMA while staying
# responsive intraday.
SIGNAL_TIMEFRAME = Timeframe.FIVE_MINUTE


class TradingEngine:
    """Owns the trading loop and the modules it drives.

    Args:
        settings: validated configuration for the whole bot.
        client: Alpaca trading wrapper (account, clock, orders).
        market_data: OHLCV/quote source.
        finnhub: news/sentiment source (optional; ``None`` -> technicals only).
        signal_engine: blends technicals + sentiment into a decision.
        risk_manager: the policy gate that sizes positions and sets brackets.
        order_manager: turns approved decisions into bracket orders.
        indicator_params: tunables for the indicator set.
    """

    def __init__(
        self,
        settings: Settings,
        client: AlpacaClient,
        market_data: MarketData,
        finnhub: FinnhubData | None,
        signal_engine: SignalEngine,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        indicator_params: IndicatorParams | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._market_data = market_data
        self._finnhub = finnhub
        self._signals = signal_engine
        self._risk = risk_manager
        self._orders = order_manager
        self._indicator_params = indicator_params or IndicatorParams()

        self._symbol_long = settings.symbol_long
        self._symbol_short = settings.symbol_short
        self._poll_interval = settings.engine.poll_interval_seconds

        # Set by signal handlers; the loop checks it and sleeps on it so a
        # shutdown interrupts an in-progress wait immediately.
        self._stop = threading.Event()
        # Guards a single EOD flatten per session so we don't re-flatten (and
        # re-cancel) every cycle in the closing window.
        self._eod_flattened = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def request_stop(self, *_: Any) -> None:
        """Signal the loop to finish the current cycle and exit (idempotent)."""
        if not self._stop.is_set():
            logger.info("Shutdown requested — finishing current cycle and stopping")
        self._stop.set()

    def install_signal_handlers(self) -> None:
        """Route SIGINT/SIGTERM to :meth:`request_stop` for a graceful exit."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                logger.warning("Could not install handler for %s", sig)

    def run(self) -> int:
        """Run the trading loop until a stop is requested. Returns an exit code."""
        logger.info(
            "Trading loop starting (poll=%ds, close_at_eod=%s, options=%s)",
            self._poll_interval,
            self._settings.engine.close_at_eod,
            self._settings.engine.use_options,
        )
        if self._settings.engine.use_options:
            logger.warning(
                "USE_OPTIONS is set but the options execution path is not wired; "
                "trading shares via the order manager instead."
            )

        while not self._stop.is_set():
            try:
                self._run_cycle()
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the bot
                logger.exception("Cycle failed, continuing: %s", exc)
            # Interruptible sleep: returns early the moment a stop is requested.
            self._stop.wait(self._poll_interval)

        logger.info("Trading loop stopped")
        return 0

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------
    def _run_cycle(self) -> None:
        now = datetime.now(timezone.utc)
        clock = self._client.get_clock()

        if not clock.is_open:
            self._eod_flattened_reset()
            logger.info(
                "Market closed (next open %s) — idle", _fmt_ts(clock.next_open)
            )
            return

        equity = self._client.get_equity()
        self._risk.maybe_reset_for_day(equity, now=now)

        # Settle any in-flight orders first so fills count and terminal orders
        # are dropped before we reason about positions.
        self._orders.reconcile()

        # Safety net: flatten on an invariant breach (both legs) or daily-loss stop.
        forced = self._orders.handle_forced_exits(equity)
        if forced:
            logger.warning("Forced %d exit(s) this cycle", len(forced))

        # End-of-day flat: stop opening new exposure and close out before the bell.
        if self._should_flatten_for_eod(clock, now):
            if not self._eod_flattened:
                logger.info(
                    "Within EOD buffer (close %s) — flattening for end of day",
                    _fmt_ts(clock.next_close),
                )
                self._orders.close_all(reason="end-of-day flat")
                self._eod_flattened = True
            self._log_status(equity)
            return

        signal = self._build_signal()
        if signal is None:
            self._log_status(equity)
            return

        if not signal.is_actionable:
            logger.info("No actionable signal: %s", signal)
            self._log_status(equity)
            return

        self._execute(signal, equity)
        self._log_status(equity)

    # ------------------------------------------------------------------
    # Signal pipeline
    # ------------------------------------------------------------------
    def _build_signal(self) -> TradeSignal | None:
        """Pull bars for the long leg, snapshot indicators, and generate a signal.

        Returns ``None`` when data is insufficient (the cycle then no-ops); the
        signal engine folds in Finnhub sentiment when a client is configured.
        """
        snapshot = self._snapshot_for(self._symbol_long)
        if snapshot is None:
            return None
        return self._signals.evaluate(snapshot, self._finnhub)

    def _snapshot_for(self, symbol: str) -> IndicatorSnapshot | None:
        """Fetch bars for ``symbol`` and distill the latest :class:`IndicatorSnapshot`."""
        try:
            bars = self._market_data.get_bars(symbol, SIGNAL_TIMEFRAME, fill_gaps=True)
        except MarketDataError as exc:
            logger.warning("Market data unavailable for %s: %s", symbol, exc)
            return None
        try:
            return latest_snapshot(bars, self._indicator_params)
        except IndicatorError as exc:
            logger.warning("Indicators unavailable for %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _execute(self, signal: TradeSignal, equity: float) -> None:
        """Size and place the trade for an actionable signal on its target leg."""
        target = signal.target_symbol
        assert target is not None  # actionable signals always name a target

        # The risk bracket is built from the *target* leg's own price and ATR,
        # which differ from the long leg when the signal favors SOXS.
        target_snapshot = self._snapshot_for(target)
        if target_snapshot is None:
            logger.warning("Skipping entry: no indicator data for target %s", target)
            return

        try:
            entry_price = self._market_data.get_current_price(target)
        except MarketDataError as exc:
            logger.warning("No live price for %s, using last bar close: %s", target, exc)
            entry_price = target_snapshot.close

        result = self._orders.execute_signal(
            signal,
            equity=equity,
            entry_price=entry_price,
            atr=target_snapshot.atr,
        )
        logger.info("Execution result: %s (%s)", result.action, result.message)

    # ------------------------------------------------------------------
    # End-of-day handling
    # ------------------------------------------------------------------
    def _should_flatten_for_eod(self, clock: Any, now: datetime) -> bool:
        """True when EOD flattening is enabled and we're inside the close buffer."""
        if not self._settings.engine.close_at_eod:
            return False
        next_close = clock.next_close
        if next_close is None:
            return False
        if next_close.tzinfo is None:
            next_close = next_close.replace(tzinfo=timezone.utc)
        buffer_seconds = self._settings.engine.eod_flat_buffer_minutes * 60
        return (next_close - now).total_seconds() <= buffer_seconds

    def _eod_flattened_reset(self) -> None:
        """Clear the once-per-session EOD guard while the market is closed."""
        self._eod_flattened = False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _log_status(self, equity: float) -> None:
        status = self._risk.status(equity)
        open_orders = len(self._orders.open_orders)
        logger.info(
            "Status | equity=%.2f drawdown=%.2f%% trades=%d/%d open_orders=%d",
            float(equity),
            status["daily_drawdown"] * 100,
            status["trades_today"],
            status["max_trades_per_day"],
            open_orders,
        )


def _fmt_ts(value: Any) -> str:
    """Compact ISO-ish rendering of a clock timestamp (handles ``None``)."""
    if value is None:
        return "unknown"
    try:
        return value.isoformat(timespec="minutes")
    except (AttributeError, TypeError):
        return str(value)


def build_engine(settings: Settings) -> TradingEngine:
    """Construct the engine and all its collaborators from ``settings``."""
    client = AlpacaClient(settings.alpaca)
    market_data = MarketData(settings.alpaca)

    finnhub: FinnhubData | None = None
    if settings.strategy.sentiment_weight > 0:
        try:
            finnhub = FinnhubData(settings.finnhub)
        except Exception as exc:  # noqa: BLE001 - sentiment is optional, degrade cleanly
            logger.warning("Finnhub init failed, running technicals-only: %s", exc)

    signal_engine = SignalEngine(
        settings.strategy,
        symbol_long=settings.symbol_long,
        symbol_short=settings.symbol_short,
    )
    risk_manager = RiskManager(
        settings.risk,
        symbol_long=settings.symbol_long,
        symbol_short=settings.symbol_short,
    )
    order_manager = OrderManager(
        client,
        risk_manager,
        symbol_long=settings.symbol_long,
        symbol_short=settings.symbol_short,
        poll_interval=min(float(settings.engine.poll_interval_seconds), 2.0),
    )
    return TradingEngine(
        settings,
        client,
        market_data,
        finnhub,
        signal_engine,
        risk_manager,
        order_manager,
    )


def main() -> int:
    try:
        settings = load_settings()
    except ValueError as exc:
        # Logging may not be configured yet, so write straight to stderr.
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    setup_logging(settings.logging)

    mode = "PAPER" if settings.alpaca.paper else "LIVE"
    logger.info("Starting SOXL/SOXS trading bot")
    logger.info("Mode: %s | Symbols: %s/%s", mode, settings.symbol_long, settings.symbol_short)
    logger.info(
        "Risk: max_pos=%.0f%% max_daily_loss=%.0f%% max_trades=%d",
        settings.risk.max_position_pct * 100,
        settings.risk.max_daily_loss_pct * 100,
        settings.risk.max_trades_per_day,
    )
    logger.info(
        "Strategy weights: technical=%.2f sentiment=%.2f | options=%s",
        settings.strategy.technical_weight,
        settings.strategy.sentiment_weight,
        settings.engine.use_options,
    )

    try:
        engine = build_engine(settings)
    except AlpacaClientError as exc:
        logger.error("Failed to connect to Alpaca: %s", exc)
        return 1

    engine.install_signal_handlers()
    return engine.run()


if __name__ == "__main__":
    raise SystemExit(main())
