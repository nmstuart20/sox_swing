"""Shared fixtures, factories, and fakes for the SOXL/SOXS test suite.

Everything here is offline: OHLCV frames are synthesized in-memory and the
Alpaca/Finnhub layers are replaced with light fakes, so the whole suite runs
with no network, no API keys, and no real orders.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from config.settings import RiskConfig, StrategyConfig
from data.market_data import OHLCV_COLUMNS
from strategy.indicators import IndicatorSnapshot
from strategy.signal_engine import Direction, TradeSignal


# ----------------------------------------------------------------------
# OHLCV frame factory
# ----------------------------------------------------------------------
def make_ohlcv(
    n: int = 120,
    *,
    start: float = 20.0,
    slope: float = 0.05,
    freq: str = "5min",
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Synthesize a clean OHLCV frame with a constant per-bar ``slope``.

    A positive ``slope`` trends up (bullish technicals), negative trends down
    (bearish), and ``0`` is flat. Columns/index match what
    :mod:`data.market_data` produces, so :func:`strategy.indicators.compute_indicators`
    consumes it unchanged.
    """
    end = end or pd.Timestamp("2026-06-04 20:00", tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq=freq)
    close = start + slope * np.arange(n, dtype=float)
    if (close <= 0).any():
        raise ValueError("make_ohlcv produced non-positive prices; adjust start/slope.")
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.02
    low = np.minimum(open_, close) - 0.02
    volume = np.full(n, 1_000.0)
    trade_count = np.full(n, 10.0)
    vwap = close.copy()
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trade_count": trade_count,
            "vwap": vwap,
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    return frame[OHLCV_COLUMNS]


# ----------------------------------------------------------------------
# Snapshot factory — all flags neutral by default, override per test
# ----------------------------------------------------------------------
def make_snapshot(**overrides) -> IndicatorSnapshot:
    """Build an :class:`IndicatorSnapshot` with sane, flat defaults.

    Defaults encode a do-nothing tape (every boolean flag False / mid-range
    values); pass keyword overrides to flip individual flags for a test.
    """
    base = dict(
        timestamp=pd.Timestamp("2026-06-04 20:00", tz="UTC"),
        close=20.0,
        rsi=50.0,
        macd=0.0,
        macd_signal=0.0,
        macd_diff=0.0,
        ema_fast=20.0,
        ema_mid=20.0,
        ema_slow=20.0,
        bb_upper=21.0,
        bb_mid=20.0,
        bb_lower=19.0,
        bb_pband=0.5,
        bb_wband=10.0,
        atr=0.5,
        atr_pct=0.025,
        vwap_roll=20.0,
        rsi_oversold=False,
        rsi_overbought=False,
        macd_bullish_cross=False,
        macd_bearish_cross=False,
        macd_above_signal=False,
        price_above_ema_fast=False,
        price_above_ema_mid=False,
        price_above_ema_slow=False,
        ema_bullish_stack=False,
        ema_bearish_stack=False,
        price_above_bb_upper=False,
        price_below_bb_lower=False,
        price_above_vwap=False,
    )
    base.update(overrides)
    return IndicatorSnapshot(**base)


def bullish_snapshot(**overrides) -> IndicatorSnapshot:
    """A snapshot with every technical flag leaning bullish."""
    flags = dict(
        macd_bullish_cross=True,
        macd_above_signal=True,
        price_above_ema_fast=True,
        price_above_ema_mid=True,
        price_above_ema_slow=True,
        ema_bullish_stack=True,
        price_below_bb_lower=True,  # mean-reversion bullish tell
        price_above_vwap=True,
        rsi_oversold=True,
    )
    flags.update(overrides)
    return make_snapshot(**flags)


def neutral_snapshot(**overrides) -> IndicatorSnapshot:
    """A snapshot whose technical score sits inside the entry threshold.

    The two-sided flags (price-vs-EMA, MACD-vs-signal, VWAP) vote -1 when False,
    so an all-False snapshot actually reads *bearish*. Here the two-sided flags
    are balanced and every three-state flag is off, leaving a small net score
    (~+0.09 with default weights) that stays NEUTRAL.
    """
    flags = dict(
        price_above_ema_slow=True,
        price_above_ema_mid=True,
        price_above_ema_fast=True,
        macd_above_signal=False,
        price_above_vwap=False,
    )
    flags.update(overrides)
    return make_snapshot(**flags)


def bearish_snapshot(**overrides) -> IndicatorSnapshot:
    """A snapshot with every technical flag leaning bearish."""
    flags = dict(
        macd_bearish_cross=True,
        macd_above_signal=False,
        price_above_ema_fast=False,
        price_above_ema_mid=False,
        price_above_ema_slow=False,
        ema_bearish_stack=True,
        price_above_bb_upper=True,
        price_above_vwap=False,
        rsi_overbought=True,
    )
    flags.update(overrides)
    return make_snapshot(**flags)


def make_signal(
    direction: Direction = Direction.BULLISH,
    *,
    target_symbol: str | None = "SOXL",
    combined_score: float = 0.5,
) -> TradeSignal:
    """A minimal :class:`TradeSignal` for risk/execution tests."""
    return TradeSignal(
        timestamp=pd.Timestamp("2026-06-04 20:00", tz="UTC"),
        direction=direction,
        target_symbol=target_symbol,
        confidence=abs(combined_score),
        combined_score=combined_score,
        technical_score=combined_score,
        sentiment_score=0.0,
        technical_weight=0.7,
        sentiment_weight=0.3,
        reasons=("test",),
    )


# ----------------------------------------------------------------------
# Config fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def strategy_config() -> StrategyConfig:
    return StrategyConfig(technical_weight=0.7, sentiment_weight=0.3)


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        max_position_pct=0.10,
        max_daily_loss_pct=0.03,
        max_trades_per_day=5,
        atr_stop_multiplier=2.0,
        atr_take_profit_multiplier=3.0,
    )


# ----------------------------------------------------------------------
# Fake Alpaca SDK objects
# ----------------------------------------------------------------------
@dataclass
class FakeSide:
    """Stand-in for alpaca's OrderSide enum (only ``.value`` is read)."""

    value: str


@dataclass
class FakePosition:
    """The subset of alpaca's Position the risk/order code reads."""

    symbol: str
    qty: str
    unrealized_pl: str = "0"


class FakeOrder:
    """A submitted order that the paper fake reports as instantly filled."""

    def __init__(self, symbol: str, qty: float, side: str, price: float) -> None:
        self.id = uuid.uuid4().hex
        self.symbol = symbol
        self.qty = str(qty)
        self.side = FakeSide(side)
        self.status = "filled"
        self.filled_qty = str(qty)
        self.filled_avg_price = str(price)
        self.client_order_id = None


class FakeClock:
    def __init__(self, *, is_open: bool = True) -> None:
        now = datetime.now(timezone.utc)
        self.is_open = is_open
        self.next_open = now + timedelta(hours=12)
        # Far enough out that the EOD-flat buffer never triggers in tests.
        self.next_close = now + timedelta(hours=6)


class FakeAlpacaClient:
    """In-memory paper-mode Alpaca stand-in.

    Tracks positions and submitted orders so tests can assert the no-both-legs
    invariant and that nothing ever escapes to a real broker. Bracket entries
    and liquidations report as instantly filled, mimicking a quiet paper fill.
    """

    def __init__(self, *, equity: float = 100_000.0) -> None:
        self.is_paper = True
        self._equity = equity
        self._positions: dict[str, FakePosition] = {}
        self.orders: dict[str, FakeOrder] = {}
        self.submitted: list[FakeOrder] = []
        self.clock = FakeClock(is_open=True)
        self.cancel_all_calls = 0
        self.last_price = 20.0

    # --- account / clock ---
    def set_equity(self, equity: float) -> None:
        self._equity = equity

    def get_equity(self) -> float:
        return self._equity

    def get_clock(self) -> FakeClock:
        return self.clock

    # --- positions ---
    def set_position(self, symbol: str, qty: float, *, unrealized_pl: float = 0.0) -> None:
        if qty == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = FakePosition(symbol, str(qty), str(unrealized_pl))

    def get_position(self, symbol: str) -> FakePosition | None:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_pair_positions(self, symbol_long: str, symbol_short: str) -> dict[str, FakePosition | None]:
        return {
            symbol_long: self._positions.get(symbol_long),
            symbol_short: self._positions.get(symbol_short),
        }

    @property
    def held_symbols(self) -> list[str]:
        return sorted(self._positions)

    # --- orders ---
    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side,
        take_profit_price: float,
        stop_loss_price: float,
        client_order_id: str | None = None,
        **_: object,
    ) -> FakeOrder:
        side_value = getattr(side, "value", str(side))
        order = FakeOrder(symbol, qty, side_value, self.last_price)
        order.client_order_id = client_order_id
        # Paper fill opens the position immediately.
        self.set_position(symbol, qty)
        self.orders[order.id] = order
        self.submitted.append(order)
        return order

    def close_position(self, symbol: str) -> FakeOrder | None:
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        order = FakeOrder(symbol, float(pos.qty), "sell", self.last_price)
        self.set_position(symbol, 0)
        self.orders[order.id] = order
        self.submitted.append(order)
        return order

    def get_order(self, order_id: str) -> FakeOrder:
        return self.orders[order_id]

    def cancel_all_orders(self) -> None:
        self.cancel_all_calls += 1


class FakeFinnhub:
    """A Finnhub stand-in returning a fixed sentiment score for any symbol."""

    def __init__(self, score: float = 0.0) -> None:
        self.score = score
        self.sector_symbols: tuple[str, ...] = ()

    def get_news_sentiment(self, symbol: str, lookback_days: int = 7):
        from data.finnhub_data import SentimentResult

        return SentimentResult(symbol, self.score, article_count=1, source="fake")


class FakeMarketData:
    """Serves canned OHLCV frames and a current price per symbol."""

    def __init__(self, frames: dict[str, pd.DataFrame], prices: dict[str, float]) -> None:
        self._frames = frames
        self._prices = prices

    def get_bars(self, symbol: str, timeframe, *, fill_gaps: bool = False, **_: object) -> pd.DataFrame:
        from data.market_data import MarketDataError

        if symbol not in self._frames:
            raise MarketDataError(f"no fake bars for {symbol}")
        return self._frames[symbol]

    def get_current_price(self, symbol: str) -> float:
        return self._prices[symbol]
