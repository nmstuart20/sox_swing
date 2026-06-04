"""Technical indicators for the SOXL/SOXS strategy.

Computes a standard set of indicators on the OHLCV DataFrames produced by
:mod:`data.market_data` using the ``ta`` library, then distills the latest bar
into an :class:`IndicatorSnapshot` of current values plus boolean signal flags
(RSI oversold/overbought, MACD bullish/bearish cross, price vs. key EMAs, etc.).

The signal engine (``strategy/signal_engine.py``) consumes these flags; this
module stays purely technical and has no knowledge of news, risk, or orders.

Input frames are expected to follow :data:`data.market_data.OHLCV_COLUMNS`
(UTC-indexed, columns ``open/high/low/close/volume/trade_count/vwap``). The
``vwap`` column Alpaca returns is a *session* VWAP; the rolling VWAP we compute
here via ``ta`` is a separate, window-based measure exposed as ``vwap_roll``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import VolumeWeightedAveragePrice


class IndicatorError(Exception):
    """Raised when indicators cannot be computed (e.g. too few bars)."""


@dataclass(frozen=True)
class IndicatorParams:
    """Tunable periods/thresholds for the indicator set.

    Defaults are the conventional values; override per-timeframe if needed.
    """

    rsi_window: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50

    bb_window: int = 20
    bb_dev: float = 2.0

    atr_window: int = 14

    vwap_window: int = 14

    @property
    def min_bars(self) -> int:
        """Minimum bar count for the slowest indicator to be meaningful."""
        return max(
            self.rsi_window,
            self.macd_slow + self.macd_signal,
            self.ema_slow,
            self.bb_window,
            self.atr_window,
            self.vwap_window,
        ) + 1


# Indicator columns appended by :func:`compute_indicators`.
INDICATOR_COLUMNS = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_diff",
    "ema_fast",
    "ema_mid",
    "ema_slow",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "bb_pband",
    "bb_wband",
    "atr",
    "vwap_roll",
]


@dataclass(frozen=True)
class IndicatorSnapshot:
    """Latest-bar indicator values plus derived boolean signal flags.

    Values mirror the final row of :func:`compute_indicators`; flags encode the
    common decisions a strategy makes from those values. Flags that hinge on a
    cross compare the last two bars, so a snapshot needs at least two valid rows
    for cross detection (single-bar snapshots simply report ``False``).
    """

    timestamp: pd.Timestamp
    close: float

    # --- raw values ---
    rsi: float
    macd: float
    macd_signal: float
    macd_diff: float
    ema_fast: float
    ema_mid: float
    ema_slow: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    bb_pband: float
    bb_wband: float
    atr: float
    atr_pct: float
    vwap_roll: float

    # --- boolean signal flags ---
    rsi_oversold: bool
    rsi_overbought: bool
    macd_bullish_cross: bool
    macd_bearish_cross: bool
    macd_above_signal: bool
    price_above_ema_fast: bool
    price_above_ema_mid: bool
    price_above_ema_slow: bool
    ema_bullish_stack: bool
    ema_bearish_stack: bool
    price_above_bb_upper: bool
    price_below_bb_lower: bool
    price_above_vwap: bool

    @property
    def bullish_flags(self) -> tuple[str, ...]:
        """Names of the flags currently leaning bullish."""
        leans = {
            "rsi_oversold": self.rsi_oversold,
            "macd_bullish_cross": self.macd_bullish_cross,
            "macd_above_signal": self.macd_above_signal,
            "price_above_ema_fast": self.price_above_ema_fast,
            "price_above_ema_mid": self.price_above_ema_mid,
            "price_above_ema_slow": self.price_above_ema_slow,
            "ema_bullish_stack": self.ema_bullish_stack,
            "price_below_bb_lower": self.price_below_bb_lower,
            "price_above_vwap": self.price_above_vwap,
        }
        return tuple(name for name, on in leans.items() if on)

    @property
    def bearish_flags(self) -> tuple[str, ...]:
        """Names of the flags currently leaning bearish."""
        leans = {
            "rsi_overbought": self.rsi_overbought,
            "macd_bearish_cross": self.macd_bearish_cross,
            "macd_below_signal": not self.macd_above_signal,
            "price_below_ema_fast": not self.price_above_ema_fast,
            "price_below_ema_mid": not self.price_above_ema_mid,
            "price_below_ema_slow": not self.price_above_ema_slow,
            "ema_bearish_stack": self.ema_bearish_stack,
            "price_above_bb_upper": self.price_above_bb_upper,
            "price_below_vwap": not self.price_above_vwap,
        }
        return tuple(name for name, on in leans.items() if on)

    @property
    def net_lean(self) -> int:
        """Crude tally of bullish minus bearish flags (sign = direction).

        A convenience for quick inspection/logging only; the signal engine does
        the real weighting between technicals and sentiment.
        """
        return len(self.bullish_flags) - len(self.bearish_flags)


def compute_indicators(
    df: pd.DataFrame,
    params: IndicatorParams | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with :data:`INDICATOR_COLUMNS` appended.

    Indicators are computed on the ``close`` (and ``high``/``low``/``volume``
    where relevant). Early rows where a window has not yet filled are ``NaN``,
    matching ``ta``'s behavior; callers should rely on the most recent rows.

    Raises:
        IndicatorError: if ``df`` is empty, missing required OHLCV columns, or
            holds fewer than :attr:`IndicatorParams.min_bars` rows.
    """
    params = params or IndicatorParams()

    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise IndicatorError(f"DataFrame missing required columns: {sorted(missing)}")
    if df.empty:
        raise IndicatorError("Cannot compute indicators on an empty DataFrame.")
    if len(df) < params.min_bars:
        # Below this, the slowest window can't fill and some ta indicators
        # (e.g. ATR) index out of bounds rather than returning NaN. Fail clean.
        raise IndicatorError(
            f"Need at least {params.min_bars} bars to compute indicators, "
            f"got {len(df)}. Widen the lookback window."
        )

    out = df.copy()
    close, high, low, volume = out["close"], out["high"], out["low"], out["volume"]

    out["rsi"] = RSIIndicator(close=close, window=params.rsi_window).rsi()

    macd = MACD(
        close=close,
        window_slow=params.macd_slow,
        window_fast=params.macd_fast,
        window_sign=params.macd_signal,
    )
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()  # histogram: macd - signal

    out["ema_fast"] = EMAIndicator(close=close, window=params.ema_fast).ema_indicator()
    out["ema_mid"] = EMAIndicator(close=close, window=params.ema_mid).ema_indicator()
    out["ema_slow"] = EMAIndicator(close=close, window=params.ema_slow).ema_indicator()

    bb = BollingerBands(close=close, window=params.bb_window, window_dev=params.bb_dev)
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_pband"] = bb.bollinger_pband()  # %B: 0 at lower band, 1 at upper
    out["bb_wband"] = bb.bollinger_wband()  # bandwidth as % of middle band

    out["atr"] = AverageTrueRange(
        high=high, low=low, close=close, window=params.atr_window
    ).average_true_range()

    out["vwap_roll"] = VolumeWeightedAveragePrice(
        high=high, low=low, close=close, volume=volume, window=params.vwap_window
    ).volume_weighted_average_price()

    return out


def latest_snapshot(
    df: pd.DataFrame,
    params: IndicatorParams | None = None,
    *,
    precomputed: bool = False,
) -> IndicatorSnapshot:
    """Compute indicators and distill the final bar into an :class:`IndicatorSnapshot`.

    Args:
        df: OHLCV frame, or a frame already carrying :data:`INDICATOR_COLUMNS`
            when ``precomputed=True`` (avoids recomputing in a hot loop).
        params: indicator parameters; defaults to :class:`IndicatorParams`.
        precomputed: treat ``df`` as the output of :func:`compute_indicators`.

    Raises:
        IndicatorError: if there are no bars or the latest bar's core indicators
            are ``NaN`` (typically not enough history).
    """
    params = params or IndicatorParams()
    enriched = df if precomputed else compute_indicators(df, params)

    if enriched.empty:
        raise IndicatorError("No bars available for a snapshot.")

    last = enriched.iloc[-1]
    prev = enriched.iloc[-2] if len(enriched) >= 2 else None

    # Core values that must be present for the snapshot to be meaningful.
    core = ["close", "rsi", "macd", "macd_signal", "ema_fast", "ema_slow", "atr"]
    if last[core].isna().any():
        raise IndicatorError(
            "Latest bar has NaN indicators; supply at least "
            f"~{params.min_bars} bars of history."
        )

    close = float(last["close"])
    macd_val = float(last["macd"])
    macd_sig = float(last["macd_signal"])
    ema_fast = float(last["ema_fast"])
    ema_mid = float(last["ema_mid"])
    ema_slow = float(last["ema_slow"])
    bb_upper = float(last["bb_upper"])
    bb_lower = float(last["bb_lower"])
    atr = float(last["atr"])
    vwap_roll = float(last["vwap_roll"])

    # Cross detection needs the prior bar's MACD-vs-signal relationship.
    macd_above_signal = macd_val > macd_sig
    bullish_cross = bearish_cross = False
    if prev is not None and not pd.isna(prev["macd"]) and not pd.isna(prev["macd_signal"]):
        prev_above = float(prev["macd"]) > float(prev["macd_signal"])
        bullish_cross = macd_above_signal and not prev_above
        bearish_cross = (not macd_above_signal) and prev_above

    return IndicatorSnapshot(
        timestamp=enriched.index[-1],
        close=close,
        rsi=float(last["rsi"]),
        macd=macd_val,
        macd_signal=macd_sig,
        macd_diff=float(last["macd_diff"]),
        ema_fast=ema_fast,
        ema_mid=ema_mid,
        ema_slow=ema_slow,
        bb_upper=bb_upper,
        bb_mid=float(last["bb_mid"]),
        bb_lower=bb_lower,
        bb_pband=float(last["bb_pband"]),
        bb_wband=float(last["bb_wband"]),
        atr=atr,
        atr_pct=(atr / close) if close else 0.0,
        vwap_roll=vwap_roll,
        rsi_oversold=float(last["rsi"]) <= params.rsi_oversold,
        rsi_overbought=float(last["rsi"]) >= params.rsi_overbought,
        macd_bullish_cross=bullish_cross,
        macd_bearish_cross=bearish_cross,
        macd_above_signal=macd_above_signal,
        price_above_ema_fast=close > ema_fast,
        price_above_ema_mid=close > ema_mid,
        price_above_ema_slow=close > ema_slow,
        ema_bullish_stack=ema_fast > ema_mid > ema_slow,
        ema_bearish_stack=ema_fast < ema_mid < ema_slow,
        price_above_bb_upper=close > bb_upper,
        price_below_bb_lower=close < bb_lower,
        price_above_vwap=close > vwap_roll,
    )
