"""Unit tests for the technical indicator layer."""

from __future__ import annotations

import pandas as pd
import pytest

from strategy.indicators import (
    INDICATOR_COLUMNS,
    IndicatorError,
    IndicatorParams,
    compute_indicators,
    latest_snapshot,
)
from tests.conftest import make_ohlcv


# ----------------------------------------------------------------------
# compute_indicators — input validation
# ----------------------------------------------------------------------
def test_missing_columns_raises():
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(IndicatorError, match="missing required columns"):
        compute_indicators(df)


def test_empty_frame_raises():
    df = make_ohlcv(n=120).iloc[0:0]
    with pytest.raises(IndicatorError):
        compute_indicators(df)


def test_too_few_bars_raises():
    params = IndicatorParams()
    df = make_ohlcv(n=params.min_bars - 1)
    with pytest.raises(IndicatorError, match="at least"):
        compute_indicators(df)


def test_min_bars_is_driven_by_slowest_window():
    # Default: ema_slow=50 dominates; macd_slow+signal=35; +1 buffer.
    assert IndicatorParams().min_bars == 51


# ----------------------------------------------------------------------
# compute_indicators — output shape & values
# ----------------------------------------------------------------------
def test_appends_all_indicator_columns_without_dropping_ohlcv():
    df = make_ohlcv(n=120)
    out = compute_indicators(df)
    for col in INDICATOR_COLUMNS:
        assert col in out.columns
    # Original OHLCV columns are preserved and the input is not mutated.
    assert "close" in out.columns
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    assert len(out) == len(df)


def test_rsi_within_bounds_and_last_row_populated():
    out = compute_indicators(make_ohlcv(n=120))
    rsi = out["rsi"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()
    # The final bar's core indicators must be real numbers.
    assert not out.iloc[-1][["rsi", "macd", "ema_slow", "atr"]].isna().any()


# ----------------------------------------------------------------------
# latest_snapshot — uptrend vs downtrend flags
# ----------------------------------------------------------------------
def test_uptrend_snapshot_flags_bullish():
    snap = latest_snapshot(make_ohlcv(n=120, start=10.0, slope=0.1))
    assert snap.ema_bullish_stack
    assert not snap.ema_bearish_stack
    assert snap.price_above_ema_slow
    assert snap.price_above_ema_mid
    assert snap.macd_above_signal
    assert snap.net_lean > 0


def test_downtrend_snapshot_flags_bearish():
    snap = latest_snapshot(make_ohlcv(n=120, start=30.0, slope=-0.1))
    assert snap.ema_bearish_stack
    assert not snap.ema_bullish_stack
    assert not snap.price_above_ema_slow
    assert not snap.macd_above_signal
    assert snap.net_lean < 0


def test_atr_pct_is_atr_over_close():
    snap = latest_snapshot(make_ohlcv(n=120))
    assert snap.atr_pct == pytest.approx(snap.atr / snap.close)


# ----------------------------------------------------------------------
# latest_snapshot — precomputed path & MACD cross detection
# ----------------------------------------------------------------------
def _precomputed_two_bar(prev_macd, prev_sig, last_macd, last_sig) -> pd.DataFrame:
    """A 2-row frame carrying every INDICATOR_COLUMN, for cross-detection tests."""
    idx = pd.date_range("2026-06-04 19:55", periods=2, freq="5min", tz="UTC")
    row = dict(
        open=20.0, high=20.5, low=19.5, close=20.0, volume=1000.0,
        trade_count=10.0, vwap=20.0, rsi=50.0,
        macd_diff=0.0, ema_fast=20.0, ema_mid=20.0, ema_slow=20.0,
        bb_upper=21.0, bb_mid=20.0, bb_lower=19.0, bb_pband=0.5, bb_wband=10.0,
        atr=0.5, vwap_roll=20.0,
    )
    df = pd.DataFrame([dict(row), dict(row)], index=idx)
    df.loc[idx[0], ["macd", "macd_signal"]] = [prev_macd, prev_sig]
    df.loc[idx[1], ["macd", "macd_signal"]] = [last_macd, last_sig]
    return df


def test_macd_bullish_cross_detected():
    # Was below signal, now above -> fresh bullish cross.
    snap = latest_snapshot(_precomputed_two_bar(-0.1, 0.0, 0.1, 0.0), precomputed=True)
    assert snap.macd_bullish_cross
    assert not snap.macd_bearish_cross
    assert snap.macd_above_signal


def test_macd_bearish_cross_detected():
    snap = latest_snapshot(_precomputed_two_bar(0.1, 0.0, -0.1, 0.0), precomputed=True)
    assert snap.macd_bearish_cross
    assert not snap.macd_bullish_cross
    assert not snap.macd_above_signal


def test_no_cross_when_relationship_unchanged():
    snap = latest_snapshot(_precomputed_two_bar(0.2, 0.0, 0.3, 0.0), precomputed=True)
    assert not snap.macd_bullish_cross
    assert not snap.macd_bearish_cross
    assert snap.macd_above_signal


def test_snapshot_with_nan_core_raises():
    df = _precomputed_two_bar(0.1, 0.0, 0.1, 0.0)
    df.iloc[-1, df.columns.get_loc("rsi")] = float("nan")
    with pytest.raises(IndicatorError, match="NaN"):
        latest_snapshot(df, precomputed=True)
