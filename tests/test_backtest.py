"""Offline tests for the backtesting harness.

Everything runs in-memory: synthetic OHLCV from :func:`tests.conftest.make_ohlcv`
drives the *real* signal engine, risk manager, and order manager over the
simulated broker — no network, no keys, no live orders. The end-to-end test
asserts the same invariants the live integration test does (never both legs,
notional cap respected) plus that the reported metrics and trade log are
coherent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from backtest.broker import CostModel, SimBroker, SimClock
from backtest.engine import build_backtest_engine
from backtest.feeds import BacktestFinnhub
from backtest.metrics import compute_metrics
from config.settings import (
    AlpacaConfig,
    EngineConfig,
    FinnhubConfig,
    LoggingConfig,
    MonitoringConfig,
    RiskConfig,
    Settings,
    StrategyConfig,
)
from data.finnhub_data import NEWS_COLUMNS
from tests.conftest import make_ohlcv

INITIAL = 100_000.0
MAX_POSITION_PCT = 0.10


def _settings(
    *, technical_weight: float = 1.0, sentiment_weight: float = 0.0, max_trades: int = 20
) -> Settings:
    return Settings(
        symbol_long="SOXL",
        symbol_short="SOXS",
        alpaca=AlpacaConfig(api_key="k", secret_key="s", paper=True),
        finnhub=FinnhubConfig(api_key="f"),
        risk=RiskConfig(
            max_position_pct=MAX_POSITION_PCT,
            max_daily_loss_pct=0.03,
            max_trades_per_day=max_trades,
            atr_stop_multiplier=2.0,
            atr_take_profit_multiplier=3.0,
        ),
        strategy=StrategyConfig(
            technical_weight=technical_weight, sentiment_weight=sentiment_weight
        ),
        engine=EngineConfig(
            poll_interval_seconds=60,
            close_at_eod=True,
            use_options=False,
            eod_flat_buffer_minutes=15,
        ),
        monitoring=MonitoringConfig(
            alerts_enabled=False, discord_webhook_url="", alert_min_level="INFO", bot_name="t"
        ),
        logging=LoggingConfig(
            level="INFO", log_dir=Path("logs"), log_file="t.log", max_bytes=1024, backup_count=1
        ),
    )


def _session_bars(slope_long: float, slope_short: float, *, n: int = 120) -> dict[str, pd.DataFrame]:
    """A pair of frames whose final bar sits at the session close (20:00 UTC)."""
    end = pd.Timestamp("2026-06-04 20:00", tz="UTC")
    return {
        "SOXL": make_ohlcv(n=n, start=15.0, slope=slope_long, end=end),
        "SOXS": make_ohlcv(n=n, start=30.0, slope=slope_short, end=end),
    }


# ----------------------------------------------------------------------
# Broker unit tests
# ----------------------------------------------------------------------
def _clock() -> SimClock:
    now = datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc)
    return SimClock(is_open=True, next_open=now, next_close=now)


def test_broker_applies_slippage_on_buy_and_sell():
    broker = SimBroker(initial_capital=INITIAL, cost_model=CostModel(slippage_bps=10.0))
    ts = datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc)
    broker.set_marks({"SOXL": 20.0})
    broker.process_bar(ts, {"SOXL": {"open": 20, "high": 20, "low": 20, "close": 20}}, 0, _clock())

    broker.submit_stop_entry_order("SOXL", 10, side="buy", stop_loss_price=18.0)
    pos = broker.get_position("SOXL")
    # Buy fills above the mark by 10 bps.
    assert pos.entry_price == pytest.approx(20.0 * 1.001)

    order = broker.close_position("SOXL", reason="manual")
    # Sell fills below the mark by 10 bps.
    assert float(order.filled_avg_price) == pytest.approx(20.0 * 0.999)
    assert broker.trades[-1].exit_reason == "manual"


def test_broker_triggers_stop_on_breach():
    broker = SimBroker(initial_capital=INITIAL, cost_model=CostModel(slippage_bps=0.0))
    ts = datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc)
    broker.set_marks({"SOXL": 20.0})
    broker.process_bar(ts, {"SOXL": {"open": 20, "high": 20, "low": 20, "close": 20}}, 0, _clock())
    broker.submit_stop_entry_order("SOXL", 10, side="buy", stop_loss_price=19.0)

    # Next bar dips through the stop: low 18.5 < stop 19.0 -> stopped out at 19.0.
    ts2 = datetime(2026, 6, 4, 15, 5, tzinfo=timezone.utc)
    broker.set_marks({"SOXL": 18.8})
    broker.process_bar(ts2, {"SOXL": {"open": 19.5, "high": 19.5, "low": 18.5, "close": 18.8}}, 1, _clock())

    assert not broker.has_position("SOXL")
    trade = broker.trades[-1]
    assert trade.exit_reason == "stop-loss"
    assert trade.exit_price == pytest.approx(19.0)
    assert trade.net_pnl < 0
    assert trade.bars_held == 1


def test_broker_gap_through_stop_fills_at_open():
    broker = SimBroker(initial_capital=INITIAL, cost_model=CostModel(slippage_bps=0.0))
    ts = datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc)
    broker.set_marks({"SOXL": 20.0})
    broker.process_bar(ts, {"SOXL": {"open": 20, "high": 20, "low": 20, "close": 20}}, 0, _clock())
    broker.submit_stop_entry_order("SOXL", 10, side="buy", stop_loss_price=19.0)

    # Gap down: bar opens at 18.0, already below the 19.0 stop -> fill at 18.0.
    ts2 = datetime(2026, 6, 4, 15, 5, tzinfo=timezone.utc)
    broker.set_marks({"SOXL": 18.0})
    broker.process_bar(ts2, {"SOXL": {"open": 18.0, "high": 18.2, "low": 17.5, "close": 18.0}}, 1, _clock())

    assert broker.trades[-1].exit_price == pytest.approx(18.0)


# ----------------------------------------------------------------------
# News replay (no look-ahead)
# ----------------------------------------------------------------------
def _news_frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"timestamp": ts, "symbol": "SOXL", "headline": h, "source": "", "url": str(i),
          "summary": "", "sentiment": None} for i, (ts, h) in enumerate(rows)],
        columns=NEWS_COLUMNS,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_news_feed_is_point_in_time():
    news = _news_frame([
        ("2026-06-01 12:00", "chip demand surges, record growth and upgrade"),  # bullish
        ("2026-06-03 12:00", "sector plunge on glut fears, downgrade and selloff"),  # bearish
    ])
    feed = BacktestFinnhub({"SOXL": news}, sector_symbols=())

    # As of 06-02, only the bullish article is visible.
    feed.set_time(pd.Timestamp("2026-06-02 20:00", tz="UTC"))
    early = feed.get_news_sentiment("SOXL", lookback_days=7)
    assert early.article_count == 1 and early.score > 0

    # As of 06-03, both are visible and the bearish one drags the mean down.
    feed.set_time(pd.Timestamp("2026-06-03 20:00", tz="UTC"))
    late = feed.get_news_sentiment("SOXL", lookback_days=7)
    assert late.article_count == 2 and late.score < early.score


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def test_metrics_total_return_and_drawdown():
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    curve = [
        (t0, 100.0),
        (t0.replace(day=2), 120.0),
        (t0.replace(day=3), 90.0),   # peak-to-trough from 120 -> 90 == -25%
        (t0.replace(day=4), 110.0),
    ]
    m = compute_metrics(curve, trades=[], total_commission=0.0)
    assert m.total_return == pytest.approx(0.10)
    assert m.max_drawdown == pytest.approx(-0.25)
    assert m.num_trades == 0 and m.win_rate == 0.0


# ----------------------------------------------------------------------
# End-to-end engine
# ----------------------------------------------------------------------
def test_backtest_runs_and_respects_invariants():
    bars = _session_bars(slope_long=0.05, slope_short=-0.02)
    engine = build_backtest_engine(
        _settings(), bars, initial_capital=INITIAL, cost_model=CostModel(slippage_bps=2.0)
    )
    result = engine.run()

    # It traded, and only ever held one leg at a time.
    assert result.metrics.num_trades > 0
    assert engine._broker.max_concurrent <= 1
    assert {t.symbol for t in result.trades} <= {"SOXL", "SOXS"}

    # Every entry honored the notional cap (cap grows with equity; final equity
    # bounds it from above in an uptrend). Slippage adds a hair to the fill.
    cap = MAX_POSITION_PCT * result.metrics.final_equity * 1.02
    for t in result.trades:
        assert t.qty * t.entry_price <= cap

    # The trade log and equity curve are coherent with the metrics.
    assert len(result.trades) == result.metrics.num_trades
    assert len(result.equity_curve) == len(bars["SOXL"])
    assert result.metrics.initial_equity == pytest.approx(INITIAL)


def test_backtest_runs_with_news_sentiment():
    # Sentiment-weighted config plus a bullish news stream over the window.
    settings = _settings(technical_weight=0.7, sentiment_weight=0.3)
    bars = _session_bars(slope_long=0.05, slope_short=-0.02)
    news = {"SOXL": _news_frame([
        ("2026-06-04 14:00", "demand surges, strong growth, upgrade and record profit"),
    ])}
    engine = build_backtest_engine(
        settings, bars, initial_capital=INITIAL, news=news, cost_model=CostModel(slippage_bps=2.0)
    )
    result = engine.run()
    assert result.metrics.num_trades > 0
    assert engine._broker.max_concurrent <= 1


def test_backtest_needs_enough_overlapping_bars():
    bars = _session_bars(slope_long=0.05, slope_short=-0.02, n=40)  # < min_bars
    with pytest.raises(ValueError, match="overlapping bars"):
        build_backtest_engine(_settings(), bars, initial_capital=INITIAL)
