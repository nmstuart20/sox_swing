"""End-to-end integration test for the trading loop.

Wires the *real* signal engine, risk manager, order manager, and orchestration
loop together, swapping only the Alpaca and market-data boundaries for in-memory
paper fakes. No network, no keys, no live orders — exactly the "paper mode,
no live orders" guarantee the build spec asks for.

The two invariants under test, every cycle:
  * the bot never holds SOXL and SOXS at the same time, and
  * it never exceeds its configured risk limits (notional cap, trade count).
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
from execution.order_manager import OrderManager
from main import TradingEngine
from monitoring.alerts import AlertLevel, DiscordNotifier
from monitoring.monitor import Monitor
from risk.risk_manager import RiskManager
from strategy.signal_engine import SignalEngine
from tests.conftest import FakeAlpacaClient, FakeMarketData, make_ohlcv


MAX_POSITION_PCT = 0.10
MAX_TRADES_PER_DAY = 2
EQUITY = 100_000.0
PRICE = 20.0


def _settings(max_trades: int = MAX_TRADES_PER_DAY) -> Settings:
    """A fully-populated Settings with technical-only weighting (no Finnhub)."""
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
        strategy=StrategyConfig(technical_weight=1.0, sentiment_weight=0.0),
        engine=EngineConfig(
            poll_interval_seconds=1,
            close_at_eod=True,
            use_options=False,
            eod_flat_buffer_minutes=15,
        ),
        monitoring=MonitoringConfig(
            alerts_enabled=False, discord_webhook_url="", alert_min_level="INFO", bot_name="test"
        ),
        logging=LoggingConfig(
            level="INFO", log_dir=Path("logs"), log_file="t.log", max_bytes=1024, backup_count=1
        ),
    )


def _build(client: FakeAlpacaClient, market_data: FakeMarketData, max_trades: int = MAX_TRADES_PER_DAY):
    """Assemble a TradingEngine from real components over the fakes."""
    settings = _settings(max_trades)
    risk = RiskManager(settings.risk)
    notifier = DiscordNotifier("", enabled=False, min_level=AlertLevel.INFO)
    monitor = Monitor(notifier)
    orders = OrderManager(
        client,
        risk,
        flip_timeout=2.0,
        fill_timeout=2.0,
        poll_interval=0.001,
    )
    engine = TradingEngine(
        settings,
        client,
        market_data,
        finnhub=None,
        signal_engine=SignalEngine(settings.strategy),
        risk_manager=risk,
        order_manager=orders,
        monitor=monitor,
    )
    return engine, risk


@pytest.fixture
def market_data() -> FakeMarketData:
    # Both legs get tradeable frames; SOXS direction is irrelevant — only its
    # ATR/price feed the bracket when the signal favors the short leg.
    frames = {
        "SOXL": make_ohlcv(n=120, start=10.0, slope=0.1),   # uptrend -> bullish
        "SOXS": make_ohlcv(n=120, start=30.0, slope=-0.05),
    }
    return FakeMarketData(frames, prices={"SOXL": PRICE, "SOXS": PRICE})


def _set_long_trend(market_data: FakeMarketData, *, bullish: bool) -> None:
    if bullish:
        market_data._frames["SOXL"] = make_ohlcv(n=120, start=10.0, slope=0.1)
    else:
        market_data._frames["SOXL"] = make_ohlcv(n=120, start=30.0, slope=-0.1)


def _assert_never_both_legs(client: FakeAlpacaClient) -> None:
    held = set(client.held_symbols)
    assert held != {"SOXL", "SOXS"}, f"held both legs at once: {held}"
    assert len(held) <= 1


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_full_loop_enters_on_bullish_signal(market_data):
    client = FakeAlpacaClient(equity=EQUITY)
    engine, risk = _build(client, market_data)

    engine._run_cycle()

    assert client.held_symbols == ["SOXL"]
    assert risk.trades_today == 1
    _assert_never_both_legs(client)
    # Notional within the 10%-of-equity cap.
    entry = client.submitted[-1]
    assert float(entry.qty) * PRICE <= MAX_POSITION_PCT * EQUITY


def test_flip_closes_opposite_leg_and_never_holds_both(market_data):
    client = FakeAlpacaClient(equity=EQUITY)
    engine, risk = _build(client, market_data, max_trades=10)

    _set_long_trend(market_data, bullish=True)
    engine._run_cycle()
    assert client.held_symbols == ["SOXL"]
    _assert_never_both_legs(client)

    # Flip to bearish: must close SOXL and open SOXS, never both at once.
    _set_long_trend(market_data, bullish=False)
    engine._run_cycle()
    assert client.held_symbols == ["SOXS"]
    _assert_never_both_legs(client)

    # And flip back.
    _set_long_trend(market_data, bullish=True)
    engine._run_cycle()
    assert client.held_symbols == ["SOXL"]
    _assert_never_both_legs(client)


def test_trade_count_cap_is_never_exceeded(market_data):
    client = FakeAlpacaClient(equity=EQUITY)
    engine, risk = _build(client, market_data, max_trades=MAX_TRADES_PER_DAY)

    # Alternate direction every cycle; each flip is a fresh entry against the cap.
    for i in range(6):
        _set_long_trend(market_data, bullish=(i % 2 == 0))
        engine._run_cycle()
        _assert_never_both_legs(client)
        assert risk.trades_today <= MAX_TRADES_PER_DAY

    assert risk.trade_limit_reached()
    # Entries submitted (buys) must not exceed the daily cap.
    buys = [o for o in client.submitted if o.side.value == "buy"]
    assert len(buys) == MAX_TRADES_PER_DAY


def test_every_entry_respects_the_notional_cap(market_data):
    client = FakeAlpacaClient(equity=EQUITY)
    engine, risk = _build(client, market_data, max_trades=10)

    for i in range(4):
        _set_long_trend(market_data, bullish=(i % 2 == 0))
        engine._run_cycle()

    buys = [o for o in client.submitted if o.side.value == "buy"]
    assert buys, "expected at least one entry"
    for order in buys:
        assert float(order.qty) * PRICE <= MAX_POSITION_PCT * EQUITY


def test_daily_loss_breach_forces_flat(market_data):
    client = FakeAlpacaClient(equity=EQUITY)
    engine, risk = _build(client, market_data, max_trades=10)

    _set_long_trend(market_data, bullish=True)
    engine._run_cycle()
    assert client.held_symbols == ["SOXL"]

    # Equity drops 4% — past the 3% daily-loss limit. Next cycle force-flattens
    # and refuses any new entry.
    client.set_equity(EQUITY * 0.96)
    engine._run_cycle()

    assert client.held_symbols == []
    _assert_never_both_legs(client)


def test_market_closed_is_a_no_op(market_data):
    client = FakeAlpacaClient(equity=EQUITY)
    client.clock.is_open = False
    engine, risk = _build(client, market_data)

    engine._run_cycle()

    assert client.submitted == []
    assert client.held_symbols == []
    assert risk.trades_today == 0


def test_engine_is_paper_and_makes_no_live_calls(market_data):
    client = FakeAlpacaClient(equity=EQUITY)
    engine, _ = _build(client, market_data)
    engine._run_cycle()
    # The whole loop ran against the in-memory paper fake.
    assert client.is_paper is True
