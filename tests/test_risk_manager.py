"""Unit tests for the risk manager — sizing, brackets, and the safety rules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from risk.risk_manager import RiskManager, RiskParams
from strategy.signal_engine import Direction
from tests.conftest import FakePosition, make_signal


@pytest.fixture
def rm(risk_config) -> RiskManager:
    return RiskManager(risk_config)


def _pos(qty: float, unrealized_pl: float = 0.0) -> FakePosition:
    return FakePosition("X", str(qty), str(unrealized_pl))


# ----------------------------------------------------------------------
# Position sizing
# ----------------------------------------------------------------------
def test_position_size_capped_by_notional(risk_config):
    rm = RiskManager(risk_config, params=RiskParams(risk_per_trade_pct=1.0))
    # Risk budget is huge (100% of equity), so the 10% notional cap binds.
    qty = rm.position_size(equity=100_000, entry_price=20.0, stop_loss=19.0)
    # 10% of 100k = 10k notional / 20 = 500 shares.
    assert qty == 500


def test_position_size_capped_by_risk_budget(risk_config):
    rm = RiskManager(risk_config, params=RiskParams(risk_per_trade_pct=0.01))
    # Tight 1% risk budget with a $1 stop distance binds before the notional cap.
    qty = rm.position_size(equity=100_000, entry_price=20.0, stop_loss=19.0)
    # 1% of 100k = $1000 risk / $1 stop = 1000 shares... but notional cap = 500.
    # Risk budget here (1000) exceeds notional (500), so notional wins -> 500.
    assert qty == 500
    # Widen the stop so the risk budget becomes the binding constraint.
    qty2 = rm.position_size(equity=100_000, entry_price=20.0, stop_loss=15.0)
    # $1000 / $5 = 200 shares < notional cap 500.
    assert qty2 == 200


def test_position_size_notional_only_when_risk_sizing_disabled(risk_config):
    from dataclasses import replace

    # risk_based_sizing off + full notional cap => deploy 100% of equity,
    # ignoring the (otherwise binding) per-trade risk budget.
    config = replace(risk_config, max_position_pct=1.0, risk_based_sizing=False)
    rm = RiskManager(config, params=RiskParams(risk_per_trade_pct=0.01))
    # A wide stop would normally bind the risk budget hard ($1000 / $5 = 200),
    # but with risk sizing off only the notional cap applies: 100k / 20 = 5000.
    qty = rm.position_size(equity=100_000, entry_price=20.0, stop_loss=15.0)
    assert qty == 5000


def test_position_size_rounds_down_to_whole_shares(rm):
    qty = rm.position_size(equity=10_000, entry_price=30.0, stop_loss=29.0)
    assert isinstance(qty, int)
    assert qty == int(qty)


def test_position_size_zero_on_bad_inputs(rm):
    assert rm.position_size(equity=0, entry_price=20.0, stop_loss=19.0) == 0
    assert rm.position_size(equity=100_000, entry_price=0, stop_loss=0) == 0


# ----------------------------------------------------------------------
# Stop / take-profit brackets
# ----------------------------------------------------------------------
def test_stop_below_and_take_profit_above_entry(rm):
    stop, take = rm.stop_and_take_profit(Direction.BULLISH, entry_price=20.0, atr=0.5)
    # stop = entry - 2*ATR, take = entry + 3*ATR.
    assert stop == pytest.approx(19.0)
    assert take == pytest.approx(21.5)
    assert stop < 20.0 < take


def test_atr_is_floored_to_avoid_hairline_stop(rm):
    # A near-zero ATR is floored to min_atr_pct (0.1%) of price.
    stop, _ = rm.stop_and_take_profit(Direction.BULLISH, entry_price=20.0, atr=0.0)
    floored_atr = 0.001 * 20.0
    assert stop == pytest.approx(20.0 - 2.0 * floored_atr)


def test_stop_never_negative(rm):
    stop, _ = rm.stop_and_take_profit(Direction.BULLISH, entry_price=1.0, atr=100.0)
    assert stop >= 0.01


# ----------------------------------------------------------------------
# evaluate() — the gate
# ----------------------------------------------------------------------
def _ts(rm) -> None:
    rm.reset_daily(100_000)


def test_approves_clean_entry(rm):
    rm.reset_daily(100_000)
    sig = make_signal(Direction.BULLISH, target_symbol="SOXL")
    d = rm.evaluate(sig, equity=100_000, entry_price=20.0, atr=0.5, positions={})
    assert d.approved
    assert d.target_symbol == "SOXL"
    assert d.qty > 0
    assert d.notional <= 0.10 * 100_000 + 20.0  # within the cap (+ one share slack)
    assert d.close_symbols == ()


def test_neutral_signal_vetoed(rm):
    rm.reset_daily(100_000)
    sig = make_signal(Direction.NEUTRAL, target_symbol=None)
    d = rm.evaluate(sig, equity=100_000, entry_price=20.0, atr=0.5)
    assert not d.approved
    assert "neutral" in d.reason


def test_unmanaged_target_vetoed(rm):
    rm.reset_daily(100_000)
    sig = make_signal(Direction.BULLISH, target_symbol="AAPL")
    d = rm.evaluate(sig, equity=100_000, entry_price=20.0, atr=0.5)
    assert not d.approved
    assert "not a managed symbol" in d.reason


def test_invalid_price_or_atr_vetoed(rm):
    rm.reset_daily(100_000)
    sig = make_signal(Direction.BULLISH, target_symbol="SOXL")
    assert not rm.evaluate(sig, 100_000, entry_price=0.0, atr=0.5).approved
    assert not rm.evaluate(sig, 100_000, entry_price=20.0, atr=0.0).approved


def test_already_holding_target_is_not_pyramided(rm):
    rm.reset_daily(100_000)
    sig = make_signal(Direction.BULLISH, target_symbol="SOXL")
    d = rm.evaluate(sig, 100_000, 20.0, 0.5, positions={"SOXL": _pos(100)})
    assert not d.approved
    assert "already holding" in d.reason


# ----------------------------------------------------------------------
# Daily loss limit
# ----------------------------------------------------------------------
def test_daily_loss_breach_vetoes_new_entry(rm):
    rm.reset_daily(100_000)
    sig = make_signal(Direction.BULLISH, target_symbol="SOXL")
    # 4% drawdown exceeds the 3% limit.
    d = rm.evaluate(sig, equity=96_000, entry_price=20.0, atr=0.5, positions={})
    assert not d.approved
    assert "daily loss limit" in d.reason


def test_drawdown_is_zero_when_flat_or_up(rm):
    rm.reset_daily(100_000)
    assert rm.daily_drawdown(100_000) == 0.0
    assert rm.daily_drawdown(110_000) == 0.0
    assert rm.daily_loss_breached(100_000) is False


# ----------------------------------------------------------------------
# Trade-count cap
# ----------------------------------------------------------------------
def test_trade_count_cap_vetoes(rm, risk_config):
    rm.reset_daily(100_000)
    for _ in range(risk_config.max_trades_per_day):
        rm.register_entry()
    assert rm.trade_limit_reached()
    sig = make_signal(Direction.BULLISH, target_symbol="SOXL")
    d = rm.evaluate(sig, 100_000, 20.0, 0.5, positions={})
    assert not d.approved
    assert "max trades/day" in d.reason


def test_daily_reset_clears_trade_count(rm):
    rm.reset_daily(100_000)
    rm.register_entry()
    rm.register_entry()
    assert rm.trades_today == 2
    rm.reset_daily(100_000)
    assert rm.trades_today == 0


def test_maybe_reset_rolls_over_on_new_day(rm):
    day1 = datetime(2026, 6, 4, 14, 0, tzinfo=timezone.utc)
    rm.maybe_reset_for_day(100_000, now=day1)
    rm.register_entry()
    assert rm.trades_today == 1
    # Same day -> no reset.
    assert rm.maybe_reset_for_day(100_000, now=day1 + timedelta(hours=1)) is False
    assert rm.trades_today == 1
    # Next calendar day -> reset.
    assert rm.maybe_reset_for_day(100_000, now=day1 + timedelta(days=1)) is True
    assert rm.trades_today == 0


# ----------------------------------------------------------------------
# The no-both-legs rule
# ----------------------------------------------------------------------
def test_entry_with_opposite_leg_open_requires_closing_it_first(rm):
    rm.reset_daily(100_000)
    sig = make_signal(Direction.BULLISH, target_symbol="SOXL")
    # Holding SOXS while a bullish SOXL entry comes in -> approve but flag the flip.
    d = rm.evaluate(sig, 100_000, 20.0, 0.5, positions={"SOXS": _pos(100)})
    assert d.approved
    assert d.target_symbol == "SOXL"
    assert d.close_symbols == ("SOXS",)


def test_forced_exit_flattens_both_legs_on_invariant_breach(rm):
    rm.reset_daily(100_000)
    exits = rm.forced_exits(
        100_000, positions={"SOXL": _pos(50), "SOXS": _pos(50)}
    )
    flattened = {e.symbol for e in exits}
    assert flattened == {"SOXL", "SOXS"}
    assert all("invariant" in e.reason for e in exits)


def test_forced_exit_flattens_held_leg_on_daily_loss(rm):
    rm.reset_daily(100_000)
    exits = rm.forced_exits(96_000, positions={"SOXL": _pos(50), "SOXS": None})
    assert [e.symbol for e in exits] == ["SOXL"]
    assert "daily loss" in exits[0].reason


def test_no_forced_exit_when_single_leg_and_within_limits(rm):
    rm.reset_daily(100_000)
    exits = rm.forced_exits(100_000, positions={"SOXL": _pos(50), "SOXS": None})
    assert exits == []


def test_zero_qty_position_is_treated_as_flat(rm):
    rm.reset_daily(100_000)
    exits = rm.forced_exits(100_000, positions={"SOXL": _pos(0), "SOXS": _pos(0)})
    assert exits == []


# ----------------------------------------------------------------------
# Params validation
# ----------------------------------------------------------------------
def test_risk_params_validation():
    with pytest.raises(ValueError):
        RiskParams(risk_per_trade_pct=0.0)
    with pytest.raises(ValueError):
        RiskParams(risk_per_trade_pct=1.5)
    with pytest.raises(ValueError):
        RiskParams(min_qty=0)
