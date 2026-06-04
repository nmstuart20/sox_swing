"""Unit tests for the combined signal engine."""

from __future__ import annotations

import pytest

from config.settings import StrategyConfig
from strategy.signal_engine import Direction, SignalEngine, SignalParams
from tests.conftest import (
    FakeFinnhub,
    bearish_snapshot,
    bullish_snapshot,
    make_snapshot,
    neutral_snapshot,
)


@pytest.fixture
def engine(strategy_config) -> SignalEngine:
    return SignalEngine(strategy_config)


# ----------------------------------------------------------------------
# Direction & target selection
# ----------------------------------------------------------------------
def test_bullish_snapshot_goes_long_soxl(engine):
    sig = engine.generate(bullish_snapshot(), sentiment_score=0.0)
    assert sig.direction is Direction.BULLISH
    assert sig.target_symbol == "SOXL"
    assert sig.is_actionable
    assert sig.combined_score > 0


def test_bearish_snapshot_goes_long_soxs(engine):
    sig = engine.generate(bearish_snapshot(), sentiment_score=0.0)
    assert sig.direction is Direction.BEARISH
    assert sig.target_symbol == "SOXS"
    assert sig.combined_score < 0


def test_flat_snapshot_is_neutral(engine):
    sig = engine.generate(neutral_snapshot(), sentiment_score=0.0)
    assert sig.direction is Direction.NEUTRAL
    assert sig.target_symbol is None
    assert not sig.is_actionable


def test_target_is_always_single_leg_or_none(engine):
    """The engine never names both legs — the no-both-legs rule starts here."""
    for snap, sent in [
        (bullish_snapshot(), 0.5),
        (bearish_snapshot(), -0.5),
        (make_snapshot(), 0.0),
    ]:
        sig = engine.generate(snap, sentiment_score=sent)
        assert sig.target_symbol in {"SOXL", "SOXS", None}


# ----------------------------------------------------------------------
# Threshold behavior
# ----------------------------------------------------------------------
def test_subthreshold_score_stays_neutral():
    # Pure-technical engine; a single weak flag falls under the entry threshold.
    cfg = StrategyConfig(technical_weight=1.0, sentiment_weight=0.0)
    eng = SignalEngine(cfg, params=SignalParams(entry_threshold=0.5))
    # One light flag active out of the full weighted set -> small score.
    snap = make_snapshot(price_above_vwap=True)
    sig = eng.generate(snap)
    assert sig.direction is Direction.NEUTRAL
    assert abs(sig.combined_score) < 0.5


def test_threshold_boundary_is_inclusive():
    cfg = StrategyConfig(technical_weight=0.0, sentiment_weight=1.0)
    eng = SignalEngine(cfg, params=SignalParams(entry_threshold=0.15))
    # Sentiment exactly at the threshold should trigger (>=).
    sig = eng.generate(make_snapshot(), sentiment_score=0.15)
    assert sig.direction is Direction.BULLISH


# ----------------------------------------------------------------------
# Technical / sentiment blending
# ----------------------------------------------------------------------
def test_strong_bearish_sentiment_overrides_weak_bullish_tech():
    cfg = StrategyConfig(technical_weight=0.3, sentiment_weight=0.7)
    eng = SignalEngine(cfg)
    # Mildly bullish technicals, strongly bearish news -> net bearish.
    snap = make_snapshot(price_above_vwap=True, price_above_ema_fast=True)
    sig = eng.generate(snap, sentiment_score=-1.0)
    assert sig.direction is Direction.BEARISH
    assert sig.target_symbol == "SOXS"


def test_weights_are_reported_on_the_signal(engine):
    sig = engine.generate(bullish_snapshot(), sentiment_score=0.2)
    assert sig.technical_weight == pytest.approx(0.7)
    assert sig.sentiment_weight == pytest.approx(0.3)


def test_confidence_is_abs_combined_score(engine):
    sig = engine.generate(bearish_snapshot(), sentiment_score=-0.5)
    assert sig.confidence == pytest.approx(abs(sig.combined_score))
    assert 0.0 <= sig.confidence <= 1.0


def test_scores_are_clamped_to_unit_range():
    cfg = StrategyConfig(technical_weight=0.5, sentiment_weight=0.5)
    eng = SignalEngine(cfg)
    sig = eng.generate(bullish_snapshot(), sentiment_score=5.0)  # out of range
    assert sig.sentiment_score == 1.0
    assert -1.0 <= sig.combined_score <= 1.0


def test_zero_sentiment_weight_ignores_news():
    cfg = StrategyConfig(technical_weight=1.0, sentiment_weight=0.0)
    eng = SignalEngine(cfg)
    bull = eng.generate(bullish_snapshot(), sentiment_score=-1.0)
    # News is bearish but carries zero weight, so technicals win.
    assert bull.direction is Direction.BULLISH


# ----------------------------------------------------------------------
# evaluate() — sentiment plumbing
# ----------------------------------------------------------------------
def test_evaluate_without_finnhub_is_technical_only(engine):
    sig = engine.evaluate(bullish_snapshot(), finnhub=None)
    assert sig.sentiment_score == 0.0
    assert sig.direction is Direction.BULLISH


def test_evaluate_folds_in_finnhub_sentiment(engine):
    sig = engine.evaluate(neutral_snapshot(), finnhub=FakeFinnhub(score=1.0))
    # Flat technicals + strongly bullish sentiment (weight 0.3) clears threshold.
    assert sig.direction is Direction.BULLISH
    assert sig.sentiment_score == pytest.approx(1.0)


def test_evaluate_swallows_finnhub_errors(engine):
    class Boom:
        sector_symbols = ()

        def get_news_sentiment(self, *a, **k):
            raise RuntimeError("finnhub down")

    sig = engine.evaluate(bullish_snapshot(), finnhub=Boom())
    # Sentiment failed but the cycle still produces a (technical-only) signal.
    assert sig.sentiment_score == 0.0
    assert sig.direction is Direction.BULLISH


# ----------------------------------------------------------------------
# Continuous vs. ternary scoring
# ----------------------------------------------------------------------
def _technical_engine(**param_overrides) -> SignalEngine:
    """A pure-technical engine (sentiment weight 0) for scoring tests."""
    cfg = StrategyConfig(technical_weight=1.0, sentiment_weight=0.0)
    return SignalEngine(cfg, params=SignalParams(**param_overrides))


def test_default_params_use_continuous_scoring():
    assert SignalParams().continuous_scoring is True


def test_continuous_score_tracks_price_without_crossing_a_flag():
    """The core fix: the score glides as price moves, not only at crossings.

    Both snapshots stay above every EMA/VWAP (identical boolean flags), so the
    legacy ternary scorer would give them the *same* score. With continuous
    scoring the one sitting further above scores more bullish.
    """
    eng = _technical_engine()  # continuous by default
    far = make_snapshot(
        close=21.0, atr=0.5,
        price_above_ema_slow=True, price_above_ema_mid=True,
        price_above_ema_fast=True, price_above_vwap=True,
    )
    near = make_snapshot(
        close=20.2, atr=0.5,
        price_above_ema_slow=True, price_above_ema_mid=True,
        price_above_ema_fast=True, price_above_vwap=True,
    )
    score_far = eng.generate(far).technical_score
    score_near = eng.generate(near).technical_score
    assert score_far > score_near > 0


def test_binary_scoring_is_opt_in_and_stepwise():
    """With continuous_scoring=False the same two prices score identically."""
    eng = _technical_engine(continuous_scoring=False)
    far = make_snapshot(
        close=21.0, price_above_ema_slow=True,
        price_above_ema_mid=True, price_above_vwap=True,
    )
    near = make_snapshot(
        close=20.2, price_above_ema_slow=True,
        price_above_ema_mid=True, price_above_vwap=True,
    )
    assert eng.generate(far).technical_score == eng.generate(near).technical_score


def test_continuous_rsi_leans_before_reaching_the_extremes():
    """RSI between 30 and 70 contributes nothing in ternary mode, a lean here."""
    eng = _technical_engine()
    mild_bull = eng.generate(make_snapshot(rsi=40.0)).technical_score
    mild_bear = eng.generate(make_snapshot(rsi=60.0)).technical_score
    assert mild_bull > 0 > mild_bear

    # In ternary mode RSI 40 and 60 are both inside 30..70, so neither votes and
    # the score is identical — the very blind spot the continuous lean removes.
    binary = _technical_engine(continuous_scoring=False)
    assert (
        binary.generate(make_snapshot(rsi=40.0)).technical_score
        == binary.generate(make_snapshot(rsi=60.0)).technical_score
    )


# ----------------------------------------------------------------------
# Params validation
# ----------------------------------------------------------------------
def test_invalid_entry_threshold_rejected():
    with pytest.raises(ValueError):
        SignalParams(entry_threshold=1.0)
    with pytest.raises(ValueError):
        SignalParams(entry_threshold=-0.1)


def test_invalid_scale_rejected():
    with pytest.raises(ValueError):
        SignalParams(atr_scale=0.0)
    with pytest.raises(ValueError):
        SignalParams(macd_scale=-1.0)
