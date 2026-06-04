"""Combined signal engine for the SOXL/SOXS pair.

Folds the technical :class:`~strategy.indicators.IndicatorSnapshot` and Finnhub
news sentiment into a single, structured trade decision. SOXL and SOXS are
*inverse* leveraged ETFs on the same semiconductor index, so the engine takes
one view of the underlying sector and expresses it as a single position:

  * **bullish** on semis  -> go long SOXL  (``Direction.BULLISH``),
  * **bearish** on semis  -> go long SOXS  (``Direction.BEARISH``),
  * **no edge**           -> stay flat     (``Direction.NEUTRAL``).

It never recommends holding both legs at once: a decision is a single target
symbol (or none). The split between technical and sentiment influence is
configurable via :class:`~config.settings.StrategyConfig` (the two weights sum
to 1.0); the entry threshold and per-flag technical weights live in
:class:`SignalParams`.

This module is pure decision logic. It consumes a snapshot and a sentiment
score and emits a :class:`TradeSignal`; it does not fetch bars, place orders,
or enforce risk limits. A convenience helper (:meth:`SignalEngine.sentiment_for_symbols`)
can pull and blend sentiment from :class:`~data.finnhub_data.FinnhubData`, but the
core :meth:`SignalEngine.generate` accepts an already-computed score so it stays
trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config.logging_setup import get_logger
from config.settings import StrategyConfig
from data.finnhub_data import FinnhubData
from strategy.indicators import IndicatorSnapshot

logger = get_logger(__name__)


class Direction(Enum):
    """Which way the combined signal leans on the semiconductor sector."""

    BULLISH = "bullish"  # long SOXL
    BEARISH = "bearish"  # long SOXS
    NEUTRAL = "neutral"  # stay flat


@dataclass(frozen=True)
class SignalParams:
    """Tunable thresholds and per-flag technical weights.

    The technical score is a weighted average of the snapshot's boolean flags,
    each contributing ``+w`` (bullish), ``-w`` (bearish), or ``0`` (neutral),
    normalized to ``[-1, 1]`` by the total weight in play. Defaults emphasize
    trend and momentum over mean-reversion, which suits trending leveraged ETFs.
    """

    # Minimum |combined score| required to take a side; below this -> NEUTRAL.
    entry_threshold: float = 0.15

    # Per-signal technical weights (relative; only their ratios matter).
    w_ema_stack: float = 1.0          # fast > mid > slow (or inverse)
    w_price_vs_ema_slow: float = 1.0  # price above/below the slow EMA (trend filter)
    w_price_vs_ema_mid: float = 0.5
    w_macd_cross: float = 1.0         # fresh MACD cross of its signal line
    w_macd_above: float = 0.5         # MACD above/below signal (state)
    w_rsi_extreme: float = 0.5        # oversold (bullish) / overbought (bearish)
    w_bollinger: float = 0.5          # close beyond a band (mean-reversion)
    w_vwap: float = 0.5               # price above/below rolling VWAP

    def __post_init__(self) -> None:
        if not 0.0 <= self.entry_threshold < 1.0:
            raise ValueError("entry_threshold must be in [0, 1).")


@dataclass(frozen=True)
class TradeSignal:
    """A structured trade decision: direction, confidence, and the reasons.

    ``confidence`` is ``abs(combined_score)`` clamped to ``[0, 1]`` — how
    strongly the blended technical/sentiment view leans. For ``NEUTRAL`` it
    still reports the (sub-threshold) magnitude so callers can see how close a
    setup came to triggering.
    """

    timestamp: Any  # pd.Timestamp from the snapshot
    direction: Direction
    target_symbol: str | None  # symbol to be long, or None when NEUTRAL
    confidence: float
    combined_score: float       # [-1, 1], sign = direction
    technical_score: float      # [-1, 1]
    sentiment_score: float      # [-1, 1]
    technical_weight: float
    sentiment_weight: float
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_actionable(self) -> bool:
        """True when the signal recommends taking (or holding) a position."""
        return self.direction is not Direction.NEUTRAL

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view for logging / serialization."""
        return {
            "timestamp": str(self.timestamp),
            "direction": self.direction.value,
            "target_symbol": self.target_symbol,
            "confidence": round(self.confidence, 4),
            "combined_score": round(self.combined_score, 4),
            "technical_score": round(self.technical_score, 4),
            "sentiment_score": round(self.sentiment_score, 4),
            "technical_weight": self.technical_weight,
            "sentiment_weight": self.sentiment_weight,
            "reasons": list(self.reasons),
        }

    def __str__(self) -> str:
        target = self.target_symbol or "—"
        return (
            f"{self.direction.value.upper()} {target} "
            f"(conf={self.confidence:.2f}, tech={self.technical_score:+.2f}, "
            f"sent={self.sentiment_score:+.2f}, combined={self.combined_score:+.2f})"
        )


class SignalEngine:
    """Blends technical and sentiment signals into a :class:`TradeSignal`.

    Args:
        config: weighting between technicals and sentiment (sums to 1.0).
        symbol_long: the bullish-leg symbol (default SOXL).
        symbol_short: the bearish-leg symbol (default SOXS).
        params: technical-scoring weights and the entry threshold.
    """

    def __init__(
        self,
        config: StrategyConfig,
        symbol_long: str = "SOXL",
        symbol_short: str = "SOXS",
        params: SignalParams | None = None,
    ) -> None:
        self._config = config
        self._symbol_long = symbol_long
        self._symbol_short = symbol_short
        self._params = params or SignalParams()
        logger.info(
            "SignalEngine initialized (long=%s, short=%s, tech_w=%.2f, sent_w=%.2f, entry=%.2f)",
            symbol_long,
            symbol_short,
            config.technical_weight,
            config.sentiment_weight,
            self._params.entry_threshold,
        )

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------
    def generate(
        self,
        snapshot: IndicatorSnapshot,
        sentiment_score: float = 0.0,
        *,
        sentiment_meta: str | None = None,
    ) -> TradeSignal:
        """Combine a technical ``snapshot`` and a ``sentiment_score`` into a signal.

        Args:
            snapshot: latest-bar technicals from :func:`strategy.indicators.latest_snapshot`.
            sentiment_score: semis bullishness in ``[-1, 1]`` (positive = bullish on
                the sector, i.e. favoring SOXL). Defaults to neutral so the engine
                degrades to a purely technical decision if news is unavailable.
            sentiment_meta: optional note (e.g. "finnhub", "keyword/12 articles")
                folded into the reasons for traceability.

        Returns:
            A :class:`TradeSignal` whose ``direction`` is ``NEUTRAL`` when the
            blended score is within ``entry_threshold`` of zero.
        """
        sentiment_score = _clamp(sentiment_score, -1.0, 1.0)
        technical_score, tech_reasons = self._score_technical(snapshot)

        tw = self._config.technical_weight
        sw = self._config.sentiment_weight
        combined = _clamp(tw * technical_score + sw * sentiment_score, -1.0, 1.0)

        reasons = list(tech_reasons)
        if sw > 0:
            note = f" ({sentiment_meta})" if sentiment_meta else ""
            reasons.append(_lean_phrase("sentiment", sentiment_score) + note)

        threshold = self._params.entry_threshold
        if combined >= threshold:
            direction = Direction.BULLISH
            target = self._symbol_long
        elif combined <= -threshold:
            direction = Direction.BEARISH
            target = self._symbol_short
        else:
            direction = Direction.NEUTRAL
            target = None
            reasons.append(
                f"combined score {combined:+.2f} within ±{threshold:.2f} threshold — no edge"
            )

        signal = TradeSignal(
            timestamp=snapshot.timestamp,
            direction=direction,
            target_symbol=target,
            confidence=abs(combined),
            combined_score=combined,
            technical_score=technical_score,
            sentiment_score=sentiment_score,
            technical_weight=tw,
            sentiment_weight=sw,
            reasons=tuple(reasons),
        )
        logger.info("Signal: %s", signal)
        return signal

    # ------------------------------------------------------------------
    # Technical scoring
    # ------------------------------------------------------------------
    def _score_technical(
        self, snap: IndicatorSnapshot
    ) -> tuple[float, list[str]]:
        """Weighted-average the snapshot's flags into a score in ``[-1, 1]``.

        Each component votes ``+1``/``-1``/``0``; the score is the weight-
        weighted mean of the votes, so a flat tape (all neutral) scores 0 and a
        unanimous bullish read scores +1. Returns the score plus human-readable
        reasons for every non-neutral contributor.
        """
        p = self._params
        # (name, vote in {-1, 0, +1}, weight)
        components: list[tuple[str, int, float]] = [
            ("EMA stack", _stack_vote(snap), p.w_ema_stack),
            ("price vs slow EMA", _bool_vote(snap.price_above_ema_slow), p.w_price_vs_ema_slow),
            ("price vs mid EMA", _bool_vote(snap.price_above_ema_mid), p.w_price_vs_ema_mid),
            ("MACD cross", _macd_cross_vote(snap), p.w_macd_cross),
            ("MACD vs signal", _bool_vote(snap.macd_above_signal), p.w_macd_above),
            ("RSI extreme", _rsi_vote(snap), p.w_rsi_extreme),
            ("Bollinger band", _bollinger_vote(snap), p.w_bollinger),
            ("price vs VWAP", _bool_vote(snap.price_above_vwap), p.w_vwap),
        ]

        weighted_sum = 0.0
        total_weight = 0.0
        reasons: list[str] = []
        for name, vote, weight in components:
            if weight <= 0:
                continue
            total_weight += weight
            if vote != 0:
                weighted_sum += vote * weight
                reasons.append(_lean_phrase(name, float(vote)))

        score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
        if not reasons:
            reasons.append("no technical flags active")
        return _clamp(score, -1.0, 1.0), reasons

    # ------------------------------------------------------------------
    # Sentiment helper (optional convenience)
    # ------------------------------------------------------------------
    def sentiment_for_symbols(
        self,
        finnhub: FinnhubData,
        lookback_days: int = 7,
        sector_weight: float = 0.5,
    ) -> tuple[float, str]:
        """Pull and blend semis sentiment from Finnhub into a ``[-1, 1]`` score.

        Combines the long leg's company-news sentiment with the broader
        semiconductor-sector sentiment (a more stable, less ETF-specific read).
        News about SOXS itself is sparse and its sign would invert, so we anchor
        on the long symbol and the sector and let the sign express bullishness.

        Args:
            finnhub: shared :class:`~data.finnhub_data.FinnhubData` instance.
            lookback_days: news window passed through to Finnhub.
            sector_weight: blend weight on sector sentiment vs. the long leg, in
                ``[0, 1]``. ``0`` uses only the long symbol; ``1`` only the sector.

        Returns:
            ``(score, meta)`` where ``score`` is the blended sentiment and
            ``meta`` is a short provenance string for the signal reasons.
        """
        sector_weight = _clamp(sector_weight, 0.0, 1.0)
        long_res = finnhub.get_news_sentiment(self._symbol_long, lookback_days=lookback_days)

        sector_scores: list[float] = []
        for sym in finnhub.sector_symbols:
            try:
                sector_scores.append(
                    finnhub.get_news_sentiment(sym, lookback_days=lookback_days).score
                )
            except Exception as exc:  # noqa: BLE001 - never let one ticker sink the cycle
                logger.warning("Sentiment fetch failed for %s: %s", sym, exc)
        sector_avg = sum(sector_scores) / len(sector_scores) if sector_scores else 0.0

        if sector_scores:
            blended = (1.0 - sector_weight) * long_res.score + sector_weight * sector_avg
            meta = (
                f"{long_res.source}: {self._symbol_long}={long_res.score:+.2f}, "
                f"sector({len(sector_scores)})={sector_avg:+.2f}"
            )
        else:
            blended = long_res.score
            meta = f"{long_res.source}: {self._symbol_long}={long_res.score:+.2f}"

        return _clamp(blended, -1.0, 1.0), meta

    def evaluate(
        self,
        snapshot: IndicatorSnapshot,
        finnhub: FinnhubData | None = None,
        lookback_days: int = 7,
    ) -> TradeSignal:
        """Convenience: pull sentiment (if a Finnhub client is given) and generate.

        Skips the sentiment fetch entirely when ``finnhub`` is ``None`` or the
        sentiment weight is zero, decaying to a purely technical decision.
        """
        if finnhub is None or self._config.sentiment_weight <= 0:
            return self.generate(snapshot)
        try:
            score, meta = self.sentiment_for_symbols(finnhub, lookback_days=lookback_days)
        except Exception as exc:  # noqa: BLE001 - sentiment is best-effort
            logger.warning("Sentiment unavailable, using technicals only: %s", exc)
            return self.generate(snapshot, 0.0, sentiment_meta="unavailable")
        return self.generate(snapshot, score, sentiment_meta=meta)


# ----------------------------------------------------------------------
# Vote helpers — each maps snapshot state to -1 / 0 / +1
# ----------------------------------------------------------------------
def _bool_vote(bullish: bool) -> int:
    """A two-sided boolean flag: +1 if True (bullish), -1 if False (bearish)."""
    return 1 if bullish else -1


def _stack_vote(snap: IndicatorSnapshot) -> int:
    if snap.ema_bullish_stack:
        return 1
    if snap.ema_bearish_stack:
        return -1
    return 0


def _macd_cross_vote(snap: IndicatorSnapshot) -> int:
    if snap.macd_bullish_cross:
        return 1
    if snap.macd_bearish_cross:
        return -1
    return 0


def _rsi_vote(snap: IndicatorSnapshot) -> int:
    # Oversold is a (mean-reversion) bullish tell; overbought is bearish.
    if snap.rsi_oversold:
        return 1
    if snap.rsi_overbought:
        return -1
    return 0


def _bollinger_vote(snap: IndicatorSnapshot) -> int:
    # Close below the lower band leans bullish (snap-back); above upper, bearish.
    if snap.price_below_bb_lower:
        return 1
    if snap.price_above_bb_upper:
        return -1
    return 0


def _lean_phrase(name: str, value: float) -> str:
    lean = "bullish" if value > 0 else "bearish" if value < 0 else "neutral"
    return f"{name} {lean} ({value:+.2f})"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
