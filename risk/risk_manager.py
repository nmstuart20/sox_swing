"""Risk management for the SOXL/SOXS pair.

The :class:`RiskManager` is the gate every signal passes through before it can
become an order. It is *pure policy*: it never touches Alpaca or fetches data,
it only reasons over a signal, the account equity, the proposed entry price and
ATR, and the current positions, and returns a structured verdict.

It enforces five rules from the build spec:

  1. **Max position size** — capped as a fraction of equity (notional cap).
  2. **Max daily loss** — once equity has drawn down past the limit, all new
     entries are vetoed and open positions are flagged for a forced exit.
  3. **Max trades per day** — a hard count, reset each session.
  4. **ATR-based stop-loss / take-profit** — exit levels derived from the
     symbol's ATR, so they breathe with volatility.
  5. **Never hold SOXL and SOXS at once** — an entry into one leg while the
     other is open is approved only with an instruction to close the opposite
     leg first; if both legs are somehow held, both are force-flattened.

SOXL and SOXS are **3x leveraged** ETFs, so sizing is deliberately
conservative: the position is the *smaller* of a notional cap and a per-trade
dollar-risk budget measured against the ATR stop, and it is always rounded down
to whole shares (leveraged-ETF brackets can't be fractional on Alpaca anyway).

State is per-trading-day: the manager tracks the day's starting equity and the
trade count, auto-resetting when the calendar date rolls over (or explicitly via
:meth:`reset_daily`). The order manager (step 8) is responsible for *acting* on
a decision — closing the opposite leg, placing the bracket, and calling
:meth:`register_entry` once a fill is confirmed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Mapping

from config.logging_setup import get_logger
from config.settings import RiskConfig
from strategy.signal_engine import Direction, TradeSignal

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Position protocol
# ----------------------------------------------------------------------
# The manager only needs a couple of fields off Alpaca's ``Position`` (qty and
# unrealized P&L, both string-typed on the SDK model). Accepting anything with
# those attributes keeps the module trivially testable with a stub.
def _position_qty(position: object) -> float:
    """Best-effort signed share count for a position-like object."""
    raw = getattr(position, "qty", 0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _position_unrealized(position: object) -> float:
    raw = getattr(position, "unrealized_pl", 0) or 0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class RiskParams:
    """Tunables that refine :class:`~config.settings.RiskConfig`.

    ``risk_per_trade_pct`` is the fraction of equity put at risk between the
    entry and the ATR stop on a single trade; together with the notional cap it
    bounds size from two directions. ``min_qty`` is the smallest tradeable lot
    (1 whole share). ``min_atr_pct`` floors a suspiciously small ATR so a quiet
    tape can't produce a hair-thin stop and an enormous position.
    """

    risk_per_trade_pct: float = 0.01      # 1% of equity risked to the stop
    min_qty: int = 1
    min_atr_pct: float = 0.001            # floor ATR at 0.1% of price

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade_pct <= 1:
            raise ValueError("risk_per_trade_pct must be in (0, 1].")
        if self.min_qty < 1:
            raise ValueError("min_qty must be >= 1.")
        if self.min_atr_pct < 0:
            raise ValueError("min_atr_pct must be >= 0.")


@dataclass(frozen=True)
class RiskDecision:
    """The verdict on a single signal.

    When ``approved`` is True the order manager may open ``qty`` shares of
    ``target_symbol`` with the given ``entry_price`` and bracket
    (``stop_loss``/``take_profit``), but it must first close any symbol listed
    in ``close_symbols`` (the opposite leg) to preserve the no-both-legs rule.
    When ``approved`` is False, ``reason`` explains the veto and the sizing
    fields are zeroed.
    """

    approved: bool
    reason: str
    target_symbol: str | None = None
    direction: Direction = Direction.NEUTRAL
    qty: int = 0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    close_symbols: tuple[str, ...] = field(default_factory=tuple)

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price

    def to_dict(self) -> dict[str, object]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "target_symbol": self.target_symbol,
            "direction": self.direction.value,
            "qty": self.qty,
            "entry_price": round(self.entry_price, 4),
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
            "notional": round(self.notional, 2),
            "close_symbols": list(self.close_symbols),
        }

    def __str__(self) -> str:
        if not self.approved:
            return f"VETO: {self.reason}"
        flip = f" (close {','.join(self.close_symbols)} first)" if self.close_symbols else ""
        return (
            f"APPROVE {self.target_symbol} x{self.qty} @ {self.entry_price:.2f} "
            f"[sl={self.stop_loss:.2f} tp={self.take_profit:.2f}]{flip}"
        )


@dataclass(frozen=True)
class ForcedExit:
    """An instruction to flatten a held position regardless of signal."""

    symbol: str
    reason: str


class RiskManager:
    """Validates/vetoes signals and orders forced exits when limits are hit.

    Args:
        config: account-level risk limits from :class:`~config.settings.RiskConfig`.
        symbol_long: the bullish leg (default SOXL).
        symbol_short: the bearish leg (default SOXS).
        params: sizing/threshold refinements; defaults to :class:`RiskParams`.
    """

    def __init__(
        self,
        config: RiskConfig,
        symbol_long: str = "SOXL",
        symbol_short: str = "SOXS",
        params: RiskParams | None = None,
    ) -> None:
        self._config = config
        self._symbol_long = symbol_long
        self._symbol_short = symbol_short
        self._params = params or RiskParams()
        self._symbols = (symbol_long, symbol_short)

        # Per-day state — lazily initialized on first use / explicit reset.
        self._day: date | None = None
        self._start_equity: float = 0.0
        self._trades_today: int = 0
        self._realized_pnl_today: float = 0.0

        logger.info(
            "RiskManager initialized (max_pos=%.0f%% equity, max_daily_loss=%.0f%%, "
            "max_trades/day=%d, stop=%.1f*ATR, tp=%.1f*ATR, risk/trade=%.0f%%)",
            config.max_position_pct * 100,
            config.max_daily_loss_pct * 100,
            config.max_trades_per_day,
            config.atr_stop_multiplier,
            config.atr_take_profit_multiplier,
            self._params.risk_per_trade_pct * 100,
        )

    # ------------------------------------------------------------------
    # Per-day state
    # ------------------------------------------------------------------
    @property
    def trades_today(self) -> int:
        return self._trades_today

    @property
    def start_equity(self) -> float:
        return self._start_equity

    @property
    def realized_pnl_today(self) -> float:
        return self._realized_pnl_today

    def reset_daily(self, equity: float, *, now: datetime | None = None) -> None:
        """Start a fresh trading day: anchor the day's equity and zero counters."""
        now = now or datetime.now(timezone.utc)
        self._day = now.date()
        self._start_equity = max(float(equity), 0.0)
        self._trades_today = 0
        self._realized_pnl_today = 0.0
        logger.info(
            "Risk day reset for %s (start_equity=%.2f)", self._day, self._start_equity
        )

    def maybe_reset_for_day(self, equity: float, *, now: datetime | None = None) -> bool:
        """Reset counters if the calendar date has rolled (or on first use).

        Returns True if a reset happened. The orchestration loop calls this each
        cycle so daily limits track the actual trading session.
        """
        now = now or datetime.now(timezone.utc)
        if self._day != now.date():
            self.reset_daily(equity, now=now)
            return True
        return False

    def register_entry(self) -> None:
        """Record that an entry order was placed (counts against the daily cap).

        Call this once per confirmed entry, from the order manager — not on
        forced exits, which don't open new exposure.
        """
        self._trades_today += 1
        logger.info(
            "Trade %d/%d recorded for %s",
            self._trades_today, self._config.max_trades_per_day, self._day,
        )

    def record_realized_pnl(self, amount: float) -> None:
        """Accumulate realized P&L for the day (for reporting/alerts)."""
        self._realized_pnl_today += float(amount)

    # ------------------------------------------------------------------
    # Limit checks
    # ------------------------------------------------------------------
    def daily_drawdown(self, equity: float) -> float:
        """Fractional drop from the day's starting equity (0 if up or flat)."""
        if self._start_equity <= 0:
            return 0.0
        drop = (self._start_equity - float(equity)) / self._start_equity
        return max(drop, 0.0)

    def daily_loss_breached(self, equity: float) -> bool:
        """True once the day's drawdown meets/exceeds the configured limit."""
        return self.daily_drawdown(equity) >= self._config.max_daily_loss_pct

    def trade_limit_reached(self) -> bool:
        return self._trades_today >= self._config.max_trades_per_day

    # ------------------------------------------------------------------
    # Sizing & bracket levels
    # ------------------------------------------------------------------
    def stop_and_take_profit(
        self, direction: Direction, entry_price: float, atr: float
    ) -> tuple[float, float]:
        """ATR-based (stop_loss, take_profit) for a long entry into the target.

        Both legs are entered *long* (we buy shares of whichever ETF the signal
        favors), so the stop sits below entry and the take-profit above it. ATR
        is floored by ``min_atr_pct`` of price to avoid a degenerate stop.
        """
        atr = max(atr, self._params.min_atr_pct * entry_price)
        stop_dist = self._config.atr_stop_multiplier * atr
        tp_dist = self._config.atr_take_profit_multiplier * atr
        # Keep the stop strictly positive even for a wild ATR on a low-priced ETF.
        stop = max(entry_price - stop_dist, 0.01)
        # With a trailing stop the position has no fixed ceiling — the ratcheting
        # stop harvests winners instead — so a take-profit of 0 disables it.
        take = 0.0 if self._config.trailing_stop else entry_price + tp_dist
        return stop, take

    def position_size(self, equity: float, entry_price: float, stop_loss: float) -> int:
        """Whole-share size: the smaller of the notional cap and risk budget.

        * notional cap — ``max_position_pct`` of equity / entry price,
        * risk budget — ``risk_per_trade_pct`` of equity / stop distance.

        Taking the minimum keeps leveraged exposure conservative regardless of
        how tight or wide the ATR stop happens to be. Rounded down to whole
        shares.
        """
        if entry_price <= 0 or equity <= 0:
            return 0
        notional_cap = self._config.max_position_pct * equity
        qty_by_notional = notional_cap / entry_price

        stop_dist = entry_price - stop_loss
        if stop_dist > 0:
            risk_budget = self._params.risk_per_trade_pct * equity
            qty_by_risk = risk_budget / stop_dist
            qty = min(qty_by_notional, qty_by_risk)
        else:
            qty = qty_by_notional

        return max(int(math.floor(qty)), 0)

    # ------------------------------------------------------------------
    # The gate
    # ------------------------------------------------------------------
    def evaluate(
        self,
        signal: TradeSignal,
        equity: float,
        entry_price: float,
        atr: float,
        positions: Mapping[str, object | None] | None = None,
        *,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Approve or veto ``signal`` and, if approved, size the position.

        Args:
            signal: the engine's decision for the pair.
            equity: current total account equity.
            entry_price: latest price of the *target* symbol (the leg to buy).
            atr: ATR of the *target* symbol, in price terms.
            positions: current ``{symbol: position_or_None}`` for the pair; used
                to enforce the no-both-legs rule and skip redundant re-entries.
            now: clock override for testing the daily roll.

        Returns:
            A :class:`RiskDecision`. Approval may carry ``close_symbols`` when
            the opposite leg must be flattened first (a flip).
        """
        self.maybe_reset_for_day(equity, now=now)
        positions = positions or {}

        # 1. Nothing to do on a neutral / non-actionable signal.
        if not signal.is_actionable or signal.target_symbol is None:
            return self._veto("signal is neutral — no entry")

        target = signal.target_symbol
        if target not in self._symbols:
            return self._veto(f"target {target} is not a managed symbol")
        opposite = self._other_leg(target)

        # 2. Daily loss limit — hard stop on new exposure.
        if self.daily_loss_breached(equity):
            return self._veto(
                f"daily loss limit hit "
                f"(drawdown {self.daily_drawdown(equity):.1%} >= "
                f"{self._config.max_daily_loss_pct:.1%})"
            )

        # 3. Trade-count cap.
        if self.trade_limit_reached():
            return self._veto(
                f"max trades/day reached ({self._trades_today}/"
                f"{self._config.max_trades_per_day})"
            )

        # 4. Already positioned in the target leg — don't pyramid.
        if _has_position(positions.get(target)):
            return self._veto(f"already holding {target} — no add")

        # 5. Sanity on price/ATR inputs.
        if entry_price <= 0:
            return self._veto(f"invalid entry price {entry_price!r}")
        if atr <= 0:
            return self._veto(f"invalid ATR {atr!r}")

        # 6. Size it and derive the bracket.
        stop, take = self.stop_and_take_profit(signal.direction, entry_price, atr)
        qty = self.position_size(equity, entry_price, stop)
        if qty < self._params.min_qty:
            return self._veto(
                f"sized position {qty} < min {self._params.min_qty} "
                f"(equity={equity:.2f}, entry={entry_price:.2f})"
            )

        # 7. No-both-legs rule: if the opposite leg is open, it must close first.
        #    Reversals also pass a hysteresis gate so a marginal opposite signal
        #    can't whipsaw an existing position in and out (both directions pay
        #    slippage + commission).
        close_symbols: tuple[str, ...] = ()
        if _has_position(positions.get(opposite)):
            flip_bar = self._config.flip_confidence_threshold
            if signal.confidence < flip_bar:
                return self._veto(
                    f"flip to {target} below hysteresis "
                    f"(confidence {signal.confidence:.2f} < {flip_bar:.2f}) — holding {opposite}"
                )
            close_symbols = (opposite,)

        decision = RiskDecision(
            approved=True,
            reason="within risk limits",
            target_symbol=target,
            direction=signal.direction,
            qty=qty,
            entry_price=entry_price,
            stop_loss=stop,
            take_profit=take,
            close_symbols=close_symbols,
        )
        logger.info("Risk decision: %s", decision)
        return decision

    # ------------------------------------------------------------------
    # Forced exits
    # ------------------------------------------------------------------
    def forced_exits(
        self,
        equity: float,
        positions: Mapping[str, object | None],
        *,
        now: datetime | None = None,
    ) -> list[ForcedExit]:
        """Positions that must be closed *now*, independent of any signal.

        Two triggers:
          * both legs held at once — an invariant violation, flatten everything;
          * the daily loss limit is breached — flatten all open exposure.

        Returns an empty list when nothing needs flattening.
        """
        self.maybe_reset_for_day(equity, now=now)
        held = [sym for sym in self._symbols if _has_position(positions.get(sym))]
        if not held:
            return []

        # Both legs open simultaneously should never happen; treat as an alarm
        # and flatten both so the next cycle can re-enter cleanly.
        if len(held) >= 2:
            logger.error(
                "INVARIANT BREACH: holding both %s and %s — forcing flat",
                *self._symbols,
            )
            return [ForcedExit(sym, "both legs held — invariant violation") for sym in held]

        if self.daily_loss_breached(equity):
            reason = (
                f"daily loss limit breached "
                f"(drawdown {self.daily_drawdown(equity):.1%})"
            )
            logger.warning("Forcing exit of %s: %s", held, reason)
            return [ForcedExit(sym, reason) for sym in held]

        return []

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def status(self, equity: float) -> dict[str, object]:
        """A compact snapshot of risk state for monitoring/logging."""
        return {
            "day": str(self._day),
            "start_equity": round(self._start_equity, 2),
            "equity": round(float(equity), 2),
            "daily_drawdown": round(self.daily_drawdown(equity), 4),
            "daily_loss_breached": self.daily_loss_breached(equity),
            "trades_today": self._trades_today,
            "max_trades_per_day": self._config.max_trades_per_day,
            "trade_limit_reached": self.trade_limit_reached(),
            "realized_pnl_today": round(self._realized_pnl_today, 2),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _other_leg(self, symbol: str) -> str:
        return self._symbol_short if symbol == self._symbol_long else self._symbol_long

    @staticmethod
    def _veto(reason: str) -> RiskDecision:
        logger.info("Risk veto: %s", reason)
        return RiskDecision(approved=False, reason=reason)


def _has_position(position: object | None) -> bool:
    """True when ``position`` represents a non-zero open holding."""
    return position is not None and abs(_position_qty(position)) > 0
