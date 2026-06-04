"""Monitoring, P&L tracking, and alerting for the trading loop.

The :class:`Monitor` is the bot's observability layer. It does not make
decisions — it *watches* the ones the rest of the pipeline makes and:

  * **logs every decision and trade** (signals, executions, exits, errors) at a
    consistent altitude, so the log file is a faithful audit trail;
  * **computes running P&L** — both session-to-date (since the bot started) and
    intraday (against the risk manager's start-of-day equity), plus peak equity
    and max drawdown — purely from account equity, so it stays correct without
    reconstructing fill-level cost basis;
  * **sends Discord alerts** on the events that matter operationally: entries,
    exits (forced, flip, or end-of-day), errors, and risk-limit breaches;
  * **prints a compact status summary** each cycle, returned as a string and
    logged at INFO.

Alert delivery is delegated to :class:`~monitoring.alerts.DiscordNotifier`,
which is non-blocking and best-effort, so monitoring never slows or breaks the
trading cycle. Risk-limit-breach alerts are de-duplicated per trading day so a
breached limit fires once, not every cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from config.logging_setup import get_logger
from monitoring.alerts import AlertLevel, DiscordNotifier

logger = get_logger(__name__)


@dataclass
class SessionStats:
    """Running counters for the life of the bot process (not per-day)."""

    cycles: int = 0
    signals: int = 0
    entries: int = 0
    flips: int = 0
    exits: int = 0
    vetoes: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "cycles": self.cycles,
            "signals": self.signals,
            "entries": self.entries,
            "flips": self.flips,
            "exits": self.exits,
            "vetoes": self.vetoes,
            "errors": self.errors,
        }


class Monitor:
    """Observes the trading loop: logs, tracks P&L, and raises alerts.

    Args:
        notifier: the Discord notifier (may be disabled; calls are still safe).
        symbol_long: the bullish leg, for context in alerts (default SOXL).
        symbol_short: the bearish leg, for context in alerts (default SOXS).
    """

    def __init__(
        self,
        notifier: DiscordNotifier,
        *,
        symbol_long: str = "SOXL",
        symbol_short: str = "SOXS",
    ) -> None:
        self._notifier = notifier
        self._symbol_long = symbol_long
        self._symbol_short = symbol_short

        self.stats = SessionStats()

        # P&L anchors. Session start is fixed at the first equity we observe;
        # daily start comes from the risk manager's status each cycle.
        self._session_start_equity: float | None = None
        self._peak_equity: float = 0.0
        self._last_equity: float = 0.0

        # Per-day de-dup so a breached limit alerts once, not every cycle.
        self._breach_day: str | None = None
        self._breached: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_start(self, equity: float, *, mode: str, poll_interval: int) -> None:
        """Anchor session P&L and announce startup."""
        self._observe_equity(equity)
        self._session_start_equity = float(equity)
        logger.info(
            "Monitor online | mode=%s start_equity=%.2f poll=%ds",
            mode, float(equity), poll_interval,
        )
        self._notifier.send(
            "🟢 Trading bot started",
            f"Watching **{self._symbol_long}/{self._symbol_short}** in **{mode}** mode.",
            level=AlertLevel.INFO,
            fields={
                "Equity": f"${float(equity):,.2f}",
                "Poll": f"{poll_interval}s",
            },
        )

    def on_stop(self, equity: float) -> None:
        """Report final session P&L, announce shutdown, and flush alerts."""
        pnl, pct = self.session_pnl(equity)
        logger.info(
            "Monitor offline | session P&L=%+.2f (%+.2f%%) | %s",
            pnl, pct, self.stats.as_dict(),
        )
        self._notifier.send(
            "🔴 Trading bot stopped",
            f"Session P&L **{_money(pnl)}** ({pct:+.2f}%).",
            level=AlertLevel.WARNING if pnl < 0 else AlertLevel.INFO,
            fields={
                "Equity": f"${float(equity):,.2f}",
                "Entries": str(self.stats.entries),
                "Exits": str(self.stats.exits),
                "Errors": str(self.stats.errors),
            },
        )
        self._notifier.close()

    # ------------------------------------------------------------------
    # Decisions & executions
    # ------------------------------------------------------------------
    def record_cycle(self) -> None:
        self.stats.cycles += 1

    def record_signal(self, signal: object) -> None:
        """Log a generated trade decision (every decision is recorded)."""
        self.stats.signals += 1
        logger.info("Decision: %s", signal)

    def record_execution(self, result: object) -> None:
        """Log a trade outcome and alert on entries, flips, vetoes, or errors.

        ``result`` is an :class:`~execution.order_manager.ExecutionResult`.
        Routine no-ops (``skipped``) are not alerted; risk-limit vetoes surface
        via :meth:`record_risk_state` instead, so this stays low-noise.
        """
        action = getattr(result, "action", "unknown")
        message = getattr(result, "message", "")
        decision = getattr(result, "decision", None)

        if action in ("entered", "flipped"):
            self.stats.entries += 1
            fields = _decision_fields(decision)
            if action == "flipped":
                self.stats.flips += 1
                closed = getattr(result, "closed_orders", None) or []
                legs = ", ".join(getattr(o, "symbol", "?") for o in closed)
                if legs:
                    fields["Closed"] = legs
            title = "📈 Entry" if action == "entered" else "🔄 Flip"
            logger.info("Trade %s: %s", action, message)
            self._notifier.send(title, message, level=AlertLevel.SUCCESS, fields=fields)

        elif action == "vetoed":
            self.stats.vetoes += 1
            logger.info("Trade vetoed: %s", message)

        elif action == "error":
            self.stats.errors += 1
            logger.error("Execution error: %s", message)
            self._notifier.send(
                "⚠️ Execution error",
                message,
                level=AlertLevel.ERROR,
                fields=_decision_fields(decision),
            )

        else:  # "skipped" / unknown — log only, no alert
            logger.debug("Execution %s: %s", action, message)

    def record_exits(
        self, orders: Sequence[object], *, reason: str, forced: bool = False
    ) -> None:
        """Alert on positions closed outside a normal entry (EOD, flip, forced).

        ``forced`` raises the alert to WARNING (risk-driven flatten / invariant
        breach) versus a routine INFO exit (end-of-day flat).
        """
        if not orders:
            return
        self.stats.exits += len(orders)
        legs = ", ".join(
            f"{getattr(o, 'symbol', '?')} x{getattr(o, 'qty', '?')}" for o in orders
        )
        level = AlertLevel.WARNING if forced else AlertLevel.INFO
        title = "🛑 Forced exit" if forced else "📉 Exit"
        log = logger.warning if forced else logger.info
        log("%s (%s): %s", title, reason, legs)
        self._notifier.send(title, reason, level=level, fields={"Closed": legs})

    def record_error(self, context: str, exc: BaseException) -> None:
        """Alert on an unexpected error in the loop (one bad cycle, a failure)."""
        self.stats.errors += 1
        logger.error("%s: %s", context, exc)
        self._notifier.send(
            "🔥 Bot error",
            f"{context}: `{exc}`",
            level=AlertLevel.ERROR,
        )

    # ------------------------------------------------------------------
    # Risk-limit breaches (de-duplicated per day)
    # ------------------------------------------------------------------
    def record_risk_state(self, risk_status: Mapping[str, object]) -> None:
        """Inspect the risk snapshot and alert once per day on a breached limit.

        Fires for the daily-loss stop and the max-trades-per-day cap; each key
        alerts a single time per trading day (reset when the day rolls over).
        """
        day = str(risk_status.get("day"))
        if day != self._breach_day:
            self._breach_day = day
            self._breached.clear()

        if risk_status.get("daily_loss_breached") and self._first_breach("daily_loss"):
            dd = float(risk_status.get("daily_drawdown", 0.0)) * 100
            self._notifier.send(
                "🚨 Daily loss limit breached",
                f"Drawdown **{dd:.2f}%** — new entries vetoed and open exposure flattened.",
                level=AlertLevel.ERROR,
                fields={
                    "Equity": f"${float(risk_status.get('equity', 0.0)):,.2f}",
                    "Start": f"${float(risk_status.get('start_equity', 0.0)):,.2f}",
                },
            )

        if risk_status.get("trade_limit_reached") and self._first_breach("trade_limit"):
            self._notifier.send(
                "🚧 Max trades/day reached",
                f"{risk_status.get('trades_today')}/"
                f"{risk_status.get('max_trades_per_day')} trades — no new entries today.",
                level=AlertLevel.WARNING,
            )

    def _first_breach(self, key: str) -> bool:
        """True the first time ``key`` breaches on the current day."""
        if key in self._breached:
            return False
        self._breached.add(key)
        return True

    # ------------------------------------------------------------------
    # P&L
    # ------------------------------------------------------------------
    def session_pnl(self, equity: float) -> tuple[float, float]:
        """``(dollars, percent)`` P&L since the bot started."""
        start = self._session_start_equity
        if start is None or start <= 0:
            return 0.0, 0.0
        pnl = float(equity) - start
        return pnl, pnl / start * 100.0

    @property
    def max_drawdown_pct(self) -> float:
        """Worst peak-to-current drawdown seen this session, as a percent."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._last_equity) / self._peak_equity * 100.0

    def _observe_equity(self, equity: float) -> None:
        self._last_equity = float(equity)
        self._peak_equity = max(self._peak_equity, self._last_equity)

    # ------------------------------------------------------------------
    # Status summary
    # ------------------------------------------------------------------
    def status_summary(
        self,
        equity: float,
        risk_status: Mapping[str, object],
        open_orders: int,
    ) -> str:
        """Build, log, and return the per-cycle status line; checks breaches.

        Side effects: updates the equity high-water mark and fires any
        first-time risk-limit-breach alerts via :meth:`record_risk_state`.
        """
        self._observe_equity(equity)
        self.record_risk_state(risk_status)

        sess_pnl, sess_pct = self.session_pnl(equity)
        day_dd = float(risk_status.get("daily_drawdown", 0.0)) * 100
        summary = (
            f"Status | equity=${float(equity):,.2f} "
            f"| session P&L {_money(sess_pnl)} ({sess_pct:+.2f}%) "
            f"| day dd={day_dd:.2f}% maxdd={self.max_drawdown_pct:.2f}% "
            f"| trades={risk_status.get('trades_today')}/"
            f"{risk_status.get('max_trades_per_day')} "
            f"| open_orders={open_orders}"
        )
        logger.info(summary)
        return summary


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------
def _money(amount: float) -> str:
    """Signed dollar amount, e.g. ``+$1,234.50`` / ``-$42.00``."""
    sign = "-" if amount < 0 else "+"
    return f"{sign}${abs(amount):,.2f}"


def _decision_fields(decision: object) -> dict[str, object]:
    """Extract entry/bracket fields off a ``RiskDecision`` for an alert embed."""
    if decision is None:
        return {}
    fields: dict[str, object] = {}
    target = getattr(decision, "target_symbol", None)
    if target:
        fields["Symbol"] = target
    qty = getattr(decision, "qty", None)
    if qty:
        fields["Qty"] = str(qty)
    entry = getattr(decision, "entry_price", None)
    if entry:
        fields["Entry"] = f"${float(entry):,.2f}"
    stop = getattr(decision, "stop_loss", None)
    take = getattr(decision, "take_profit", None)
    if stop and take:
        fields["Bracket"] = f"sl ${float(stop):,.2f} / tp ${float(take):,.2f}"
    return fields
