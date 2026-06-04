"""Order execution and position management for the SOXL/SOXS pair.

The :class:`OrderManager` is the bridge between a *decision* and a *fill*. It
takes a :class:`~strategy.signal_engine.TradeSignal`, runs it through the
:class:`~risk.risk_manager.RiskManager` to size the position and derive the
ATR bracket, and then — if the trade is approved — places the order through the
:class:`~execution.alpaca_client.AlpacaClient` as a bracket entry (market buy +
attached take-profit limit and stop-loss). Whichever bracket leg fills first
closes the position; the loop can still close early (signal flip, forced exit,
or end-of-day flat).

It owns the messy, stateful parts the risk manager deliberately stays out of:

  * **Flipping legs.** SOXL and SOXS must never be held at once, so when a
    decision carries ``close_symbols`` the manager liquidates the opposite leg
    and *waits until the account is flat in that symbol* before opening the new
    bracket. This avoids a window where both legs are live, and frees the
    buying power tied up in the leg being closed.
  * **Tracking open orders.** Every order the manager submits (entries and the
    liquidations behind a flip) is recorded as a :class:`ManagedOrder` and
    followed to a terminal state.
  * **Reconciling fills.** :meth:`reconcile` polls Alpaca for the latest status
    of each tracked order, updates the local record, and — the moment an entry
    is confirmed filled — calls :meth:`RiskManager.register_entry` exactly once
    so the daily trade counter reflects real exposure (not merely submitted
    orders, which may be rejected).

Everything routes through ``AlpacaClient`` and ``RiskManager``; this module
holds no SDK or policy logic of its own, only orchestration and bookkeeping.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

from alpaca.trading.enums import OrderSide
from alpaca.trading.models import Order

from config.logging_setup import get_logger
from execution.alpaca_client import AlpacaClient, AlpacaClientError
from risk.risk_manager import ForcedExit, RiskDecision, RiskManager
from strategy.signal_engine import TradeSignal

logger = get_logger(__name__)


# Order statuses Alpaca considers final — nothing further will happen to them.
TERMINAL_STATUSES = frozenset(
    {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day"}
)


def _order_status(order: object) -> str:
    """Lower-cased status string off an Alpaca ``Order`` (enum or str)."""
    status = getattr(order, "status", None)
    value = getattr(status, "value", status)
    return str(value).lower() if value is not None else "unknown"


def _to_float(value: object, default: float = 0.0) -> float:
    """Coerce an Alpaca string/Decimal field to float, defaulting on junk."""
    if value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


@dataclass
class ManagedOrder:
    """A single order this manager submitted and is following to completion.

    ``kind`` is ``"entry"`` for a position-opening bracket or ``"exit"`` for a
    liquidation (the close side of a flip or a forced/EOD exit). ``registered``
    guards the one-shot call into :meth:`RiskManager.register_entry`, so an
    entry counts against the daily cap exactly once even if it is reconciled
    repeatedly.
    """

    order_id: str
    symbol: str
    side: str
    qty: float
    kind: str  # "entry" | "exit"
    client_order_id: str | None = None
    status: str = "new"
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    reason: str = ""
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    registered: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_filled(self) -> bool:
        return self.status == "filled" or self.filled_qty > 0

    def update_from(self, order: Order) -> None:
        """Refresh local fields from a freshly fetched Alpaca ``Order``."""
        self.status = _order_status(order)
        self.filled_qty = _to_float(getattr(order, "filled_qty", None))
        self.filled_avg_price = _to_float(getattr(order, "filled_avg_price", None))

    def to_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "kind": self.kind,
            "status": self.status,
            "filled_qty": self.filled_qty,
            "filled_avg_price": round(self.filled_avg_price, 4),
            "reason": self.reason,
        }


@dataclass
class ExecutionResult:
    """Outcome of handling a single signal.

    ``action`` is one of:
      * ``"entered"``  — a new bracket was opened (no flip needed),
      * ``"flipped"``  — the opposite leg was closed and a new bracket opened,
      * ``"vetoed"``   — the risk manager rejected the signal (see ``message``),
      * ``"skipped"``  — nothing actionable (neutral signal / already positioned),
      * ``"error"``    — an Alpaca/operational failure (see ``message``).
    """

    action: str
    message: str
    decision: RiskDecision | None = None
    entry_order: ManagedOrder | None = None
    closed_orders: list[ManagedOrder] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.action not in ("error",)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "message": self.message,
            "decision": self.decision.to_dict() if self.decision else None,
            "entry_order": self.entry_order.to_dict() if self.entry_order else None,
            "closed_orders": [o.to_dict() for o in self.closed_orders],
        }


class OrderManager:
    """Turns approved signals into bracket orders and manages the lifecycle.

    Args:
        client: the Alpaca trading wrapper.
        risk_manager: the policy gate that sizes positions and sets brackets.
        symbol_long: the bullish leg (default SOXL).
        symbol_short: the bearish leg (default SOXS).
        flip_timeout: max seconds to wait for the opposite leg to go flat
            before opening the new bracket on a flip.
        fill_timeout: max seconds to wait for an entry to fill before returning
            (the order is still tracked and reconciled afterwards).
        poll_interval: seconds between status polls while waiting.
    """

    def __init__(
        self,
        client: AlpacaClient,
        risk_manager: RiskManager,
        symbol_long: str = "SOXL",
        symbol_short: str = "SOXS",
        *,
        flip_timeout: float = 30.0,
        fill_timeout: float = 30.0,
        poll_interval: float = 2.0,
    ) -> None:
        self._client = client
        self._risk = risk_manager
        self._symbol_long = symbol_long
        self._symbol_short = symbol_short
        self._symbols = (symbol_long, symbol_short)
        self._flip_timeout = flip_timeout
        self._fill_timeout = fill_timeout
        self._poll_interval = poll_interval

        # Orders this manager is actively tracking, keyed by Alpaca order id.
        self._open_orders: dict[str, ManagedOrder] = {}

        logger.info(
            "OrderManager initialized (long=%s, short=%s, flip_timeout=%.0fs, "
            "fill_timeout=%.0fs)",
            symbol_long, symbol_short, flip_timeout, fill_timeout,
        )

    # ------------------------------------------------------------------
    # Tracked-order accessors
    # ------------------------------------------------------------------
    @property
    def open_orders(self) -> list[ManagedOrder]:
        """Orders still being followed (not yet in a terminal state)."""
        return [o for o in self._open_orders.values() if not o.is_terminal]

    def tracked(self, order_id: str) -> ManagedOrder | None:
        return self._open_orders.get(order_id)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def execute_signal(
        self,
        signal: TradeSignal,
        equity: float,
        entry_price: float,
        atr: float,
        positions: Mapping[str, object | None] | None = None,
        *,
        now: datetime | None = None,
        wait_for_fill: bool = True,
    ) -> ExecutionResult:
        """Size, gate, and (if approved) place a bracket order for ``signal``.

        Args:
            signal: the engine's decision for the pair.
            equity: current account equity (for sizing and the daily-loss gate).
            entry_price: latest price of the *target* leg the signal favors.
            atr: ATR of the target leg, in price terms (for the bracket).
            positions: current ``{symbol: position_or_None}`` for the pair. If
                omitted, it is fetched from Alpaca.
            now: clock override (forwarded to the risk manager's daily roll).
            wait_for_fill: when True, block up to ``fill_timeout`` for the entry
                to fill so :meth:`RiskManager.register_entry` runs before this
                returns; when False, the entry is tracked and reconciled later.

        Returns:
            An :class:`ExecutionResult` describing what happened.
        """
        if positions is None:
            positions = self._fetch_positions()

        decision = self._risk.evaluate(
            signal, equity, entry_price, atr, positions, now=now
        )

        if not decision.approved:
            # A neutral signal is a normal no-op; a real veto is worth flagging.
            action = "skipped" if not signal.is_actionable else "vetoed"
            return ExecutionResult(action=action, message=decision.reason, decision=decision)

        try:
            closed = self._flip_if_needed(decision)
            entry = self._open_entry(decision)
        except AlpacaClientError as exc:
            logger.error("Execution failed for %s: %s", decision.target_symbol, exc)
            return ExecutionResult(action="error", message=str(exc), decision=decision)

        if wait_for_fill:
            self._wait_for_fill(entry)

        action = "flipped" if closed else "entered"
        message = str(decision)
        logger.info("Execution %s: %s", action, message)
        return ExecutionResult(
            action=action,
            message=message,
            decision=decision,
            entry_order=entry,
            closed_orders=closed,
        )

    # ------------------------------------------------------------------
    # Flipping & opening
    # ------------------------------------------------------------------
    def _flip_if_needed(self, decision: RiskDecision) -> list[ManagedOrder]:
        """Close any opposite leg named in the decision and wait until flat.

        Returns the liquidation orders submitted (empty when no flip was
        required). Raises :class:`AlpacaClientError` if a leg won't go flat.
        """
        closed: list[ManagedOrder] = []
        for symbol in decision.close_symbols:
            order = self._close_symbol(symbol, reason="flip — opposite leg of new entry")
            if order is not None:
                closed.append(order)
            self._wait_until_flat(symbol)
        return closed

    def _open_entry(self, decision: RiskDecision) -> ManagedOrder:
        """Submit the entry for an approved decision and track it.

        The entry is a market buy with a full ATR bracket attached: a resting
        take-profit limit and a stop-loss, as OCO children. Whichever leg fills
        first closes the position and cancels the other, so a winner is taken at
        the +N*ATR target without waiting on a poll. The orchestration loop can
        still close early on a signal flip, forced exit, or end-of-day flat.
        """
        client_order_id = self._new_client_order_id("entry", decision.target_symbol or "")
        order = self._client.submit_bracket_order(
            symbol=decision.target_symbol,  # type: ignore[arg-type]
            qty=decision.qty,
            side=OrderSide.BUY,  # both legs are entered long (we buy the favored ETF)
            take_profit_price=decision.take_profit,
            stop_loss_price=decision.stop_loss,
            client_order_id=client_order_id,
        )
        managed = self._track(order, kind="entry", reason=decision.reason)
        return managed

    def _close_symbol(self, symbol: str, *, reason: str) -> ManagedOrder | None:
        """Liquidate the whole position in ``symbol`` at market and track it."""
        order = self._client.close_position(symbol)
        if order is None:
            logger.info("Nothing to close in %s (already flat)", symbol)
            return None
        return self._track(order, kind="exit", reason=reason)

    # ------------------------------------------------------------------
    # Forced / requested exits
    # ------------------------------------------------------------------
    def handle_forced_exits(
        self,
        equity: float,
        positions: Mapping[str, object | None] | None = None,
        *,
        now: datetime | None = None,
    ) -> list[ManagedOrder]:
        """Ask the risk manager what must be flattened *now* and do it.

        Covers the invariant breach (both legs held) and the daily-loss stop.
        Returns the liquidation orders submitted.
        """
        if positions is None:
            positions = self._fetch_positions()
        exits: list[ForcedExit] = self._risk.forced_exits(equity, positions, now=now)
        orders: list[ManagedOrder] = []
        for forced in exits:
            order = self._close_symbol(forced.symbol, reason=forced.reason)
            if order is not None:
                orders.append(order)
        return orders

    def close_all(self, *, reason: str = "flatten requested") -> list[ManagedOrder]:
        """Close both legs of the pair (e.g. for end-of-day flat).

        Cancels resting child orders first so the liquidation isn't fought by a
        stale stop leg, then market-closes each open symbol.
        """
        self._client.cancel_all_orders()
        orders: list[ManagedOrder] = []
        for symbol in self._symbols:
            order = self._close_symbol(symbol, reason=reason)
            if order is not None:
                orders.append(order)
                self._wait_until_flat(symbol)
        if not orders:
            logger.info("close_all: already flat in both legs")
        return orders

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------
    def reconcile(self) -> list[ManagedOrder]:
        """Refresh every tracked order from Alpaca and settle bookkeeping.

        For each non-terminal order: fetch its latest state, update the local
        record, and — the first time an *entry* is confirmed filled — call
        :meth:`RiskManager.register_entry`. Terminal orders are dropped from the
        active set. Returns the orders that changed status this pass.
        """
        changed: list[ManagedOrder] = []
        for order_id, managed in list(self._open_orders.items()):
            # Re-poll anything not yet final; an order can already be terminal
            # here if it filled instantly in its submit response, in which case
            # we still fall through to the registration/cleanup below.
            if not managed.is_terminal:
                try:
                    latest = self._client.get_order(order_id)
                except AlpacaClientError as exc:
                    logger.warning("Reconcile: could not fetch order %s: %s", order_id, exc)
                    continue

                prev_status = managed.status
                managed.update_from(latest)
                if managed.status != prev_status:
                    changed.append(managed)
                    logger.info(
                        "Order %s (%s %s) %s -> %s (filled %s/%s)",
                        order_id, managed.side, managed.symbol,
                        prev_status, managed.status, managed.filled_qty, managed.qty,
                    )

            # Count an entry against the daily cap exactly once, on first fill.
            if managed.kind == "entry" and managed.is_filled and not managed.registered:
                self._risk.register_entry()
                managed.registered = True
                logger.info(
                    "Entry filled: %s x%s @ %.4f — registered with risk manager",
                    managed.symbol, managed.filled_qty, managed.filled_avg_price,
                )

            if managed.is_terminal:
                self._open_orders.pop(order_id, None)

        return changed

    # ------------------------------------------------------------------
    # Waiting helpers
    # ------------------------------------------------------------------
    def _wait_for_fill(self, managed: ManagedOrder) -> bool:
        """Poll until the entry fills (or ``fill_timeout`` elapses).

        Drives :meth:`reconcile` so a fill observed here also registers the
        entry. Returns True if the order filled within the window.
        """
        deadline = time.monotonic() + self._fill_timeout
        while time.monotonic() < deadline:
            self.reconcile()
            if managed.is_filled:
                return True
            if managed.is_terminal:
                logger.warning(
                    "Entry %s reached terminal status %s without filling",
                    managed.order_id, managed.status,
                )
                return False
            time.sleep(self._poll_interval)
        logger.warning(
            "Entry %s not filled within %.0fs — will reconcile next cycle",
            managed.order_id, self._fill_timeout,
        )
        return False

    def _wait_until_flat(self, symbol: str) -> None:
        """Block until ``symbol`` has no open position, or raise on timeout.

        Used on a flip so the new bracket isn't opened while the opposite leg is
        still being liquidated.
        """
        deadline = time.monotonic() + self._flip_timeout
        while time.monotonic() < deadline:
            if not self._client.has_position(symbol):
                logger.info("%s is flat", symbol)
                return
            time.sleep(self._poll_interval)
        raise AlpacaClientError(
            f"{symbol} did not go flat within {self._flip_timeout:.0f}s; "
            f"aborting to avoid holding both legs"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _track(self, order: Order, *, kind: str, reason: str) -> ManagedOrder:
        """Wrap a submitted Alpaca order in a :class:`ManagedOrder` and store it."""
        managed = ManagedOrder(
            order_id=str(order.id),
            symbol=getattr(order, "symbol", "") or "",
            side=getattr(getattr(order, "side", None), "value", "") or "",
            qty=_to_float(getattr(order, "qty", None)),
            kind=kind,
            client_order_id=getattr(order, "client_order_id", None),
            reason=reason,
        )
        managed.update_from(order)
        self._open_orders[managed.order_id] = managed
        return managed

    def _fetch_positions(self) -> dict[str, object | None]:
        return self._client.get_pair_positions(self._symbol_long, self._symbol_short)

    @staticmethod
    def _new_client_order_id(kind: str, symbol: str) -> str:
        """A unique, traceable client order id (idempotency + log correlation)."""
        return f"{kind}-{symbol}-{uuid.uuid4().hex[:12]}"
