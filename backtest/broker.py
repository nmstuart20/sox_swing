"""A simulated Alpaca broker for backtesting.

:class:`SimBroker` is a drop-in stand-in for
:class:`~execution.alpaca_client.AlpacaClient`: it exposes the same methods the
:class:`~execution.order_manager.OrderManager` calls (``submit_stop_entry_order``,
``close_position``, ``get_order``, ``reconcile`` helpers, ``get_pair_positions``,
``has_position``, ``cancel_all_orders``, ``get_equity``, ``get_clock``), so the
*real* order manager runs against it unchanged.

What it simulates that the production fakes don't:

  * **Fills with slippage and commission.** A market buy fills at the bar price
    nudged up by ``slippage_bps``; a sell is nudged down. Commission is charged
    per share and/or as a fraction of notional, with a per-order floor.
  * **The bracket legs.** ``submit_bracket_order`` carries an attached
    take-profit and stop (exactly as live). The broker remembers both and, on
    each subsequent bar, triggers a stop exit when the bar's low pierces the
    stop (a market fill that models gap-throughs and slippage) or a take-profit
    exit when the bar's high reaches the target (a limit fill at the target, or
    a gapped-up open, with no adverse slippage). If a bar spans both, the stop
    is assumed to fire first.
  * **Mark-to-market equity.** Each bar updates the marks and records an
    ``(timestamp, equity)`` point, so drawdown and Sharpe come from the same
    curve the loop traded on.

Every realized round-trip is recorded as a :class:`Trade` for the trade log.
The broker holds no strategy or risk logic — it is a pure execution/accounting
venue, just like the real Alpaca it replaces.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from config.logging_setup import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Cost model
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CostModel:
    """Slippage and commission applied to every simulated fill.

    Args:
        slippage_bps: half-spread/impact in basis points of price. A buy fills
            at ``price * (1 + slippage_bps/1e4)``; a sell at the mirror. The
            SOXL/SOXS pair is liquid, so a couple of bps is realistic for the
            small clips this bot trades.
        commission_per_share: dollars per share (US equities on Alpaca are
            commission-free, so this defaults to 0; set it to stress-test or to
            model an options/other venue).
        commission_pct: commission as a fraction of notional (e.g. 0.0005).
        commission_min: per-order floor once any commission is charged.
    """

    slippage_bps: float = 2.0
    commission_per_share: float = 0.0
    commission_pct: float = 0.0
    commission_min: float = 0.0

    def buy_price(self, price: float) -> float:
        return price * (1.0 + self.slippage_bps / 1e4)

    def sell_price(self, price: float) -> float:
        return price * (1.0 - self.slippage_bps / 1e4)

    def commission(self, qty: float, fill_price: float) -> float:
        per_share = abs(qty) * self.commission_per_share
        pct = abs(qty) * fill_price * self.commission_pct
        fee = per_share + pct
        if fee <= 0:
            return 0.0
        return max(fee, self.commission_min)


# ----------------------------------------------------------------------
# Order / position / trade records
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _Side:
    """Mirrors alpaca's ``OrderSide`` enum — only ``.value`` is read downstream."""

    value: str


class SimOrder:
    """A simulated order, shaped like the fields the order manager reads.

    Backtest fills are synchronous, so an order is created already ``filled``
    (or rejected with zero qty), mirroring a quiet paper fill.
    """

    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        *,
        filled_avg_price: float,
        status: str = "filled",
        client_order_id: str | None = None,
    ) -> None:
        self.id = order_id
        self.symbol = symbol
        self.side = _Side(side)
        self.qty = str(qty)
        self.status = status
        self.filled_qty = str(qty if status == "filled" else 0.0)
        self.filled_avg_price = str(filled_avg_price)
        self.client_order_id = client_order_id


@dataclass
class SimPosition:
    """An open long position, exposing the fields the risk manager reads."""

    symbol: str
    qty: float
    entry_price: float          # actual fill price (slippage included)
    entry_time: datetime
    entry_commission: float
    entry_index: int            # bar index at entry, for bars-held bookkeeping
    stop_price: float           # bracket stop (sell-stop below entry)
    take_profit: float          # bracket target (sell-limit above entry; 0 = none)
    mark: float                 # latest close
    trail_distance: float = 0.0  # if > 0, stop ratchets to keep this far below the high

    @property
    def unrealized_pl(self) -> float:
        return (self.mark - self.entry_price) * self.qty

    @property
    def market_value(self) -> float:
        return self.mark * self.qty


@dataclass(frozen=True)
class Trade:
    """A completed round-trip (entry → exit), the unit of the trade log."""

    symbol: str
    qty: float
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str
    gross_pnl: float        # (exit - entry) * qty, before costs
    commission: float       # entry + exit commission
    net_pnl: float          # gross - commission
    return_pct: float       # net_pnl / cost basis
    bars_held: int

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": round(self.entry_price, 4),
            "exit_time": self.exit_time.isoformat(),
            "exit_price": round(self.exit_price, 4),
            "exit_reason": self.exit_reason,
            "gross_pnl": round(self.gross_pnl, 2),
            "commission": round(self.commission, 4),
            "net_pnl": round(self.net_pnl, 2),
            "return_pct": round(self.return_pct, 5),
            "bars_held": self.bars_held,
        }


# ----------------------------------------------------------------------
# The simulated clock
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SimClock:
    """A minimal stand-in for Alpaca's market ``Clock``."""

    is_open: bool
    next_open: datetime
    next_close: datetime


# ----------------------------------------------------------------------
# The broker
# ----------------------------------------------------------------------
class SimBroker:
    """An in-memory execution venue with slippage, commission, and stops.

    Args:
        initial_capital: starting cash (and the equity baseline for returns).
        cost_model: slippage/commission applied to every fill.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        cost_model: CostModel | None = None,
        *,
        trailing_stop: bool = False,
    ) -> None:
        self._initial = float(initial_capital)
        self._cash = float(initial_capital)
        self._costs = cost_model or CostModel()
        # When True, an open position's stop ratchets up to stay a fixed distance
        # (its initial stop distance) below the highest bar high seen — a
        # chandelier trail that locks in favorable moves.
        self._trailing = bool(trailing_stop)

        self._positions: dict[str, SimPosition] = {}
        self._orders: dict[str, SimOrder] = {}
        self._ids = itertools.count(1)
        # Current price per tradable symbol, pushed by the driver each bar so we
        # can fill a buy in a symbol we don't yet hold.
        self._mark_map: dict[str, float] = {}

        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.total_commission: float = 0.0
        # Diagnostic: the most legs ever held at once. The pair invariant means
        # this must stay <= 1; it's a cheap guard against an execution bug.
        self.max_concurrent: int = 0

        # The reason tagged onto the next order-manager-initiated close. The
        # order manager calls ``close_position(symbol)`` without a reason, so the
        # driver sets this before each phase (flip / forced-exit / eod-flat).
        self.default_close_reason: str = "close"

        # Driver-controlled view of "now" and the market clock.
        self._now: datetime = datetime.now(timezone.utc)
        self._clock = SimClock(
            is_open=True,
            next_open=self._now,
            next_close=self._now + timedelta(hours=6),
        )
        self._bar_index = 0

        logger.info(
            "SimBroker initialized (capital=%.2f, slippage=%.1fbps, comm/share=%.4f)",
            self._initial, self._costs.slippage_bps, self._costs.commission_per_share,
        )

    # ------------------------------------------------------------------
    # Driver hooks (used by the backtest engine, not by the order manager)
    # ------------------------------------------------------------------
    def process_bar(
        self,
        timestamp: datetime,
        ohlc: dict[str, dict[str, float]],
        bar_index: int,
        clock: SimClock,
    ) -> None:
        """Advance the broker one bar: trigger stops, re-mark, record equity.

        Args:
            timestamp: the bar's UTC timestamp (the simulated "now").
            ohlc: ``{symbol: {open, high, low, close}}`` for this bar.
            bar_index: monotonic bar counter (for bars-held bookkeeping).
            clock: the market clock to report for this bar.
        """
        self._now = timestamp
        self._bar_index = bar_index
        self._clock = clock

        # 1. Bracket exits first: a position held into this bar can hit its stop
        #    or take-profit before any new decision is made at the close. When a
        #    bar's range spans both levels we can't know the intrabar path, so we
        #    assume the worse outcome (stop) fires — the standard conservative
        #    convention, avoiding an optimistic bias in the results.
        for symbol in list(self._positions):
            bar = ohlc.get(symbol)
            if bar is None:
                continue
            pos = self._positions[symbol]
            if bar["low"] <= pos.stop_price:
                # Stop is a market order: gap-through fills at the open if the bar
                # opened below the stop (worse of the two), then slippage applies.
                trigger = min(pos.stop_price, bar["open"])
                self._exit(symbol, self._costs.sell_price(trigger), reason="stop-loss")
            elif pos.take_profit > 0 and bar["high"] >= pos.take_profit:
                # Take-profit is a resting limit: it fills at the target, or at a
                # gapped-up open (better), with no adverse slippage.
                trigger = max(pos.take_profit, bar["open"])
                self._exit(symbol, trigger, reason="take-profit")

        # 2. Re-mark survivors to this bar's close and ratchet any trailing stop.
        #    The trail uses *this* bar's high but only tightens the stop for the
        #    next bar onward (it never fires against the same bar's low above),
        #    so there's no intrabar lookahead. The stop only ever moves up.
        for symbol, pos in self._positions.items():
            bar = ohlc.get(symbol)
            if bar is not None:
                pos.mark = bar["close"]
                if pos.trail_distance > 0:
                    pos.stop_price = max(pos.stop_price, bar["high"] - pos.trail_distance)

        # 3. Record the equity point for the curve.
        self.max_concurrent = max(self.max_concurrent, len(self._positions))
        self.equity_curve.append((timestamp, self.get_equity()))

    def force_close_all(self, reason: str = "backtest-end") -> None:
        """Liquidate every open position at its current mark (end of replay)."""
        for symbol in list(self._positions):
            fill = self._costs.sell_price(self._positions[symbol].mark)
            self._exit(symbol, fill, reason=reason)

    @property
    def realized_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    # ------------------------------------------------------------------
    # AlpacaClient-compatible surface
    # ------------------------------------------------------------------
    @property
    def is_paper(self) -> bool:
        return True

    def get_equity(self) -> float:
        """Mark-to-market equity: cash plus the value of open positions."""
        return self._cash + sum(p.market_value for p in self._positions.values())

    def get_clock(self) -> SimClock:
        return self._clock

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_position(self, symbol: str) -> SimPosition | None:
        return self._positions.get(symbol)

    def get_pair_positions(
        self, symbol_long: str, symbol_short: str
    ) -> dict[str, SimPosition | None]:
        return {
            symbol_long: self._positions.get(symbol_long),
            symbol_short: self._positions.get(symbol_short),
        }

    def get_order(self, order_id: str) -> SimOrder:
        return self._orders[order_id]

    def cancel_all_orders(self) -> None:
        # Backtest fills are synchronous, so there are never resting child
        # orders to cancel; the bracket legs live on the position, not as orders.
        return None

    def cancel_orders_for_symbol(self, symbol: str) -> int:
        # As with cancel_all_orders: bracket legs live on the position here, so
        # there are no resting orders to clear before a close. No-op for parity
        # with the live AlpacaClient interface.
        return 0

    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side,
        take_profit_price: float,
        stop_loss_price: float,
        client_order_id: str | None = None,
        **_: object,
    ) -> SimOrder:
        """Market-buy ``qty`` of ``symbol`` with an attached take-profit and stop.

        Mirrors :meth:`AlpacaClient.submit_bracket_order`: the entry fills at
        market (here, the current mark) and both bracket legs rest on the
        position until a later bar triggers one of them (see :meth:`process_bar`).
        """
        return self._open(
            symbol, qty, side, stop_loss_price, take_profit_price, client_order_id
        )

    def submit_stop_entry_order(
        self,
        symbol: str,
        qty: float,
        side,
        stop_loss_price: float,
        client_order_id: str | None = None,
        **_: object,
    ) -> SimOrder:
        """Market-buy with a stop only (no take-profit), like the OTO variant."""
        return self._open(symbol, qty, side, stop_loss_price, 0.0, client_order_id)

    def _open(
        self,
        symbol: str,
        qty: float,
        side,
        stop_loss_price: float,
        take_profit_price: float,
        client_order_id: str | None,
    ) -> SimOrder:
        """Open a long position at market and attach its bracket leg(s)."""
        side_value = getattr(side, "value", str(side))
        qty = float(qty)
        if qty <= 0:
            return self._rejected(symbol, side_value, client_order_id)

        fill_price = self._costs.buy_price(self._mark_or_raise(symbol))
        commission = self._costs.commission(qty, fill_price)
        self._cash -= qty * fill_price + commission
        self.total_commission += commission

        # The trail rides at the bracket's own initial stop distance below the
        # high-water mark; 0 leaves the stop fixed (trailing disabled).
        trail_distance = max(fill_price - float(stop_loss_price), 0.0) if self._trailing else 0.0
        self._positions[symbol] = SimPosition(
            symbol=symbol,
            qty=qty,
            entry_price=fill_price,
            entry_time=self._now,
            entry_commission=commission,
            entry_index=self._bar_index,
            stop_price=float(stop_loss_price),
            take_profit=float(take_profit_price),
            mark=self._mark_or_raise(symbol),
            trail_distance=trail_distance,
        )
        order = self._make_order(symbol, side_value, qty, fill_price, client_order_id)
        logger.info(
            "SIM entry: BUY %s x%.0f @ %.4f (tp=%.4f, stop=%.4f, comm=%.4f)",
            symbol, qty, fill_price, take_profit_price, stop_loss_price, commission,
        )
        return order

    def close_position(self, symbol: str, reason: str | None = None) -> SimOrder | None:
        """Liquidate the whole position in ``symbol`` at market, or None if flat."""
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        # A close is a market order, so it pays slippage.
        exit_price = self._costs.sell_price(pos.mark)
        trade = self._exit(symbol, exit_price, reason=reason or self.default_close_reason)
        return self._make_order(symbol, "sell", trade.qty, exit_price, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _exit(self, symbol: str, fill_price: float, *, reason: str) -> Trade:
        """Close a position at ``fill_price`` (the actual fill) and record the trade.

        The caller decides the fill: a stop-loss or a market close passes a
        slippage-adjusted price, while a take-profit limit passes the target
        price itself (a resting limit fills at its price or better, not worse).
        """
        pos = self._positions.pop(symbol)
        exit_price = float(fill_price)
        commission = self._costs.commission(pos.qty, exit_price)
        self._cash += pos.qty * exit_price - commission
        self.total_commission += commission

        gross = (exit_price - pos.entry_price) * pos.qty
        total_comm = pos.entry_commission + commission
        cost_basis = pos.entry_price * pos.qty
        trade = Trade(
            symbol=symbol,
            qty=pos.qty,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=self._now,
            exit_price=exit_price,
            exit_reason=reason,
            gross_pnl=gross,
            commission=total_comm,
            net_pnl=gross - total_comm,
            return_pct=(gross - total_comm) / cost_basis if cost_basis else 0.0,
            bars_held=max(self._bar_index - pos.entry_index, 0),
        )
        self.trades.append(trade)
        logger.info(
            "SIM exit (%s): SELL %s x%.0f @ %.4f, net P&L %.2f",
            reason, symbol, pos.qty, exit_price, trade.net_pnl,
        )
        return trade

    def _mark_or_raise(self, symbol: str) -> float:
        mark = self._mark_map.get(symbol)
        if mark is None:
            raise KeyError(f"no mark for {symbol}; process_bar must run before trading")
        return mark

    def set_marks(self, marks: dict[str, float]) -> None:
        """Driver pushes the current price for every tradable symbol each bar."""
        self._mark_map = dict(marks)

    def _make_order(
        self, symbol: str, side: str, qty: float, price: float, client_order_id: str | None
    ) -> SimOrder:
        order = SimOrder(
            str(next(self._ids)), symbol, side, qty,
            filled_avg_price=price, status="filled", client_order_id=client_order_id,
        )
        self._orders[order.id] = order
        return order

    def _rejected(self, symbol: str, side: str, client_order_id: str | None) -> SimOrder:
        order = SimOrder(
            str(next(self._ids)), symbol, side, 0.0,
            filled_avg_price=0.0, status="rejected", client_order_id=client_order_id,
        )
        self._orders[order.id] = order
        return order
