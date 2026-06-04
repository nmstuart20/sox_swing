"""Alpaca connection layer.

A thin, defensive wrapper around alpaca-py's ``TradingClient`` that:
  * connects using paper or live keys from config,
  * fetches account info / buying power,
  * inspects current SOXL/SOXS positions,
  * submits market, limit, and bracket orders (shares),
  * submits single-leg options orders,
  * retries transient failures and surfaces clear errors.

The trading-loop modules (risk, order manager, main) talk to Alpaca only
through this class so the rest of the codebase stays decoupled from the SDK.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, TypeVar

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.models import Clock, Order, Position, TradeAccount
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from config.logging_setup import get_logger
from config.settings import AlpacaConfig

logger = get_logger(__name__)

T = TypeVar("T")

# alpaca-py raises APIError; import lazily so the module loads even if the
# SDK's internal layout shifts between versions.
try:  # pragma: no cover - import guard
    from alpaca.common.exceptions import APIError
except Exception:  # pragma: no cover
    class APIError(Exception):  # type: ignore[no-redef]
        """Fallback if alpaca's APIError import path changes."""


class AlpacaClientError(Exception):
    """Raised when an Alpaca operation fails after retries."""


def _with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a method on transient errors with exponential backoff.

    Client errors (HTTP 4xx other than 429) are treated as fatal and not
    retried, since retrying a bad request won't help.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = base_delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except APIError as exc:
                    status = _status_code(exc)
                    fatal = status is not None and 400 <= status < 500 and status != 429
                    last_exc = exc
                    if fatal:
                        logger.error("%s failed (HTTP %s): %s", func.__name__, status, exc)
                        raise AlpacaClientError(f"{func.__name__} failed: {exc}") from exc
                    logger.warning(
                        "%s transient error (attempt %d/%d): %s",
                        func.__name__, attempt, max_attempts, exc,
                    )
                except (ConnectionError, TimeoutError, OSError) as exc:
                    last_exc = exc
                    logger.warning(
                        "%s network error (attempt %d/%d): %s",
                        func.__name__, attempt, max_attempts, exc,
                    )
                if attempt < max_attempts:
                    time.sleep(delay)
                    delay *= backoff
            raise AlpacaClientError(
                f"{func.__name__} failed after {max_attempts} attempts: {last_exc}"
            ) from last_exc

        return wrapper

    return decorator


def _status_code(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status code from an APIError."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


class AlpacaClient:
    """High-level Alpaca trading client scoped to this bot's needs."""

    def __init__(self, config: AlpacaConfig) -> None:
        self._config = config
        self._client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )
        logger.info(
            "AlpacaClient initialized (mode=%s)",
            "PAPER" if config.paper else "LIVE",
        )

    @property
    def is_paper(self) -> bool:
        return self._config.paper

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------
    @_with_retry()
    def get_account(self) -> TradeAccount:
        """Return the full Alpaca account object."""
        return self._client.get_account()

    def get_buying_power(self) -> float:
        """Return non-marginable buying power (cash available to trade)."""
        account = self.get_account()
        return float(account.buying_power)

    def get_equity(self) -> float:
        """Return total account equity."""
        account = self.get_account()
        return float(account.equity)

    @_with_retry()
    def get_clock(self) -> Clock:
        """Return Alpaca's market clock (``is_open``, ``next_open``, ``next_close``).

        The orchestration loop uses this as the single source of truth for
        market hours so it never trades — or holds into close — outside session.
        """
        return self._client.get_clock()

    def is_market_open(self) -> bool:
        return bool(self.get_clock().is_open)

    @_with_retry()
    def get_all_positions(self) -> list[Position]:
        return list(self._client.get_all_positions())

    def get_position(self, symbol: str) -> Position | None:
        """Return the open position for ``symbol``, or None if flat.

        A 404 from Alpaca means no open position, which is not an error here.
        """
        try:
            return self._client.get_open_position(symbol)
        except APIError as exc:
            if _status_code(exc) == 404:
                return None
            raise AlpacaClientError(f"get_position({symbol}) failed: {exc}") from exc

    def has_position(self, symbol: str) -> bool:
        return self.get_position(symbol) is not None

    def get_pair_positions(self, symbol_long: str, symbol_short: str) -> dict[str, Position | None]:
        """Convenience: current positions for the SOXL/SOXS pair."""
        return {
            symbol_long: self.get_position(symbol_long),
            symbol_short: self.get_position(symbol_short),
        }

    # ------------------------------------------------------------------
    # Orders — shares
    # ------------------------------------------------------------------
    @_with_retry()
    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> Order:
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )
        logger.info("Submitting MARKET %s %s x%s", side.value, symbol, qty)
        return self._client.submit_order(request)

    @_with_retry()
    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> Order:
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            limit_price=round(float(limit_price), 2),
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )
        logger.info(
            "Submitting LIMIT %s %s x%s @ %.2f", side.value, symbol, qty, limit_price
        )
        return self._client.submit_order(request)

    @_with_retry()
    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        take_profit_price: float,
        stop_loss_price: float,
        stop_limit_price: float | None = None,
        limit_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit a bracket order (entry + take-profit + stop-loss).

        If ``limit_price`` is given the entry is a limit order, otherwise it's
        a market entry. ``stop_limit_price`` makes the stop a stop-limit.
        """
        take_profit = TakeProfitRequest(limit_price=round(float(take_profit_price), 2))
        stop_loss = StopLossRequest(
            stop_price=round(float(stop_loss_price), 2),
            limit_price=round(float(stop_limit_price), 2) if stop_limit_price else None,
        )
        common = dict(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=time_in_force,
            order_class=OrderClass.BRACKET,
            take_profit=take_profit,
            stop_loss=stop_loss,
            client_order_id=client_order_id,
        )
        if limit_price is not None:
            request: MarketOrderRequest | LimitOrderRequest = LimitOrderRequest(
                limit_price=round(float(limit_price), 2), **common
            )
        else:
            request = MarketOrderRequest(**common)
        logger.info(
            "Submitting BRACKET %s %s x%s (tp=%.2f sl=%.2f)",
            side.value, symbol, qty, take_profit_price, stop_loss_price,
        )
        return self._client.submit_order(request)

    @_with_retry()
    def submit_stop_entry_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        stop_loss_price: float,
        stop_limit_price: float | None = None,
        limit_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit a one-triggers-other entry: an entry with an attached stop,
        but *no* take-profit leg.

        Unlike a bracket there is no resting take-profit limit on the sell side,
        so profit-taking is left to the orchestration loop as a market close.
        The entry is a market order unless ``limit_price`` is given; the stop is
        a plain stop (market on trigger) unless ``stop_limit_price`` makes it a
        stop-limit.
        """
        stop_loss = StopLossRequest(
            stop_price=round(float(stop_loss_price), 2),
            limit_price=round(float(stop_limit_price), 2) if stop_limit_price else None,
        )
        common = dict(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=time_in_force,
            order_class=OrderClass.OTO,
            stop_loss=stop_loss,
            client_order_id=client_order_id,
        )
        if limit_price is not None:
            request: MarketOrderRequest | LimitOrderRequest = LimitOrderRequest(
                limit_price=round(float(limit_price), 2), **common
            )
        else:
            request = MarketOrderRequest(**common)
        logger.info(
            "Submitting OTO %s %s x%s (sl=%.2f, no take-profit)",
            side.value, symbol, qty, stop_loss_price,
        )
        return self._client.submit_order(request)

    # ------------------------------------------------------------------
    # Orders — options (single leg)
    # ------------------------------------------------------------------
    @_with_retry()
    def submit_option_order(
        self,
        option_symbol: str,
        qty: int,
        side: OrderSide,
        order_type: str = "market",
        limit_price: float | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit a single-leg option order.

        ``option_symbol`` is an OCC contract symbol (e.g. ``SOXL250620C00030000``).
        Options are forced to DAY time-in-force, as Alpaca requires.
        """
        if order_type == "limit":
            if limit_price is None:
                raise AlpacaClientError("limit option order requires limit_price")
            request: MarketOrderRequest | LimitOrderRequest = LimitOrderRequest(
                symbol=option_symbol,
                qty=qty,
                side=side,
                limit_price=round(float(limit_price), 2),
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )
        else:
            request = MarketOrderRequest(
                symbol=option_symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )
        logger.info(
            "Submitting OPTION %s %s %s x%s", order_type.upper(), side.value, option_symbol, qty
        )
        return self._client.submit_order(request)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------
    @_with_retry()
    def get_order(self, order_id: str) -> Order:
        return self._client.get_order_by_id(order_id)

    @_with_retry()
    def cancel_order(self, order_id: str) -> None:
        self._client.cancel_order_by_id(order_id)
        logger.info("Cancelled order %s", order_id)

    @_with_retry()
    def cancel_all_orders(self) -> None:
        self._client.cancel_orders()
        logger.info("Cancelled all open orders")

    @_with_retry()
    def close_position(self, symbol: str) -> Order | None:
        """Close (liquidate) the entire position in ``symbol`` at market."""
        if not self.has_position(symbol):
            logger.info("No open position in %s to close", symbol)
            return None
        order = self._client.close_position(symbol)
        logger.info("Closing position in %s", symbol)
        return order

    @_with_retry()
    def close_all_positions(self, cancel_orders: bool = True) -> None:
        self._client.close_all_positions(cancel_orders=cancel_orders)
        logger.info("Closing all positions (cancel_orders=%s)", cancel_orders)
