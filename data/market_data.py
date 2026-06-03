"""Market data ingestion for SOXL/SOXS.

Pulls historical and real-time OHLCV bars from Alpaca's market-data API and
returns clean, timestamp-indexed pandas DataFrames. Responsibilities:

  * fetch 1-min, 5-min, and daily bars for any symbol (defaults to the
    SOXL/SOXS pair the bot trades),
  * normalize Alpaca's multi-index ``BarSet`` into a tidy per-symbol frame,
  * sort, de-duplicate, and optionally fill intraday gaps,
  * cache recently fetched bars so repeated calls within a poll cycle don't
    hammer the API,
  * expose the latest quote and a single "current price" for sizing/risk.

Only the market-data REST clients live here; order placement and account
state are handled by ``execution/alpaca_client.py``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

import pandas as pd

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config.logging_setup import get_logger
from config.settings import AlpacaConfig

logger = get_logger(__name__)

T = TypeVar("T")

# Canonical OHLCV column order for every frame this module returns.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]

try:  # pragma: no cover - import guard, mirrors execution/alpaca_client.py
    from alpaca.common.exceptions import APIError
except Exception:  # pragma: no cover
    class APIError(Exception):  # type: ignore[no-redef]
        """Fallback if alpaca's APIError import path changes."""


class MarketDataError(Exception):
    """Raised when a market-data fetch fails after retries."""


class Timeframe(Enum):
    """The bar resolutions the bot works with."""

    MINUTE = "1min"
    FIVE_MINUTE = "5min"
    DAILY = "1day"

    @property
    def alpaca_timeframe(self) -> TimeFrame:
        if self is Timeframe.MINUTE:
            return TimeFrame(1, TimeFrameUnit.Minute)
        if self is Timeframe.FIVE_MINUTE:
            return TimeFrame(5, TimeFrameUnit.Minute)
        return TimeFrame(1, TimeFrameUnit.Day)

    @property
    def pandas_freq(self) -> str:
        """Pandas offset alias used when reindexing to fill gaps."""
        return {
            Timeframe.MINUTE: "1min",
            Timeframe.FIVE_MINUTE: "5min",
            Timeframe.DAILY: "1D",
        }[self]

    @property
    def default_lookback(self) -> timedelta:
        """How far back to pull when the caller doesn't specify a window."""
        return {
            Timeframe.MINUTE: timedelta(days=2),
            Timeframe.FIVE_MINUTE: timedelta(days=5),
            Timeframe.DAILY: timedelta(days=365),
        }[self]


@dataclass(frozen=True)
class Quote:
    """A simplified latest quote for a symbol."""

    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float

    @property
    def mid_price(self) -> float | None:
        """Midpoint of bid/ask, or whichever side is present."""
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2.0
        return self.ask_price or self.bid_price or None

    @property
    def spread(self) -> float | None:
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return None


def _with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a method on transient errors with exponential backoff.

    Client errors (HTTP 4xx other than 429) are fatal and not retried.
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
                        raise MarketDataError(f"{func.__name__} failed: {exc}") from exc
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
            raise MarketDataError(
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


@dataclass
class _CacheEntry:
    frame: pd.DataFrame
    fetched_at: float


class MarketData:
    """Fetches and caches Alpaca OHLCV bars and quotes for the bot.

    The market-data API does not require paper/live distinction, but it does
    require a data feed. The free Alpaca plan only serves the ``IEX`` feed;
    paid plans can use ``SIP`` for full-market consolidated data.
    """

    def __init__(
        self,
        config: AlpacaConfig,
        feed: DataFeed = DataFeed.IEX,
        cache_ttl_seconds: float = 30.0,
    ) -> None:
        self._client = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )
        self._feed = feed
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[tuple[str, Timeframe], _CacheEntry] = {}
        self._lock = threading.Lock()
        logger.info("MarketData initialized (feed=%s, cache_ttl=%.0fs)", feed.value, cache_ttl_seconds)

    # ------------------------------------------------------------------
    # Bars
    # ------------------------------------------------------------------
    def get_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        fill_gaps: bool = False,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return OHLCV bars for ``symbol`` at ``timeframe``.

        The result is indexed by tz-aware (UTC) timestamp with columns
        :data:`OHLCV_COLUMNS`, sorted ascending with duplicates removed.

        Args:
            start/end: explicit UTC window. If ``start`` is omitted a sensible
                lookback for the timeframe is used; ``end`` defaults to now.
            limit: cap on number of bars returned (most recent kept).
            fill_gaps: reindex intraday bars to a continuous grid and
                forward-fill prices (volume filled with 0). Daily frames are
                left untouched since calendar gaps (weekends/holidays) are
                expected.
            use_cache: when True (and no explicit window/limit is given),
                serve a recent cached frame if still within the TTL.
        """
        cacheable = use_cache and start is None and end is None and limit is None
        if cacheable:
            cached = self._read_cache(symbol, timeframe)
            if cached is not None:
                return cached.copy()

        end = end or datetime.now(timezone.utc)
        start = start or (end - timeframe.default_lookback)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe.alpaca_timeframe,
            start=start,
            end=end,
            limit=limit,
            feed=self._feed,
        )
        barset = self._fetch_bars(request)
        frame = self._to_frame(barset, symbol)

        if limit is not None and len(frame) > limit:
            frame = frame.iloc[-limit:]
        if fill_gaps and timeframe is not Timeframe.DAILY:
            frame = self._fill_gaps(frame, timeframe)

        if cacheable:
            self._write_cache(symbol, timeframe, frame)
        return frame

    def get_minute_bars(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        return self.get_bars(symbol, Timeframe.MINUTE, **kwargs)

    def get_five_minute_bars(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        return self.get_bars(symbol, Timeframe.FIVE_MINUTE, **kwargs)

    def get_daily_bars(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        return self.get_bars(symbol, Timeframe.DAILY, **kwargs)

    @_with_retry()
    def _fetch_bars(self, request: StockBarsRequest) -> Any:
        return self._client.get_stock_bars(request)

    def _to_frame(self, barset: Any, symbol: str) -> pd.DataFrame:
        """Convert Alpaca's BarSet into a clean single-symbol OHLCV frame."""
        df = barset.df
        if df is None or df.empty:
            logger.warning("No bars returned for %s", symbol)
            return self._empty_frame()

        # BarSet.df is multi-indexed by (symbol, timestamp); select our symbol.
        if isinstance(df.index, pd.MultiIndex):
            if symbol not in df.index.get_level_values(0):
                return self._empty_frame()
            df = df.xs(symbol, level=0)

        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"

        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[OHLCV_COLUMNS]

        # Drop rows with no close (corrupt/partial), de-dup, and sort.
        df = df.dropna(subset=["close"])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        return df

    def _fill_gaps(self, frame: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
        """Reindex intraday bars onto a continuous grid, forward-filling.

        Spans only the observed [first, last] range so we don't fabricate bars
        outside the data we actually received. Forward-filled bars carry the
        prior close across O/H/L/C with zero volume, marking a quiet period.
        """
        if frame.empty:
            return frame
        full_index = pd.date_range(
            start=frame.index[0],
            end=frame.index[-1],
            freq=timeframe.pandas_freq,
            tz="UTC",
        )
        reindexed = frame.reindex(full_index)
        price_cols = ["open", "high", "low", "close", "vwap"]
        reindexed[price_cols] = reindexed[price_cols].ffill()
        # A synthetic bar (no trades) collapses to a flat candle at prior close.
        synthetic = reindexed["volume"].isna()
        for col in ("open", "high", "low"):
            reindexed.loc[synthetic, col] = reindexed.loc[synthetic, "close"]
        reindexed.loc[synthetic, "vwap"] = reindexed.loc[synthetic, "close"]
        reindexed[["volume", "trade_count"]] = reindexed[["volume", "trade_count"]].fillna(0)
        reindexed.index.name = "timestamp"
        return reindexed

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=idx)

    # ------------------------------------------------------------------
    # Quotes / current price
    # ------------------------------------------------------------------
    def get_latest_quote(self, symbol: str) -> Quote:
        """Return the latest NBBO quote for ``symbol``."""
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._feed)
        raw = self._fetch_latest_quote(request)
        q = raw[symbol]
        return Quote(
            symbol=symbol,
            timestamp=q.timestamp,
            bid_price=float(q.bid_price or 0.0),
            ask_price=float(q.ask_price or 0.0),
            bid_size=float(q.bid_size or 0.0),
            ask_size=float(q.ask_size or 0.0),
        )

    def get_current_price(self, symbol: str) -> float:
        """Best estimate of the current tradable price.

        Prefers the quote midpoint; falls back to the last trade price when the
        quote is one-sided or empty (e.g. pre/post market).
        """
        quote = self.get_latest_quote(symbol)
        mid = quote.mid_price
        if mid is not None and mid > 0:
            return mid
        logger.debug("Quote midpoint unavailable for %s; using last trade", symbol)
        return self.get_last_trade_price(symbol)

    def get_last_trade_price(self, symbol: str) -> float:
        request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=self._feed)
        raw = self._fetch_latest_trade(request)
        return float(raw[symbol].price)

    @_with_retry()
    def _fetch_latest_quote(self, request: StockLatestQuoteRequest) -> Any:
        return self._client.get_stock_latest_quote(request)

    @_with_retry()
    def _fetch_latest_trade(self, request: StockLatestTradeRequest) -> Any:
        return self._client.get_stock_latest_trade(request)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _read_cache(self, symbol: str, timeframe: Timeframe) -> pd.DataFrame | None:
        with self._lock:
            entry = self._cache.get((symbol, timeframe))
            if entry is None:
                return None
            if time.monotonic() - entry.fetched_at > self._cache_ttl:
                del self._cache[(symbol, timeframe)]
                return None
            return entry.frame

    def _write_cache(self, symbol: str, timeframe: Timeframe, frame: pd.DataFrame) -> None:
        with self._lock:
            self._cache[(symbol, timeframe)] = _CacheEntry(
                frame=frame.copy(), fetched_at=time.monotonic()
            )

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
