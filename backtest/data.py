"""Historical data loading for the backtest, via the production data classes.

Pulls the same bars and news the live bot would see — :class:`~data.market_data.MarketData`
for Alpaca OHLCV and :class:`~data.finnhub_data.FinnhubData` for company news —
over an explicit ``[start, end]`` window, and shapes them for the backtest
engine. Network access (and API keys) is required only here; the engine itself
is pure and offline-testable with in-memory frames.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from config.logging_setup import get_logger
from config.settings import Settings
from data.finnhub_data import DEFAULT_SECTOR_SYMBOLS, FinnhubData
from data.market_data import MarketData, Timeframe

logger = get_logger(__name__)


def load_bars(
    settings: Settings,
    start: datetime,
    end: datetime,
    *,
    timeframe: Timeframe = Timeframe.FIVE_MINUTE,
    market_data: MarketData | None = None,
) -> dict[str, pd.DataFrame]:
    """Load gap-filled OHLCV for both legs over ``[start, end]``.

    Gap-filling matches what the live signal pipeline requests
    (``get_bars(..., fill_gaps=True)``), so the bars the backtest sees line up
    bar-for-bar with what the bot trades on.
    """
    md = market_data or MarketData(settings.alpaca)
    bars: dict[str, pd.DataFrame] = {}
    for symbol in (settings.symbol_long, settings.symbol_short):
        frame = md.get_bars(
            symbol, timeframe, start=start, end=end, fill_gaps=True, use_cache=False
        )
        logger.info("Loaded %d %s bars for %s", len(frame), timeframe.value, symbol)
        bars[symbol] = frame
    return bars


def load_news(
    settings: Settings,
    start: datetime,
    end: datetime,
    *,
    sector_symbols: tuple[str, ...] = DEFAULT_SECTOR_SYMBOLS,
    sentiment_lookback_days: int = 7,
    finnhub: FinnhubData | None = None,
) -> dict[str, pd.DataFrame]:
    """Load company news for the pair and sector across the window.

    The window is widened back by ``sentiment_lookback_days`` so the very first
    bars of the backtest already have a populated lookback (the engine still
    only reveals articles dated at/<= each bar, so there's no look-ahead).
    """
    fh = finnhub or FinnhubData(settings.finnhub, sector_symbols=sector_symbols)
    news_start = start - timedelta(days=sentiment_lookback_days)
    symbols = (settings.symbol_long, settings.symbol_short, *sector_symbols)

    news: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frame = fh.get_company_news(symbol, start=news_start, end=end)
        except Exception as exc:  # noqa: BLE001 - one bad ticker shouldn't sink the load
            logger.warning("News load failed for %s: %s", symbol, exc)
            frame = pd.DataFrame()
        logger.info("Loaded %d news articles for %s", len(frame), symbol)
        news[symbol] = frame
    return news


def parse_window(start: str, end: str | None) -> tuple[datetime, datetime]:
    """Parse ``YYYY-MM-DD`` (or ISO) strings into a UTC ``[start, end]`` pair."""
    start_dt = _to_utc(pd.Timestamp(start))
    end_dt = _to_utc(pd.Timestamp(end)) if end else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise ValueError(f"end ({end_dt}) must be after start ({start_dt})")
    return start_dt, end_dt


def _to_utc(ts: pd.Timestamp) -> datetime:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()
