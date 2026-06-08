"""Point-in-time news/sentiment replay for the backtest.

:class:`BacktestFinnhub` mimics the slice of
:class:`~data.finnhub_data.FinnhubData` the signal engine actually uses —
:meth:`get_news_sentiment` and the :attr:`sector_symbols` property — but serves
sentiment computed *only* from articles dated at or before the current
simulated bar. The engine's :meth:`SignalEngine.evaluate` →
:meth:`sentiment_for_symbols` path therefore runs completely unchanged, with no
look-ahead leaking future headlines into a past decision.

It reproduces the production sentiment path: the score for a symbol is the mean
per-article sentiment over the lookback window, scored with the same FinBERT
model the live bot uses (falling back to VADER when torch isn't available) via
:func:`~data.finnhub_data.score_news_texts`.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from config.logging_setup import get_logger
from data.finnhub_data import NEWS_COLUMNS, SentimentResult, score_news_texts

logger = get_logger(__name__)


class BacktestFinnhub:
    """Replays loaded news as point-in-time sentiment, like ``FinnhubData``.

    Args:
        news: ``{symbol: DataFrame}`` of articles with at least ``timestamp``
            (tz-aware UTC) and ``headline``/``summary`` columns. A ``sentiment``
            column is used if present; otherwise it's computed per article with
            the production scorer (FinBERT, or the VADER fallback).
        sector_symbols: the semiconductor-sector tickers the engine blends in
            (matches :attr:`FinnhubData.sector_symbols`).
    """

    def __init__(
        self,
        news: dict[str, pd.DataFrame],
        sector_symbols: tuple[str, ...] = (),
        sentiment_method: str = "finbert",
    ) -> None:
        self._method = sentiment_method
        # Source of the per-article scores; flips to "vader" if any frame had
        # to fall back during _prepare (see score_news_texts).
        self._source = "vader" if sentiment_method == "vader" else "finbert"
        self._news = {sym: self._prepare(df) for sym, df in news.items()}
        self._sector = tuple(sector_symbols)
        self._now: datetime | None = None
        total = sum(len(df) for df in self._news.values())
        logger.info(
            "BacktestFinnhub initialized (%d symbols, %d articles, sector=%s)",
            len(self._news), total, ",".join(self._sector) or "none",
        )

    @property
    def sector_symbols(self) -> tuple[str, ...]:
        return self._sector

    def set_time(self, now: datetime) -> None:
        """Advance the replay clock; only articles at/<= ``now`` are visible."""
        self._now = now

    def get_news_sentiment(self, symbol: str, lookback_days: int = 7) -> SentimentResult:
        """Mean per-article sentiment for ``symbol`` over the trailing window.

        Matches :meth:`FinnhubData.get_news_sentiment`: an empty window scores a
        neutral 0.0 so the engine degrades to technicals when there's no fresh
        news. ``source`` reflects whichever scorer ran (FinBERT or VADER).
        """
        df = self._news.get(symbol)
        if df is None or df.empty or self._now is None:
            return SentimentResult(symbol, 0.0, article_count=0, source=self._source)

        low = pd.Timestamp(self._now) - pd.Timedelta(days=lookback_days)
        window = df[(df["timestamp"] > low) & (df["timestamp"] <= pd.Timestamp(self._now))]
        if window.empty:
            return SentimentResult(symbol, 0.0, article_count=0, source=self._source)
        score = float(window["sentiment"].mean())
        return SentimentResult(symbol, score, article_count=len(window), source=self._source)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a news frame: UTC timestamps, a per-article sentiment column."""
        if df is None or df.empty:
            empty = pd.DataFrame(columns=NEWS_COLUMNS)
            empty["timestamp"] = pd.to_datetime(empty["timestamp"], utc=True)
            return empty
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        if "sentiment" not in out.columns or out["sentiment"].isna().any():
            headline = out.get("headline", "").fillna("") if "headline" in out else ""
            summary = out.get("summary", "").fillna("") if "summary" in out else ""
            text = (headline.astype(str) + ". " + summary.astype(str)) if len(out) else []
            out["sentiment"], source = score_news_texts(list(text), method=self._method)
            if source == "vader":
                self._source = "vader"
        return out.sort_values("timestamp", ignore_index=True)
