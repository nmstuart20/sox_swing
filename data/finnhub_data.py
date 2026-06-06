"""Finnhub news and sentiment ingestion for SOXL/SOXS.

Pulls company news, news-sentiment scores, and economic data from Finnhub and
returns clean, timestamp-indexed pandas DataFrames the signal engine can fold
into a trade decision. Responsibilities:

  * fetch company news for the SOXL/SOXS pair and the underlying semiconductor
    sector (SMH, NVDA, AMD, ... — configurable),
  * normalize each article into a tidy row (timestamp, symbol, headline,
    source, url, summary, sentiment),
  * attach a sentiment score per article with FinBERT (a finance-tuned BERT),
    falling back to VADER (a rule-based sentiment analyzer) when
    torch/transformers aren't available (e.g. the 32-bit ARM Pi target, where
    torch has no wheels),
  * expose an aggregate sentiment score per symbol and best-effort economic
    data lookups,
  * stay under Finnhub's free-tier rate limit (60 calls/min) with a proactive
    request gate, and retry transient errors / HTTP 429 with backoff.

Only Finnhub access lives here; OHLCV bars come from ``data/market_data.py``
and order placement from ``execution/alpaca_client.py``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Iterable, Sequence, TypeVar

import pandas as pd

import finnhub

from config.logging_setup import get_logger
from config.settings import FinnhubConfig
from data.finbert import FinBertScorer, FinBertUnavailable, get_default_scorer

logger = get_logger(__name__)

T = TypeVar("T")

# Canonical column order for every news frame this module returns.
NEWS_COLUMNS = ["timestamp", "symbol", "headline", "source", "url", "summary", "sentiment"]

# Default semiconductor-sector tickers to pull news for alongside SOXL/SOXS.
# SMH is the sector ETF SOXL/SOXS track; NVDA/AMD are the heaviest movers.
DEFAULT_SECTOR_SYMBOLS = ("SMH", "NVDA", "AMD", "TSM", "AVGO", "INTC", "TXN", "MU", "MRVL", "CDNS", "AMAT", "QCOM", "AMD", "LRCX", "KLAC", "ASML")

# Finnhub's free tier allows 60 API calls per minute; a 1.05s floor between
# calls keeps us comfortably under it without external coordination.
_DEFAULT_MIN_REQUEST_INTERVAL = 1.05

# VADER (Valence Aware Dictionary and sEntiment Reasoner) backs the sentiment
# fallback used when FinBERT can't load. It's a pure-Python, rule-based analyzer
# with no compiled dependencies, so it installs and runs even on the 32-bit ARM
# Pi target where torch has no wheels. The analyzer carries a ~7k-word lexicon
# loaded from disk, so we build it once, lazily, and reuse it.
_vader_analyzer = None
_vader_lock = threading.Lock()


def _get_vader_analyzer():
    """Return the process-wide VADER analyzer, building it on first use."""
    global _vader_analyzer
    if _vader_analyzer is None:
        with _vader_lock:
            if _vader_analyzer is None:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

                _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


class FinnhubError(Exception):
    """Raised when a Finnhub fetch fails after retries."""


def _with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a method on transient errors with exponential backoff.

    HTTP 429 (rate limit) is retried after a delay. Other client errors
    (4xx) are fatal and not retried, since retrying a bad request won't help.
    Mirrors the retry policy in ``data/market_data.py``.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = base_delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except finnhub.FinnhubAPIException as exc:
                    status = getattr(exc, "status_code", None)
                    fatal = (
                        isinstance(status, int)
                        and 400 <= status < 500
                        and status != 429
                    )
                    last_exc = exc
                    if fatal:
                        logger.error("%s failed (HTTP %s): %s", func.__name__, status, exc)
                        raise FinnhubError(f"{func.__name__} failed: {exc}") from exc
                    logger.warning(
                        "%s rate-limited/transient (attempt %d/%d): %s",
                        func.__name__, attempt, max_attempts, exc,
                    )
                except (finnhub.FinnhubRequestException, ConnectionError, TimeoutError, OSError) as exc:
                    last_exc = exc
                    logger.warning(
                        "%s network error (attempt %d/%d): %s",
                        func.__name__, attempt, max_attempts, exc,
                    )
                if attempt < max_attempts:
                    time.sleep(delay)
                    delay *= backoff
            raise FinnhubError(
                f"{func.__name__} failed after {max_attempts} attempts: {last_exc}"
            ) from last_exc

        return wrapper

    return decorator


def vader_sentiment(text: str | None) -> float:
    """Score free text in ``[-1, 1]`` with VADER (positive = bullish).

    Returns VADER's ``compound`` score, which is already normalized to
    ``[-1, 1]``; empty text scores a neutral 0.0. This is the fallback used when
    FinBERT can't be loaded (see :func:`score_news_texts`).
    """
    if not text or not text.strip():
        return 0.0
    return float(_get_vader_analyzer().polarity_scores(text)["compound"])


def score_news_texts(
    texts: Sequence[str | None],
    scorer: FinBertScorer | None = None,
) -> tuple[list[float], str]:
    """Score article texts in ``[-1, 1]``, preferring FinBERT.

    Falls back to :func:`vader_sentiment` if FinBERT can't be loaded (missing
    torch/transformers, or weights unreachable). Returns ``(scores, source)``
    where ``source`` is ``"finbert"`` or ``"vader"`` so callers can label the
    aggregate they build from these scores.
    """
    scorer = scorer or get_default_scorer()
    try:
        return scorer.score_texts(texts), "finbert"
    except FinBertUnavailable as exc:
        logger.info("FinBERT unavailable (%s); using VADER sentiment", exc)
        return [vader_sentiment(t) for t in texts], "vader"


@dataclass(frozen=True)
class SentimentResult:
    """Aggregate sentiment for a symbol and where it came from."""

    symbol: str
    score: float  # normalized to [-1, 1]; positive = bullish
    article_count: int
    source: str  # "finbert" or "vader"


class FinnhubData:
    """Fetches and normalizes Finnhub news, sentiment, and economic data.

    A single shared instance is intended per process; the proactive request
    gate is instance-level, so funnel all Finnhub calls through one object to
    respect the rate limit.
    """

    def __init__(
        self,
        config: FinnhubConfig,
        sector_symbols: Sequence[str] = DEFAULT_SECTOR_SYMBOLS,
        min_request_interval: float = _DEFAULT_MIN_REQUEST_INTERVAL,
        finbert: FinBertScorer | None = None,
    ) -> None:
        self._client = finnhub.Client(api_key=config.api_key)
        self._sector_symbols = tuple(sector_symbols)
        self._min_interval = max(0.0, min_request_interval)
        self._gate_lock = threading.Lock()
        self._last_call_at = 0.0
        self._finbert = finbert or get_default_scorer()
        # Set once FinBERT proves unavailable (no torch wheels, weights
        # unreachable, ...) so we don't re-probe — and re-log — every cycle.
        self._finbert_unavailable = False
        logger.info(
            "FinnhubData initialized (sector=%s, min_interval=%.2fs)",
            ",".join(self._sector_symbols) or "none",
            self._min_interval,
        )

    @property
    def sector_symbols(self) -> tuple[str, ...]:
        return self._sector_symbols

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        """Block until at least ``min_interval`` has elapsed since the last call."""
        if self._min_interval <= 0:
            return
        with self._gate_lock:
            wait = self._min_interval - (time.monotonic() - self._last_call_at)
            if wait > 0:
                time.sleep(wait)
            self._last_call_at = time.monotonic()

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------
    def get_company_news(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        """Return normalized company news for ``symbol``.

        The result has columns :data:`NEWS_COLUMNS`, sorted by ascending
        timestamp (tz-aware UTC), with a per-article sentiment score.

        Args:
            start/end: explicit window. ``end`` defaults to now; ``start``
                defaults to ``end - lookback_days``. Finnhub's company-news
                endpoint expects ``YYYY-MM-DD`` dates.
            lookback_days: window size used when ``start`` is omitted.
        """
        end = end or datetime.now(timezone.utc)
        start = start or (end - timedelta(days=lookback_days))
        raw = self._fetch_company_news(
            symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )
        return self._to_news_frame(raw, symbol)

    def get_sector_news(self, lookback_days: int = 7) -> pd.DataFrame:
        """Return combined news for the configured semiconductor-sector symbols."""
        return self.get_news(self._sector_symbols, lookback_days=lookback_days)

    def get_news(
        self,
        symbols: Iterable[str],
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        """Fetch and concatenate normalized news for several symbols.

        A failure on one symbol is logged and skipped rather than aborting the
        whole pull, so a single bad ticker can't blank the sentiment feed.
        """
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            try:
                frames.append(self.get_company_news(symbol, lookback_days=lookback_days))
            except FinnhubError as exc:
                logger.warning("Skipping news for %s: %s", symbol, exc)
        if not frames:
            return self._empty_news_frame()
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["symbol", "url", "headline"])
        combined = combined.sort_values("timestamp", ignore_index=True)
        return combined

    @_with_retry()
    def _fetch_company_news(self, symbol: str, start: str, end: str) -> list[dict[str, Any]]:
        self._throttle()
        result = self._client.company_news(symbol, _from=start, to=end)
        return result if isinstance(result, list) else []

    def _to_news_frame(self, raw: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
        """Convert Finnhub's raw article list into a clean, scored frame."""
        if not raw:
            logger.debug("No news returned for %s", symbol)
            return self._empty_news_frame()

        rows: list[dict[str, Any]] = []
        for article in raw:
            ts = article.get("datetime")
            if not ts:
                continue
            headline = (article.get("headline") or "").strip()
            summary = (article.get("summary") or "").strip()
            rows.append(
                {
                    "timestamp": pd.to_datetime(int(ts), unit="s", utc=True),
                    "symbol": symbol,
                    "headline": headline,
                    "source": (article.get("source") or "").strip(),
                    "url": article.get("url") or "",
                    "summary": summary,
                }
            )
        if not rows:
            return self._empty_news_frame()

        scores = self._score_texts([f"{r['headline']}. {r['summary']}" for r in rows])
        for row, score in zip(rows, scores):
            row["sentiment"] = score

        df = pd.DataFrame(rows, columns=NEWS_COLUMNS)
        df = df[df["headline"] != ""]
        df = df.drop_duplicates(subset=["url", "headline"])
        df = df.sort_values("timestamp", ignore_index=True)
        return df

    @staticmethod
    def _empty_news_frame() -> pd.DataFrame:
        df = pd.DataFrame(columns=NEWS_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["sentiment"] = df["sentiment"].astype(float)
        return df

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------
    def get_news_sentiment(self, symbol: str, lookback_days: int = 7) -> SentimentResult:
        """Aggregate sentiment for ``symbol`` in ``[-1, 1]``.

        The score is the mean per-article FinBERT sentiment over recent company
        news (positive = bullish); each article is scored when its news frame is
        built. Reports ``source="finbert"``, or ``"vader"`` if FinBERT had to
        fall back. An empty window scores a neutral 0.0.
        """
        news = self.get_company_news(symbol, lookback_days=lookback_days)
        source = "vader" if self._finbert_unavailable else "finbert"
        if news.empty:
            return SentimentResult(symbol, 0.0, article_count=0, source=source)
        score = float(news["sentiment"].mean())
        return SentimentResult(symbol, score, article_count=len(news), source=source)

    def _score_texts(self, texts: list[str]) -> list[float]:
        """Score article texts in ``[-1, 1]`` via FinBERT, with VADER fallback.

        Once FinBERT proves unavailable we latch onto VADER for the rest of the
        process so we don't re-probe (or re-log) every cycle.
        """
        if self._finbert_unavailable:
            return [vader_sentiment(t) for t in texts]
        scores, source = score_news_texts(texts, self._finbert)
        if source == "vader":
            self._finbert_unavailable = True
        return scores

    # ------------------------------------------------------------------
    # Economic data (best-effort)
    # ------------------------------------------------------------------
    def get_economic_data(self, code: str) -> pd.DataFrame:
        """Return a Finnhub economic-data series as a (period, value) frame.

        ``code`` is a Finnhub economic code (see ``economic_code``); both that
        directory and this series are premium on many plans, so an empty frame
        is returned rather than raising if the data isn't accessible.
        """
        try:
            raw = self._fetch_economic_data(code)
        except FinnhubError as exc:
            logger.debug("Economic data unavailable for %s: %s", code, exc)
            return pd.DataFrame(columns=["period", "value"])
        points = raw.get("data", []) if isinstance(raw, dict) else []
        if not points:
            return pd.DataFrame(columns=["period", "value"])
        df = pd.DataFrame(points)
        if "period" in df.columns:
            df["period"] = pd.to_datetime(df["period"], errors="coerce")
            df = df.sort_values("period", ignore_index=True)
        return df

    @_with_retry()
    def _fetch_economic_data(self, code: str) -> dict[str, Any]:
        self._throttle()
        return self._client.economic_data(code)
