"""Configuration loader for the SOXL/SOXS trading bot.

Reads API keys and trading parameters from environment variables (populated
from a ``.env`` file via python-dotenv) and exposes them as a typed,
validated ``Settings`` object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of this file's directory (config/ -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _strip_surrounding_quotes(value: str) -> str:
    """Remove a matching pair of surrounding quotes from a config value.

    python-dotenv strips wrapping quotes when loading a ``.env`` file, but
    Docker/Podman's ``--env-file`` parser keeps them verbatim. Stripping a
    matching leading/trailing quote here keeps string config (e.g. URLs)
    consistent regardless of how the environment was populated.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _get_str(key: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(key, default)
    if value is not None:
        value = _strip_surrounding_quotes(value)
    if required and (value is None or value == "" or value.startswith("your_")):
        raise ValueError(
            f"Missing required configuration: {key}. "
            f"Set it in your .env file (see .env.example)."
        )
    return value if value is not None else ""


def _strip_inline_comment(value: str) -> str:
    """Remove a trailing inline ``# comment`` from a config value.

    python-dotenv strips inline comments when loading a ``.env`` file, but
    Docker/Podman's ``--env-file`` parser does not: it passes everything after
    ``=`` verbatim. To keep numeric/boolean config robust regardless of how the
    environment was populated, drop anything from the first ``#`` onward. (These
    value types can never legitimately contain ``#``.)
    """
    return value.split("#", 1)[0].strip()


def _get_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return _strip_inline_comment(raw).lower() in {"1", "true", "yes", "y", "on"}


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or _strip_inline_comment(raw) == "":
        return default
    cleaned = _strip_inline_comment(raw)
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Config {key} must be a number, got {cleaned!r}") from exc


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or _strip_inline_comment(raw) == "":
        return default
    cleaned = _strip_inline_comment(raw)
    try:
        return int(cleaned)
    except ValueError as exc:
        raise ValueError(f"Config {key} must be an integer, got {cleaned!r}") from exc


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str
    secret_key: str
    paper: bool


@dataclass(frozen=True)
class FinnhubConfig:
    api_key: str


@dataclass(frozen=True)
class RiskConfig:
    max_position_pct: float
    max_daily_loss_pct: float
    max_trades_per_day: int
    atr_stop_multiplier: float
    atr_take_profit_multiplier: float
    # Hysteresis on reversals: a flip (entering one leg while the opposite is
    # held) is approved only when the new signal's confidence meets this bar.
    # 0 disables the gate, preserving the original "flip on any actionable
    # opposite signal" behavior.
    flip_confidence_threshold: float = 0.0
    # When True, the fixed ATR take-profit is dropped and the stop trails the
    # favorable excursion (an ATR-chandelier exit) so winners can run. Active in
    # the backtest broker; live wiring (an Alpaca trailing-stop order) is TODO.
    trailing_stop: bool = False


@dataclass(frozen=True)
class StrategyConfig:
    technical_weight: float
    sentiment_weight: float
    # Minimum |combined score| to take a side; below this the engine stays flat.
    entry_threshold: float = 0.08
    # News-sentiment scorer: "finbert" (finance-tuned BERT, default) or "vader"
    # (rule-based, no torch dependency). "finbert" still falls back to VADER if
    # torch/transformers can't load; "vader" skips FinBERT entirely.
    sentiment_method: str = "finbert"


@dataclass(frozen=True)
class EngineConfig:
    poll_interval_seconds: int
    close_at_eod: bool
    use_options: bool
    eod_flat_buffer_minutes: int


@dataclass(frozen=True)
class MonitoringConfig:
    alerts_enabled: bool
    discord_webhook_url: str
    alert_min_level: str
    bot_name: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    log_dir: Path
    log_file: str
    max_bytes: int
    backup_count: int


@dataclass(frozen=True)
class Settings:
    symbol_long: str
    symbol_short: str
    alpaca: AlpacaConfig
    finnhub: FinnhubConfig
    risk: RiskConfig
    strategy: StrategyConfig
    engine: EngineConfig
    monitoring: MonitoringConfig
    logging: LoggingConfig

    @property
    def symbols(self) -> tuple[str, str]:
        return (self.symbol_long, self.symbol_short)


def load_settings(env_file: str | os.PathLike[str] | None = None) -> Settings:
    """Load and validate settings from the environment / a .env file.

    Args:
        env_file: Optional explicit path to a .env file. Defaults to
            ``<project_root>/.env`` if it exists.
    """
    if env_file is None:
        default_env = PROJECT_ROOT / ".env"
        load_dotenv(default_env if default_env.exists() else None)
    else:
        load_dotenv(env_file)

    log_dir_raw = _get_str("LOG_DIR", "logs")
    log_dir = Path(log_dir_raw)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir

    settings = Settings(
        symbol_long=_get_str("SYMBOL_LONG", "SOXL"),
        symbol_short=_get_str("SYMBOL_SHORT", "SOXS"),
        alpaca=AlpacaConfig(
            api_key=_get_str("ALPACA_API_KEY", required=True),
            secret_key=_get_str("ALPACA_SECRET_KEY", required=True),
            paper=_get_bool("ALPACA_PAPER", True),
        ),
        finnhub=FinnhubConfig(
            api_key=_get_str("FINNHUB_API_KEY", required=True),
        ),
        risk=RiskConfig(
            max_position_pct=_get_float("MAX_POSITION_PCT", 1.0),
            max_daily_loss_pct=_get_float("MAX_DAILY_LOSS_PCT", 0.05),
            max_trades_per_day=_get_int("MAX_TRADES_PER_DAY", 10),
            atr_stop_multiplier=_get_float("ATR_STOP_MULTIPLIER", 2.0),
            atr_take_profit_multiplier=_get_float("ATR_TAKE_PROFIT_MULTIPLIER", 3.0),
            flip_confidence_threshold=_get_float("FLIP_CONFIDENCE_THRESHOLD", 0.3),
            trailing_stop=_get_bool("TRAILING_STOP", False),
        ),
        strategy=StrategyConfig(
            technical_weight=_get_float("TECHNICAL_WEIGHT", 0.6),
            sentiment_weight=_get_float("SENTIMENT_WEIGHT", 0.4),
            entry_threshold=_get_float("ENTRY_THRESHOLD", 0.2),
            sentiment_method=_get_str("SENTIMENT_METHOD", "vader").lower(),
        ),
        engine=EngineConfig(
            poll_interval_seconds=_get_int("POLL_INTERVAL_SECONDS", 1),
            close_at_eod=_get_bool("CLOSE_AT_EOD", True),
            use_options=_get_bool("USE_OPTIONS", False),
            eod_flat_buffer_minutes=_get_int("EOD_FLAT_BUFFER_MINUTES", 5),
        ),
        monitoring=MonitoringConfig(
            alerts_enabled=_get_bool("ALERTS_ENABLED", True),
            discord_webhook_url=_get_str("DISCORD_WEBHOOK_URL", ""),
            alert_min_level=_get_str("ALERT_MIN_LEVEL", "INFO").upper(),
            bot_name=_get_str("BOT_NAME", "trade_bot"),
        ),
        logging=LoggingConfig(
            level=_get_str("LOG_LEVEL", "INFO").upper(),
            log_dir=log_dir,
            log_file=_get_str("LOG_FILE", "trading_bot.log"),
            max_bytes=_get_int("LOG_MAX_BYTES", 10 * 1024 * 1024),
            backup_count=_get_int("LOG_BACKUP_COUNT", 5),
        ),
    )

    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    weight_sum = settings.strategy.technical_weight + settings.strategy.sentiment_weight
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            "TECHNICAL_WEIGHT + SENTIMENT_WEIGHT must sum to 1.0, "
            f"got {weight_sum:.3f}"
        )
    if settings.strategy.sentiment_method not in ("finbert", "vader"):
        raise ValueError(
            "SENTIMENT_METHOD must be 'finbert' or 'vader', "
            f"got {settings.strategy.sentiment_method!r}."
        )
    if not 0 < settings.risk.max_position_pct <= 1:
        raise ValueError("MAX_POSITION_PCT must be in (0, 1].")
    if not 0 < settings.risk.max_daily_loss_pct <= 1:
        raise ValueError("MAX_DAILY_LOSS_PCT must be in (0, 1].")
    if settings.risk.max_trades_per_day < 1:
        raise ValueError("MAX_TRADES_PER_DAY must be >= 1.")
    if settings.engine.poll_interval_seconds < 1:
        raise ValueError("POLL_INTERVAL_SECONDS must be >= 1.")
    if settings.engine.eod_flat_buffer_minutes < 0:
        raise ValueError("EOD_FLAT_BUFFER_MINUTES must be >= 0.")
