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


def _get_str(key: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and (value is None or value == "" or value.startswith("your_")):
        raise ValueError(
            f"Missing required configuration: {key}. "
            f"Set it in your .env file (see .env.example)."
        )
    return value if value is not None else ""


def _get_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Config {key} must be a number, got {raw!r}") from exc


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Config {key} must be an integer, got {raw!r}") from exc


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


@dataclass(frozen=True)
class StrategyConfig:
    technical_weight: float
    sentiment_weight: float


@dataclass(frozen=True)
class EngineConfig:
    poll_interval_seconds: int
    close_at_eod: bool
    use_options: bool


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
            max_position_pct=_get_float("MAX_POSITION_PCT", 0.10),
            max_daily_loss_pct=_get_float("MAX_DAILY_LOSS_PCT", 0.03),
            max_trades_per_day=_get_int("MAX_TRADES_PER_DAY", 10),
            atr_stop_multiplier=_get_float("ATR_STOP_MULTIPLIER", 2.0),
            atr_take_profit_multiplier=_get_float("ATR_TAKE_PROFIT_MULTIPLIER", 3.0),
        ),
        strategy=StrategyConfig(
            technical_weight=_get_float("TECHNICAL_WEIGHT", 0.7),
            sentiment_weight=_get_float("SENTIMENT_WEIGHT", 0.3),
        ),
        engine=EngineConfig(
            poll_interval_seconds=_get_int("POLL_INTERVAL_SECONDS", 60),
            close_at_eod=_get_bool("CLOSE_AT_EOD", True),
            use_options=_get_bool("USE_OPTIONS", False),
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
    if not 0 < settings.risk.max_position_pct <= 1:
        raise ValueError("MAX_POSITION_PCT must be in (0, 1].")
    if not 0 < settings.risk.max_daily_loss_pct <= 1:
        raise ValueError("MAX_DAILY_LOSS_PCT must be in (0, 1].")
    if settings.risk.max_trades_per_day < 1:
        raise ValueError("MAX_TRADES_PER_DAY must be >= 1.")
    if settings.engine.poll_interval_seconds < 1:
        raise ValueError("POLL_INTERVAL_SECONDS must be >= 1.")
