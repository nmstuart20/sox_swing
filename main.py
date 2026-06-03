"""Entry point for the SOXL/SOXS trading bot.

Step 1 scaffolding: loads configuration, initializes logging, and prints a
startup summary. The trading loop itself is implemented in later steps.
"""

from __future__ import annotations

import sys

from config import load_settings, setup_logging


def main() -> int:
    try:
        settings = load_settings()
    except ValueError as exc:
        # Logging may not be configured yet, so write straight to stderr.
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    logger = setup_logging(settings.logging)

    mode = "PAPER" if settings.alpaca.paper else "LIVE"
    logger.info("Starting SOXL/SOXS trading bot")
    logger.info("Mode: %s | Symbols: %s/%s", mode, settings.symbol_long, settings.symbol_short)
    logger.info(
        "Risk: max_pos=%.0f%% max_daily_loss=%.0f%% max_trades=%d",
        settings.risk.max_position_pct * 100,
        settings.risk.max_daily_loss_pct * 100,
        settings.risk.max_trades_per_day,
    )
    logger.info(
        "Strategy weights: technical=%.2f sentiment=%.2f | options=%s",
        settings.strategy.technical_weight,
        settings.strategy.sentiment_weight,
        settings.engine.use_options,
    )

    # NOTE: The orchestration loop is built in step 10 (main orchestration loop).
    logger.info("Scaffolding ready. Trading loop not yet implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
