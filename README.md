# SOXL/SOXS Trading Bot

An automated bot that trades **only SOXL and SOXS** through **Alpaca**, using
technical analysis and FinBERT-scored Finnhub news sentiment (falling back to
VADER on hosts without torch, e.g. the 32-bit ARM Pi).

## Quick start

```bash
git clone <repo-url> && cd semis_bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your Alpaca + Finnhub API keys + Discord webhook url
python3 main.py
```

## Tests

```bash
pip install -r requirements-dev.txt    # one-time: test deps

pytest                                 # run everything
pytest -v                              # verbose, per-test output
pytest tests/test_risk_manager.py      # a single module
```

## Backtesting

Replay historical bars and news through the same strategy the live bot uses.
Keys are read from `.env`.

```bash
# Replay a date window:
python -m backtest --start 2026-05-20 --end 2026-06-03

# Bigger account, wider slippage, per-share commission, write the trade log:
python -m backtest --start 2026-05-01 --capital 250000 \
    --slippage-bps 5 --commission-per-share 0.005 --trade-log trades.csv

# Technicals only (skip the Finnhub news pull):
python -m backtest --start 2026-05-20 --no-news
```

Reports total return, annualized Sharpe, max drawdown, win rate, and a per-trade
log. `--trade-log PATH` writes the full log to CSV.

## Deploy with containers (Podman or Docker)

```bash
cp .env.example .env           # fill in your sensitive (stays on the host)

podman compose up -d --build   # build + run detached
podman compose logs -f         # follow logs
podman compose down            # stop (flattens positions on SIGTERM)
```

Docker is identical — swap `podman` for `docker`. Logs are written to `./logs`
on the host via a volume mount, so they survive rebuilds.

### Without compose

If a host has no compose plugin, build and run the image directly. The image
runs as an unprivileged user (`appuser`, UID 10001), so the host `./logs` dir
must be writable by that user. Podman's `:U` handles this automatically by
chowning the mount; Docker has no equivalent, so chown the dir once instead.
`:Z` (Podman on SELinux hosts) is a no-op elsewhere.

```bash
# Podman — :U chowns ./logs to the container user (fixes "permission denied")
podman build -t soxs-bot .
podman run -d --name soxs-bot --restart unless-stopped \
  --env-file .env -v ./logs:/app/logs:U,Z soxs-bot

# Docker — chown the host logs dir to the container's UID once, then run
docker build -t soxs-bot .
mkdir -p logs && sudo chown -R 10001:10001 logs
docker run -d --name soxs-bot --restart unless-stopped \
  --env-file .env -v "$(pwd)/logs:/app/logs" soxs-bot
```

Manage it with `podman logs -f soxs-bot` / `docker logs -f soxs-bot` and
`podman stop soxs-bot` / `docker stop soxs-bot`.

For an always-on service that starts on boot and restarts on crash,
`deploy/soxs-bot.container` is a Podman **Quadlet** unit; see the comments at the
top of that file for install steps.

## Monitoring & alerts

The bot logs every decision and trade, tracks running P&L, and can push
**Discord alerts** on entries, exits, errors, and risk-limit breaches. Alerts are
opt-in via environment variables (all optional):

| Variable | Default | Purpose |
| --- | --- | --- |
| `ALERTS_ENABLED` | `true` | Master switch for Discord alerts |
| `DISCORD_WEBHOOK_URL` | _(empty)_ | Incoming-webhook URL; empty = alerts disabled |
| `ALERT_MIN_LEVEL` | `INFO` | Minimum severity: `INFO`/`SUCCESS`/`WARNING`/`ERROR` |
| `BOT_NAME` | `trade_bot` | Display name on webhook messages |

With no webhook URL the notifier is a silent no-op, so the bot runs fine without
Discord configured.
