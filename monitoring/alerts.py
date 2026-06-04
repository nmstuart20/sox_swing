"""Discord alerting for the SOXL/SOXS trading bot.

A small, dependency-free notifier that posts rich-embed messages to a Discord
incoming webhook. It is built to sit *beside* the trading loop without ever
getting in its way:

  * **Non-blocking.** Calls to :meth:`DiscordNotifier.send` enqueue the message
    and return immediately; a single daemon worker thread drains the queue and
    performs the (potentially slow) HTTP POST. A wedged or slow webhook can
    never stall a trading cycle.
  * **Best-effort.** Network/HTTP failures are logged and swallowed — an alert
    that can't be delivered must not crash the bot or bubble into the loop.
  * **Degradable.** With no webhook URL (or ``enabled=False``) the notifier is a
    silent no-op, so the rest of the code can call ``send`` unconditionally.

Only the standard library is used (``urllib``) so monitoring adds no new
dependency to ``requirements.txt``.
"""

from __future__ import annotations

import json
import queue
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from config.logging_setup import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Severity of an alert, ordered so callers can filter by a minimum level.

    ``severity`` doubles as the sort key (higher = more urgent) and ``color`` is
    the Discord embed accent (a 24-bit RGB int).
    """

    INFO = (10, 0x5865F2)      # blurple
    SUCCESS = (20, 0x2ECC71)   # green
    WARNING = (30, 0xE67E22)   # orange
    ERROR = (40, 0xE74C3C)     # red

    def __init__(self, severity: int, color: int) -> None:
        self.severity = severity
        self.color = color

    @classmethod
    def from_name(cls, name: str) -> "AlertLevel":
        """Parse a level name (case-insensitive); falls back to ``INFO``."""
        try:
            return cls[name.strip().upper()]
        except KeyError:
            logger.warning("Unknown alert level %r — defaulting to INFO", name)
            return cls.INFO


@dataclass(frozen=True)
class _Alert:
    """A queued alert awaiting delivery by the worker thread."""

    title: str
    message: str
    level: AlertLevel
    fields: tuple[tuple[str, str], ...]


class DiscordNotifier:
    """Posts embed messages to a Discord webhook from a background thread.

    Args:
        webhook_url: the Discord incoming-webhook URL. Empty/``None`` disables
            delivery (the notifier becomes a no-op).
        enabled: master switch; when False nothing is ever sent.
        username: display name for the webhook messages.
        min_level: drop any alert below this severity.
        timeout: per-request HTTP timeout in seconds.
        queue_size: bound on undelivered alerts before the oldest are dropped.
    """

    def __init__(
        self,
        webhook_url: str | None,
        *,
        enabled: bool = True,
        username: str = "SOXL/SOXS Bot",
        min_level: AlertLevel = AlertLevel.INFO,
        timeout: float = 10.0,
        queue_size: int = 100,
    ) -> None:
        self._username = username
        self._min_level = min_level
        self._timeout = timeout
        self._webhook_url = (webhook_url or "").strip()
        self._enabled = bool(enabled and self._webhook_url)

        self._queue: "queue.Queue[_Alert | None]" = queue.Queue(maxsize=queue_size)
        self._worker: threading.Thread | None = None

        if self._enabled:
            self._worker = threading.Thread(
                target=self._run, name="discord-notifier", daemon=True
            )
            self._worker.start()
            logger.info("Discord alerts enabled (min_level=%s)", min_level.name)
        elif enabled and not self._webhook_url:
            logger.warning("Discord alerts requested but no webhook URL set — disabled")
        else:
            logger.info("Discord alerts disabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    def send(
        self,
        title: str,
        message: str = "",
        level: AlertLevel = AlertLevel.INFO,
        fields: dict[str, object] | None = None,
    ) -> None:
        """Enqueue an alert for delivery (returns immediately).

        Silently ignored when the notifier is disabled or the level is below
        ``min_level``. If the queue is full the alert is dropped with a warning
        rather than blocking the caller (the trading loop).
        """
        if not self._enabled or level.severity < self._min_level.severity:
            return
        alert = _Alert(
            title=title,
            message=message,
            level=level,
            fields=tuple((str(k), str(v)) for k, v in (fields or {}).items()),
        )
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            logger.warning("Discord alert queue full — dropping alert %r", title)

    def close(self, timeout: float = 5.0) -> None:
        """Flush pending alerts and stop the worker (best-effort)."""
        if not self._enabled or self._worker is None:
            return
        try:
            self._queue.put_nowait(None)  # sentinel: drain then exit
        except queue.Full:  # pragma: no cover - unlikely
            return
        self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            alert = self._queue.get()
            try:
                if alert is None:  # sentinel
                    return
                self._deliver(alert)
            except Exception as exc:  # noqa: BLE001 - delivery must never crash
                logger.warning("Discord delivery error: %s", exc)
            finally:
                self._queue.task_done()

    def _deliver(self, alert: _Alert) -> None:
        payload = self._build_payload(alert)
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._webhook_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "semis-bot"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                # Discord returns 204 No Content on success.
                if response.status >= 300:
                    logger.warning("Discord webhook HTTP %s for %r", response.status, alert.title)
        except urllib.error.HTTPError as exc:
            logger.warning("Discord webhook rejected %r: HTTP %s", alert.title, exc.code)
        except urllib.error.URLError as exc:
            logger.warning("Discord webhook unreachable for %r: %s", alert.title, exc.reason)

    def _build_payload(self, alert: _Alert) -> dict[str, object]:
        embed: dict[str, object] = {
            "title": alert.title[:256],
            "color": alert.level.color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": f"{self._username} • {alert.level.name}"},
        }
        if alert.message:
            embed["description"] = alert.message[:4096]
        if alert.fields:
            embed["fields"] = [
                {"name": name[:256], "value": (value or "—")[:1024], "inline": True}
                for name, value in alert.fields
            ]
        return {"username": self._username, "embeds": [embed]}
