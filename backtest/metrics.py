"""Performance metrics and report formatting for a backtest run.

Everything here is derived from two artifacts the :class:`~backtest.broker.SimBroker`
produces during a replay: the mark-to-market equity curve and the realized
trade log. Keeping the analytics downstream of those two series means the
numbers reflect exactly what the simulated account experienced bar by bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from backtest.broker import Trade

# Trading days per year, for annualizing a daily Sharpe.
_TRADING_DAYS = 252


@dataclass(frozen=True)
class PerformanceMetrics:
    """Headline backtest statistics."""

    initial_equity: float
    final_equity: float
    total_return: float          # fractional, e.g. 0.12 == +12%
    sharpe: float                # annualized, rf = 0, from daily returns
    max_drawdown: float          # fractional, negative (e.g. -0.08 == -8%)
    win_rate: float              # winning trades / closed trades
    num_trades: int
    wins: int
    losses: int
    avg_win: float               # mean net P&L of winners ($)
    avg_loss: float              # mean net P&L of losers ($)
    profit_factor: float         # gross profit / gross loss
    total_commission: float
    start: datetime | None
    end: datetime | None

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_equity": round(self.initial_equity, 2),
            "final_equity": round(self.final_equity, 2),
            "total_return": round(self.total_return, 5),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 5),
            "win_rate": round(self.win_rate, 4),
            "num_trades": self.num_trades,
            "wins": self.wins,
            "losses": self.losses,
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 4),
            "total_commission": round(self.total_commission, 2),
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
        }


def equity_series(equity_curve: list[tuple[datetime, float]]) -> pd.Series:
    """Build a tz-aware, time-indexed equity :class:`~pandas.Series`."""
    if not equity_curve:
        return pd.Series(dtype=float)
    times, values = zip(*equity_curve)
    idx = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return pd.Series(values, index=idx, dtype=float)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of the curve, as a negative fraction."""
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak
    return float(drawdown.min())


def sharpe_ratio(equity: pd.Series, periods_per_year: int = _TRADING_DAYS) -> float:
    """Annualized Sharpe (rf = 0) from the daily close-of-day equity returns.

    The intraday curve is resampled to one point per calendar day so the ratio
    isn't inflated by the bar frequency; with fewer than two days, or a flat
    curve, it's reported as 0.
    """
    if equity.empty:
        return 0.0
    daily = equity.resample("1D").last().dropna()
    returns = daily.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0 or math.isnan(std):
        return 0.0
    return float(returns.mean() / std * math.sqrt(periods_per_year))


def compute_metrics(
    equity_curve: list[tuple[datetime, float]],
    trades: list[Trade],
    total_commission: float,
) -> PerformanceMetrics:
    """Roll the equity curve and trade log into a :class:`PerformanceMetrics`."""
    equity = equity_series(equity_curve)
    initial = float(equity.iloc[0]) if not equity.empty else 0.0
    final = float(equity.iloc[-1]) if not equity.empty else 0.0
    total_return = (final / initial - 1.0) if initial else 0.0

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = -sum(t.net_pnl for t in losses)  # positive magnitude
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else math.inf

    return PerformanceMetrics(
        initial_equity=initial,
        final_equity=final,
        total_return=total_return,
        sharpe=sharpe_ratio(equity),
        max_drawdown=max_drawdown(equity),
        win_rate=(len(wins) / len(trades)) if trades else 0.0,
        num_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        avg_win=(gross_profit / len(wins)) if wins else 0.0,
        avg_loss=(-gross_loss / len(losses)) if losses else 0.0,
        profit_factor=profit_factor,
        total_commission=total_commission,
        start=equity.index[0].to_pydatetime() if not equity.empty else None,
        end=equity.index[-1].to_pydatetime() if not equity.empty else None,
    )


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def format_report(
    metrics: PerformanceMetrics,
    trades: list[Trade],
    *,
    max_trades: int | None = 50,
    title: str = "Backtest results",
) -> str:
    """Render a human-readable summary plus a trade-log table."""
    pf = "inf" if math.isinf(metrics.profit_factor) else f"{metrics.profit_factor:.2f}"
    window = "—"
    if metrics.start and metrics.end:
        window = f"{metrics.start:%Y-%m-%d %H:%M} → {metrics.end:%Y-%m-%d %H:%M} UTC"

    lines = [
        "=" * 64,
        f" {title}",
        "=" * 64,
        f" Window           : {window}",
        f" Initial equity   : ${metrics.initial_equity:,.2f}",
        f" Final equity     : ${metrics.final_equity:,.2f}",
        f" Total return     : {metrics.total_return:+.2%}",
        f" Sharpe (ann.)    : {metrics.sharpe:.2f}",
        f" Max drawdown     : {metrics.max_drawdown:.2%}",
        f" Win rate         : {metrics.win_rate:.1%} "
        f"({metrics.wins}W / {metrics.losses}L of {metrics.num_trades})",
        f" Avg win / loss   : ${metrics.avg_win:,.2f} / ${metrics.avg_loss:,.2f}",
        f" Profit factor    : {pf}",
        f" Total commission : ${metrics.total_commission:,.2f}",
        "=" * 64,
    ]

    if trades:
        lines.append(" Trade log")
        lines.append(" " + "-" * 62)
        lines.append(
            f" {'#':>3} {'sym':<5} {'qty':>5} {'entry':>9} {'exit':>9} "
            f"{'net P&L':>10} {'ret':>7} {'bars':>5}  reason"
        )
        shown = trades if max_trades is None else trades[:max_trades]
        for i, t in enumerate(shown, start=1):
            lines.append(
                f" {i:>3} {t.symbol:<5} {t.qty:>5.0f} {t.entry_price:>9.3f} "
                f"{t.exit_price:>9.3f} {t.net_pnl:>+10.2f} {t.return_pct:>+7.2%} "
                f"{t.bars_held:>5}  {t.exit_reason}"
            )
        if max_trades is not None and len(trades) > max_trades:
            lines.append(f" … {len(trades) - max_trades} more trade(s) omitted")
        lines.append(" " + "-" * 62)
    else:
        lines.append(" No trades were taken.")

    return "\n".join(lines)


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    """The trade log as a DataFrame (for CSV export / further analysis)."""
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol", "qty", "entry_time", "entry_price", "exit_time",
                "exit_price", "exit_reason", "gross_pnl", "commission",
                "net_pnl", "return_pct", "bars_held",
            ]
        )
    return pd.DataFrame([t.to_dict() for t in trades])
