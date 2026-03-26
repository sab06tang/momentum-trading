import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ── Metrics ──────────────────────────────────────────────────────────────────

def calculate_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Computes strategy performance metrics.

    Returns raw floats — format at display time via compare_strategies().

    Parameters
    ----------
    risk_free_rate : annualized, decimal (e.g. 0.04 for 4%)
    """
    returns = returns.dropna()
    n_days = len(returns)
    if n_days < 2:
        return {}

    rf_daily = risk_free_rate / 252

    cum_returns  = (1 + returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    ann_return   = (1 + total_return) ** (252 / n_days) - 1
    ann_vol      = returns.std() * np.sqrt(252)

    excess_daily = returns - rf_daily
    sharpe = (excess_daily.mean() / excess_daily.std() * np.sqrt(252)
              if excess_daily.std() > 0 else np.nan)

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ((ann_return - risk_free_rate) / downside
               if downside > 0 else np.nan)

    max_dd = ((cum_returns / cum_returns.cummax()) - 1).min()
    calmar = (ann_return / abs(max_dd)) if max_dd != 0 else np.nan

    win_rate = (returns > 0).mean()

    return {
        "total_return": total_return,
        "ann_return":   ann_return,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "max_drawdown": max_dd,
        "calmar":       calmar,
        "win_rate":     win_rate,
        "n_days":       n_days,
    }


def compare_strategies(
    strategies: dict,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Ablation comparison table. Primary output for strategy evaluation.

    Parameters
    ----------
    strategies : {"Strategy Name": pd.Series of daily net returns, ...}

    Returns
    -------
    pd.DataFrame — formatted for display, strategies as rows.
    """
    raw = {name: calculate_metrics(ret, risk_free_rate) for name, ret in strategies.items()}
    df  = pd.DataFrame(raw).T

    fmt = {
        "total_return": "{:.1%}",
        "ann_return":   "{:.1%}",
        "ann_vol":      "{:.1%}",
        "sharpe":       "{:.2f}",
        "sortino":      "{:.2f}",
        "max_drawdown": "{:.1%}",
        "calmar":       "{:.2f}",
        "win_rate":     "{:.1%}",
    }
    display = df.copy()
    for col, f in fmt.items():
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f.format(x) if pd.notna(x) else "N/A"
            )
    display.columns = [
        "Total Return", "Ann. Return", "Ann. Vol",
        "Sharpe", "Sortino", "Max DD", "Calmar", "Win Rate", "Days"
    ]
    return display


# ── Plots ─────────────────────────────────────────────────────────────────────

def _rolling_sharpe_series(
    returns: pd.Series,
    window: int,
    rf_daily: float = 0.0,
) -> pd.Series:
    excess = returns - rf_daily
    return (excess.rolling(window).mean() / excess.rolling(window).std()) * np.sqrt(252)


def plot_performance(returns_df: pd.DataFrame, save_dir: str) -> None:
    """
    Cumulative equity curves and drawdown chart for all strategies.
    Aligns all curves to the first date where ALL strategies have valid returns,
    preventing misleading visual offsets from warmup NaN periods.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Align to common start (latest first valid date across all strategies)
    first_valid = returns_df.apply(lambda c: c.first_valid_index()).max()
    aligned = returns_df.loc[first_valid:].dropna(how="all")

    cum = (1 + aligned).cumprod() * 100

    # Equity curves
    fig, ax = plt.subplots(figsize=(13, 6))
    for col in cum.columns:
        ax.plot(cum.index, cum[col], label=col, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_title("Strategy Ablation: Cumulative Equity Curve (Log Scale)", fontsize=13)
    ax.set_ylabel("Growth of $100")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "equity_curve.png"), dpi=150)
    plt.close(fig)

    # Drawdown chart
    fig, ax = plt.subplots(figsize=(13, 4))
    for col in cum.columns:
        dd = (cum[col] / cum[col].cummax()) - 1
        ax.plot(dd.index, dd * 100, label=col, linewidth=1)
    ax.set_title("Strategy Drawdowns", fontsize=13)
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "drawdowns.png"), dpi=150)
    plt.close(fig)


def plot_rolling_sharpe(
    returns_df: pd.DataFrame,
    save_dir: str,
    window: int = 252,
    risk_free_rate: float = 0.0,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    rf_daily = risk_free_rate / 252

    fig, ax = plt.subplots(figsize=(13, 5))
    for col in returns_df.columns:
        rs = _rolling_sharpe_series(returns_df[col], window, rf_daily)
        ax.plot(rs.index, rs, label=col, linewidth=1.2)
    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(1, color="green", ls=":", lw=0.8, alpha=0.4)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=13)
    ax.set_ylabel("Sharpe Ratio")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "rolling_sharpe.png"), dpi=150)
    plt.close(fig)


def plot_regime_visualization(
    spy_prices: pd.Series,
    regime_signal: pd.Series,
    save_dir: str,
) -> None:
    """
    Shades SPY price chart by ML regime signal.

    Parameters
    ----------
    regime_signal : binary (0/1) or probability (0.0–1.0).
                    Probabilities are thresholded at 0.5 automatically.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Auto-detect and threshold probabilities
    is_prob = regime_signal.between(0, 1).all() and not set(regime_signal.dropna().unique()).issubset({0, 1})
    if is_prob:
        regime_binary = (regime_signal > 0.5).astype(int)
        print("  [INFO] regime_signal detected as probabilities — thresholded at 0.5.")
    else:
        regime_binary = regime_signal.astype(int)

    common_idx = spy_prices.index.intersection(regime_binary.index)
    prices = spy_prices.loc[common_idx]
    signal = regime_binary.loc[common_idx]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(common_idx, prices, color="black", lw=1.2, label="SPY Price")
    ax.fill_between(
        common_idx,
        prices.min(), prices.max(),
        where=(signal == 1),
        color="green", alpha=0.15, label="ML: Favorable Regime",
    )
    ax.fill_between(
        common_idx,
        prices.min(), prices.max(),
        where=(signal == 0),
        color="red", alpha=0.08, label="ML: Unfavorable Regime",
    )
    ax.set_yscale("log")
    ax.set_title("ML Regime Signal vs. SPY", fontsize=13)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "regime_visualization.png"), dpi=150)
    plt.close(fig)


def plot_feature_importance(importance_series: pd.Series, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    importance_series.head(15).plot(kind="barh", color="steelblue", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Top 15 Feature Importances (Random Forest — training data only)", fontsize=12)
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)