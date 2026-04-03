import pandas as pd
import numpy as np


def run_backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    tc: float = 0.001,
) -> pd.Series:
    """
    Vectorized backtest with monthly rebalancing and daily weight drift.

    Timing convention (Convention B — end-of-period weights):
      - actual_weights[i]  = weights at END of day i (post-drift, post-rebalance)
      - gross_returns[i+1] = actual_weights[i] · returns[i+1]   (via shift(1))
      - Rebalancing at close of EOM date d → new weights applied from d+1 onwards

    Signal lag:
      - target_weights must be derived from features already lagged 1 day.
      - With shift(1), total lag = 2 days (feature lag + execution lag). Conservative and clean.

    Parameters
    ----------
    returns        : daily log or simple returns, daily index
    target_weights : desired weights; must have daily index (forward-filled from monthly signal)
    tc             : one-way transaction cost in decimal (default 10 bps = 0.001)
    """
    # alignment and validation
    # forward-fill monthly target weights to daily index
    target_weights = target_weights.reindex(returns.index).ffill()
    n_null = target_weights.isna().any(axis=1).sum()
    if n_null > 0:
        raise ValueError(
            f"{n_null} rows in target_weights are NaN after ffill. "
            "Ensure weights exist before the returns start date."
        )

    common_idx = returns.index.intersection(target_weights.index)
    returns = returns.loc[common_idx]
    target_weights = target_weights.loc[common_idx]

    # rebalance schedule
    eom_dates = returns.resample(pd.offsets.BusinessMonthEnd()).last().index
    rebalance_mask = returns.index.isin(eom_dates)

    # loop
    n = len(returns)
    n_assets = target_weights.shape[1]

    actual_weights = np.zeros((n, n_assets))
    turnover = np.zeros(n)
    current_w = np.zeros(n_assets)

    ret_vals    = returns.values
    target_vals = target_weights.values

    for i in range(n):
        if i > 0:
            # drift by TODAY's return so that it produces correct end-of-day-i weights
            # used with shift(1) to earn tomorrow's return
            portfolio_ret_i = np.nansum(current_w * ret_vals[i])
            total_value     = 1.0 + portfolio_ret_i   # cash earns 0
            if total_value > 1e-8:
                current_w = current_w * (1 + ret_vals[i]) / total_value
            else:
                current_w = np.zeros(n_assets)

        # rebalance at EOM close aka initialize at day 0
        if i == 0 or rebalance_mask[i]:
            target = target_vals[i]
            to = np.nansum(np.abs(target - current_w))
            if to > 2.0 + 1e-6:
                print(f"  [WARN] Turnover {to:.3f} > 2.0 on {returns.index[i].date()} — "
                      f"check for NaN weights.")
            turnover[i] = to
            current_w = target.copy()

        actual_weights[i] = current_w

    # returns
    actual_weights_df = pd.DataFrame(
        actual_weights, index=returns.index, columns=returns.columns
    )

    # shift(1): end-of-day-i weights applied to day-(i+1) returns
    gross_returns = (actual_weights_df.shift(1) * returns).sum(axis=1)
    tc_costs      = pd.Series(turnover, index=returns.index) * tc
    net_returns   = gross_returns - tc_costs

    # diagnostics
    total_to = turnover[turnover > 0].sum()
    n_years  = n / 252
    ann_to   = total_to / n_years
    print(f"  Backtest complete: {n} days, {rebalance_mask.sum()} rebalances. "
        f"Annualized one-way turnover: {ann_to:.2f}x "
        f"(cumulative: {total_to:.1f}x over {n_years:.1f}yr). "
        f"Mean rebalance TC drag: {tc_costs[tc_costs > 0].mean() * 10_000:.1f} bps.")
    return net_returns