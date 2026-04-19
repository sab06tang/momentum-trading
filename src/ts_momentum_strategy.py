import pandas as pd
import numpy as np


def generate_timeseries_momentum_signals(
    features: pd.DataFrame,
    assets: list,
    stop_loss: float = 0.10,
) -> pd.DataFrame:
    """
    Time-series momentum: long if 12m return > 0 for each asset independently.
    Unlike cross-sectional momentum, each asset is evaluated in absolute terms,
    not relative to the other assets.

    Stop-loss: if an asset is more than stop_loss% below its 252-day rolling high
    (i.e., its dd_252d feature < -stop_loss), force that asset's signal to 0
    regardless of the 12m return signal.

    NOTE: features must be the same unshifted features_daily passed elsewhere;
    the backtest engine applies the execution lag via shift(1).
    """
    mom_cols = [f"{a}_mom_252d" for a in assets]
    dd_cols  = [f"{a}_dd_252d"  for a in assets]
    missing  = [c for c in mom_cols + dd_cols if c not in features.columns]
    if missing:
        raise KeyError(f"Missing columns in features: {missing}")

    base_signals = (features[mom_cols] > 0).astype(int)
    base_signals.columns = assets

    drawdowns = features[dd_cols].copy()
    drawdowns.columns = assets
    stopped_out = drawdowns < -stop_loss

    signals = base_signals.where(~stopped_out, other=0)

    stop_rate = stopped_out.any(axis=1).mean()
    gross_exp  = signals.mean(axis=1).mean()
    print(f"  TS Momentum (stop={stop_loss:.0%}): stop active on {stop_rate:.1%} of days. "
          f"Mean gross exposure: {gross_exp:.1%}")
    return signals
