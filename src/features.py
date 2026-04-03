import pandas as pd
import numpy as np


def calculate_momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling returns at 1m, 3m, 6m, 12m horizons (21, 63, 126, 252 trading days).
    No shift applied here, caller is responsible for lagging before use.
    """
    chunks = []
    windows = [21, 63, 126, 252]  # 1m, 3m, 6m, 12m

    for w in windows:
        ret = prices.pct_change(periods=w)
        ret.columns = [f"{col}_mom_{w}d" for col in prices.columns]
        chunks.append(ret)

    ma_50 = prices.rolling(50).mean()
    ma_200 = prices.rolling(200).mean()
    ma_dist = (ma_50 / ma_200) - 1
    ma_dist.columns = [f"{col}_ma_dist" for col in prices.columns]
    chunks.append(ma_dist)

    return pd.concat(chunks, axis=1)


def calculate_risk_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Annualized rolling volatility (21d, 63d) and 1-year drawdown.
    min_periods=252 on drawdown ensures feature is only defined over a full year.
    No shift applied here — caller is responsible for lagging before use.
    """
    daily_returns = prices.pct_change()
    chunks = []

    for w in [21, 63]:
        vol = daily_returns.rolling(window=w).std() * np.sqrt(252)
        vol.columns = [f"{col}_vol_{w}d" for col in prices.columns]
        chunks.append(vol)

    # min_periods=252: no partial-window drawdown during warmup
    rolling_max = prices.rolling(window=252, min_periods=252).max()
    drawdown = (prices / rolling_max) - 1
    drawdown.columns = [f"{col}_dd_252d" for col in prices.columns]
    chunks.append(drawdown)

    return pd.concat(chunks, axis=1)


def calculate_cross_asset_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional return dispersion: rolling mean of daily cross-asset return std.
    Captures regime transitions (trending vs. mean-reverting environments).
    No shift applied here — caller is responsible for lagging before use.
    """
    daily_returns = prices.pct_change()
    dispersion = daily_returns.std(axis=1).rolling(window=21).mean()
    return pd.DataFrame({"market_dispersion_21d": dispersion}, index=prices.index)


def build_all_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compiles all features. Returns unshifted features aligned to their natural date.

    Features at row t use price data up to and including day t.
    Callers MUST shift by the appropriate lag before using as predictors
    (e.g., .shift(1) in a daily backtest, or resample-then-shift(1) in monthly).

    Drops warmup rows (first ~252 days) where any feature is NaN.
    """
    print("Building features (Momentum, MAs, Risk, Dispersion)...")

    raw = pd.concat([
        calculate_momentum_features(prices),
        calculate_risk_features(prices),
        calculate_cross_asset_features(prices),
    ], axis=1)

    # guard against inf values from adjusted price edge cases
    raw = raw.replace([np.inf, -np.inf], np.nan)

    n_before = len(raw)
    features = raw.dropna(how="any")
    print(f"  Dropped {n_before - len(features)} warmup rows. "
          f"Usable days: {len(features)} "
          f"(from {features.index[0].date()})")

    # warn if non-contiguous index (mid-series gap — would break downstream .shift())
    gaps = features.index.to_series().diff().dt.days.dropna()
    large_gaps = gaps[gaps > 5]
    if not large_gaps.empty:
        print(f"  [WARN] {len(large_gaps)} gaps >5 days in feature index. "
              f"Downstream .shift() will be misaligned across these gaps.")

    return features