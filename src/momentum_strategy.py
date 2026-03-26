import pandas as pd
import numpy as np


def calculate_momentum_score(features: pd.DataFrame, assets: list) -> pd.DataFrame:
    """
    Composite momentum score: average cross-sectional z-score across 1m, 3m, 6m, 12m.

    Cross-sectional z-scoring at each date means the signal captures relative
    momentum strength (which assets are strongest) not just absolute direction.
    Returns raw scores — not yet binarized.
    """
    windows = [21, 63, 126, 252]
    missing = [
        f"{a}_mom_{w}d"
        for a in assets for w in windows
        if f"{a}_mom_{w}d" not in features.columns
    ]
    if missing:
        raise KeyError(f"Missing momentum columns in features: {missing}")

    score_chunks = []
    for w in windows:
        mom_cols = [f"{a}_mom_{w}d" for a in assets]
        mom = features[mom_cols].copy()
        mom.columns = assets
        cross_mean = mom.mean(axis=1)
        cross_std = mom.std(axis=1).replace(0, np.nan)
        z = mom.sub(cross_mean, axis=0).div(cross_std, axis=0)
        score_chunks.append(z)

    composite = pd.concat(score_chunks).groupby(level=0).mean()
    return composite.reindex(features.index)


def generate_trend_signals(features: pd.DataFrame, assets: list) -> pd.DataFrame:
    """
    Binary long/flat signals derived from composite momentum score.
    Long (1) if composite z-score > 0, Flat/Cash (0) otherwise.

    NOTE: `features` must already be lagged by the caller before passing in.
    """
    scores = calculate_momentum_score(features, assets)
    signals = (scores > 0).astype(int)
    return signals


def calculate_inverse_vol_weights(
    signals: pd.DataFrame,
    features: pd.DataFrame,
    assets: list,
    max_weight: float = 0.40,
) -> pd.DataFrame:
    """
    Inverse-volatility position sizing using 63d realized vol.

    Weights are capped at max_weight per asset and re-normalized.
    When all signals are 0, all weights are 0 (100% cash).

    Parameters
    ----------
    max_weight : per-asset cap before renormalization (default 40%)
    """
    missing_vols = [f"{a}_vol_63d" for a in assets if f"{a}_vol_63d" not in features.columns]
    if missing_vols:
        raise KeyError(f"Missing volatility columns: {missing_vols}")

    raw = pd.DataFrame(index=signals.index, columns=assets, dtype=float)
    for asset in assets:
        inv_vol = 1.0 / (features[f"{asset}_vol_63d"] + 1e-6)
        raw[asset] = signals[asset] * inv_vol

    row_sums = raw.sum(axis=1)
    weights = raw.div(row_sums.where(row_sums != 0, other=np.nan), axis=0).fillna(0.0)

    # Cap and renormalize
    weights = weights.clip(upper=max_weight)
    row_sums_post = weights.sum(axis=1)
    weights = weights.div(
        row_sums_post.where(row_sums_post != 0, other=np.nan), axis=0
    ).fillna(0.0)

    high_conc = (weights.max(axis=1) > 0.50).sum()
    if high_conc > 0:
        print(f"  [WARN] {high_conc} dates with single-asset weight > 50% post-cap. "
              f"Review vol inputs.")

    return weights


def generate_equal_weight_baseline(index: pd.Index, assets: list) -> pd.DataFrame:
    """
    Passive equal-weight long-only benchmark.
    Primary ablation baseline — momentum must beat this before ML is justified.
    """
    return pd.DataFrame(1.0 / len(assets), index=index, columns=assets)


def generate_random_strategy(
    index: pd.Index,
    assets: list,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Coin-flip benchmark (sanity floor only — not a valid ablation baseline).
    Uses isolated RNG to avoid polluting global numpy/sklearn random state.
    """
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(
        rng.integers(0, 2, size=(len(index), len(assets))).astype(float),
        index=index,
        columns=assets,
    )
    row_sums = raw.sum(axis=1).replace(0, np.nan)
    return raw.div(row_sums, axis=0).fillna(0.0)