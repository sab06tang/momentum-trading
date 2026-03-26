import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

SUPPORTED_MODELS = {"rf", "logistic"}
SUPPORTED_LABEL_TYPES = {"threshold", "median"}


def create_labels(
    prices: pd.DataFrame,
    momentum_weights: pd.DataFrame,
    horizon: int = 21,
    threshold: float = -0.02,
    label_type: str = "median",
) -> pd.Series:
    """
    Creates forward-looking binary labels based on momentum strategy performance.

    Label[t] = 1 if forward strategy return over [t+1, t+horizon] is "good"
               0 otherwise
               NaN for the last `horizon` rows (no forward data)

    Two labeling modes
    ------------------
    "threshold" : label=1 if forward_return > threshold (fixed cutoff)
                  Risk: severely imbalanced in trending markets (e.g. 96% positive
                  post-2010) → model learns to always predict favorable → filter
                  becomes vestigial.

    "median"    : label=1 if forward_return > expanding-window median (relative cutoff)
                  Expanding window: median at time t is computed only on returns up
                  to t — no lookahead. Always produces ~50/50 split regardless of
                  market regime. Model learns to distinguish *relatively* good from
                  *relatively* bad momentum periods.
                  Use this as the default.

    IMPORTANT: `momentum_weights` must be pre-ML momentum-only weights.
    Passing ML-filtered weights creates a circular dependency.

    Parameters
    ----------
    threshold  : used only when label_type="threshold"
    label_type : "median" (default, self-balancing) or "threshold" (fixed cutoff)
    """
    if label_type not in SUPPORTED_LABEL_TYPES:
        raise ValueError(
            f"label_type must be one of {SUPPORTED_LABEL_TYPES}, got '{label_type}'"
        )

    daily_returns = prices.pct_change()
    common_idx = daily_returns.index.intersection(momentum_weights.index)

    strat_returns = (
        momentum_weights.loc[common_idx]
        .multiply(daily_returns.loc[common_idx])
        .sum(axis=1)
    )

    # forward_returns[t] = cumulative strategy return from t+1 to t+horizon
    forward_returns = strat_returns.rolling(window=horizon).sum().shift(-horizon)

    if label_type == "threshold":
        labels = (
            (forward_returns > threshold)
            .astype(float)
            .where(forward_returns.notna())
        )

    else:  # "median" — expanding window, no lookahead
        # expanding().median() at time t uses only returns[0..t]
        # This means early observations have noisy medians (small sample),
        # but there is zero future leakage at any point in the series.
        expanding_median = forward_returns.expanding(min_periods=horizon * 2).median()
        labels = (
            (forward_returns > expanding_median)
            .astype(float)
            .where(forward_returns.notna() & expanding_median.notna())
        )

    pos_rate = labels.mean()
    imbalanced = pos_rate < 0.35 or pos_rate > 0.65
    print(
        f"  Label positive rate: {pos_rate:.1%}  "
        f"(type='{label_type}'"
        + (f", threshold={threshold:.1%}" if label_type == "threshold" else "")
        + f", horizon={horizon}d). "
        + ("[WARN] Severe imbalance — consider switching to label_type='median'"
           if imbalanced and label_type == "threshold"
           else "[WARN] Imbalance remains even with median split — check forward_returns for NaNs"
           if imbalanced and label_type == "median"
           else "OK")
    )
    return labels


def train_and_predict_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "rf",
    horizon: int = 21,
    n_splits: int = 10,
    test_size: int = 252,
) -> pd.Series:
    """
    Walk-forward prediction with TimeSeriesSplit.

    Key design decisions:
    - gap=horizon   : purges label-overlap leakage (labels use horizon-day fwd returns)
    - test_size=252 : ~1 year per fold — interpretable, realistic OOS periods
    - Scaler instantiated per fold: no parameter bleed across folds
    - First ~n/n_splits rows will have NaN predictions (never in any test set)

    Parameters
    ----------
    horizon   : must match horizon used in create_labels (drives gap parameter)
    test_size : rows per test fold (default 252 = ~1 trading year)
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"model_type must be one of {SUPPORTED_MODELS}, got '{model_type}'"
        )

    valid_mask = y.notna()
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    predictions = pd.Series(np.nan, index=X_clean.index, dtype=float)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon, test_size=test_size)

    fold_sizes = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train = y_clean.iloc[train_idx]
        fold_sizes.append(len(X_train))

        # Scaler is fold-local: no parameter bleed
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        if model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=3,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        else:  # logistic
            model = LogisticRegression(
                C=0.1,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )

        model.fit(X_train_scaled, y_train)
        predictions.iloc[test_idx] = model.predict_proba(X_test_scaled)[:, 1]

    n_no_pred = predictions.isna().sum()
    print(
        f"  Walk-forward complete [{model_type}]: {n_splits} folds, "
        f"train sizes {min(fold_sizes)}–{max(fold_sizes)} rows. "
        f"{n_no_pred} rows without predictions (initial warm-up — "
        f"backtest defaults to momentum-only for this period)."
    )
    return predictions


def get_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Diagnostic feature importance using training data only (first 80%).

    NOT used for any prediction — for interpretation only.
    Scaler applied to match the preprocessing used in walk-forward training.
    """
    valid = y.notna()
    X_clean, y_clean = X[valid], y[valid]

    cutoff = int(len(X_clean) * 0.80)
    X_train = X_clean.iloc[:cutoff]
    y_train = y_clean.iloc[:cutoff]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=4, random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    importance = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    print(f"  Top 5 features:\n{importance.head().to_string()}")
    return importance