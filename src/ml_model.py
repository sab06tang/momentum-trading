import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

SUPPORTED_MODELS = {"rf", "logistic"}


def create_labels(
    prices: pd.DataFrame,
    momentum_weights: pd.DataFrame,
    horizon: int = 21,
    threshold: float = -0.02,
) -> pd.Series:
    """
    Creates forward-looking binary labels based on momentum strategy performance.

    Label[t] = 1 if sum of strategy daily returns over [t+1, t+horizon] > threshold
               0 otherwise, NaN for the last `horizon` rows (no forward data)

    IMPORTANT: `momentum_weights` must be pre-ML momentum-only weights.
    Passing ML-filtered weights creates a circular dependency.

    Parameters
    ----------
    threshold : cumulative return floor; default -2% (label=1 means "not a crash regime")
    """
    daily_returns = prices.pct_change()
    common_idx = daily_returns.index.intersection(momentum_weights.index)

    strat_returns = (
        momentum_weights.loc[common_idx]
        .multiply(daily_returns.loc[common_idx])
        .sum(axis=1)
    )

    # forward_returns[t] = cumulative strategy return from t+1 to t+horizon
    forward_returns = strat_returns.rolling(window=horizon).sum().shift(-horizon)

    labels = (forward_returns > threshold).astype(float).where(forward_returns.notna())

    pos_rate = labels.mean()
    print(f"  Label positive rate: {pos_rate:.1%}  "
          f"(threshold={threshold:.1%}, horizon={horizon}d). "
          f"{'[WARN] Severe imbalance' if pos_rate < 0.3 or pos_rate > 0.85 else 'OK'}")
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
    - gap=horizon: purges label-overlap leakage (labels use horizon-day fwd returns)
    - test_size=252: ~1 year per fold — interpretable, realistic OOS periods
    - Scaler instantiated per fold: no parameter bleed across folds
    - First ~n/n_splits rows will have NaN predictions (never in test set)

    Parameters
    ----------
    horizon   : must match horizon used in create_labels (drives gap parameter)
    test_size : rows per test fold (default 252 = ~1 trading year)
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"model_type must be one of {SUPPORTED_MODELS}, got '{model_type}'")

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
                min_samples_leaf=10,   # prevents single-sample leaves on small folds
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        else:  # logistic
            model = LogisticRegression(
                C=0.1,                  # L2 regularization; default C=1 often overfit on small folds
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )

        model.fit(X_train_scaled, y_train)
        predictions.iloc[test_idx] = model.predict_proba(X_test_scaled)[:, 1]

    n_no_pred = predictions.isna().sum()
    print(f"  Walk-forward complete [{model_type}]: {n_splits} folds, "
          f"train sizes {min(fold_sizes)}–{max(fold_sizes)} rows. "
          f"{n_no_pred} rows without predictions (initial warm-up — "
          f"backtest defaults to momentum-only for this period).")
    return predictions


def get_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Diagnostic feature importance using training data only (first 80%).

    NOT used for any prediction — for interpretation only.
    Uses full-history training data up to the 80% cutoff to match the
    information available in the walk-forward training window.
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