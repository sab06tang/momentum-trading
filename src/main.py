import pandas as pd
import numpy as np
import os

from data_loader import load_data
from features import build_all_features
from momentum_strategy import (
    generate_trend_signals,
    calculate_inverse_vol_weights,
    generate_equal_weight_baseline,
    generate_random_strategy,
)
from ml_model import create_labels, train_and_predict_walk_forward, get_feature_importance
from backtest import run_backtest
from evaluation import (
    compare_strategies,
    plot_performance,
    plot_rolling_sharpe,
    plot_regime_visualization,
    plot_feature_importance,
)

# configuration
TICKERS        = ["SPY", "QQQ", "TLT", "GLD", "USO"]
START_DATE     = "2010-01-01"
RESULTS_DIR    = "results/figures/"
RISK_FREE_RATE = 0.02

LABEL_HORIZON    = 21
LABEL_THRESHOLD  = -0.02      # only used when LABEL_TYPE = "threshold"
LABEL_TYPE       = "median"   # ← NEW: "median" self-balances; avoids 96% pos rate
TC               = 0.001
ML_PROB_THRESHOLD    = 0.50
ML_PROB_PARTIAL      = 0.35
ML_SMOOTHING_WINDOW  = 5
MAX_ASSET_WEIGHT     = 0.40


def build_ml_regime(
    probs: pd.Series,
    target_index: pd.Index,
    high: float = ML_PROB_THRESHOLD,
    low: float  = ML_PROB_PARTIAL,
    smooth: int = ML_SMOOTHING_WINDOW,
) -> pd.Series:
    """
    Converts raw walk-forward probabilities into a {0, 0.5, 1.0} regime scalar.

    Warmup rows (NaN predictions) default to 1.0 — full momentum exposure.
    Smoothing window reduces transaction cost drag from daily regime flips.
    """
    filled = probs.reindex(target_index).ffill().fillna(1.0)  # warmup → full exposure
    smoothed = filled.rolling(window=smooth, min_periods=1).mean()
    regime = pd.Series(
        np.where(smoothed > high, 1.0, np.where(smoothed > low, 0.5, 0.0)),
        index=smoothed.index,
    )
    on_pct = (regime > 0).mean()
    print(f"  Regime filter active {on_pct:.1%} of days "
          f"({(regime == 1.0).mean():.1%} full / "
          f"{(regime == 0.5).mean():.1%} half / "
          f"{(regime == 0.0).mean():.1%} off)")
    return regime


def main() -> None:

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # data
    prices = load_data(TICKERS, START_DATE)

    # features
    features_daily = build_all_features(prices)

    # momentum signals & base weights
    signals      = generate_trend_signals(features_daily, TICKERS)
    base_weights = calculate_inverse_vol_weights(
        signals, features_daily, TICKERS, max_weight=MAX_ASSET_WEIGHT
    )

    # ML labels & walk-forward training
    labels = create_labels(
    prices, base_weights,
    horizon=LABEL_HORIZON,
    threshold=LABEL_THRESHOLD,
    label_type=LABEL_TYPE,        # ← NEW
    )

    common_idx = features_daily.index.intersection(labels.dropna().index)
    X_ml = features_daily.loc[common_idx]
    y_ml = labels.loc[common_idx]
    print(f"\nML training set: {len(X_ml)} days "
          f"({X_ml.index[0].date()} → {X_ml.index[-1].date()})")

    rf_probs = train_and_predict_walk_forward(
        X_ml, y_ml, model_type="rf",       horizon=LABEL_HORIZON
    )
    lr_probs = train_and_predict_walk_forward(
        X_ml, y_ml, model_type="logistic", horizon=LABEL_HORIZON
    )

    # ML regime scaling
    print("\nBuilding ML regime signals...")
    rf_regime = build_ml_regime(rf_probs, base_weights.index)
    lr_regime = build_ml_regime(lr_probs, base_weights.index)

    ml_rf_weights = base_weights.multiply(rf_regime, axis=0)
    ml_lr_weights = base_weights.multiply(lr_regime, axis=0)

    # backtests
    # trim daily_rets to the first date weights are available
    # base_weights starts after the 252-day feature warmup (~1 year)
    # without this, run_backtest raises ValueError on leading NaN weightse
    weights_start = base_weights.index[0]
    daily_rets = prices.pct_change().dropna().loc[weights_start:]

    print("\nRunning backtests...")

    results_dict = {
        "1_EqualWeight_BuyHold": run_backtest(
            daily_rets,
            generate_equal_weight_baseline(daily_rets.index, TICKERS),
            tc=TC,
        ),
        "2_SPY_BuyHold":    daily_rets["SPY"],
        "3_Momentum_Only":  run_backtest(daily_rets, base_weights, tc=TC),
        "4_Mom_LogReg":     run_backtest(daily_rets, ml_lr_weights, tc=TC),
        "5_Mom_RF":         run_backtest(daily_rets, ml_rf_weights, tc=TC),
        "6_Random_Sanity":  run_backtest(
            daily_rets,
            generate_random_strategy(daily_rets.index, TICKERS),
            tc=TC,
        ),
    }
    results_df = pd.DataFrame(results_dict)

    ml_start = results_df.apply(lambda c: c.first_valid_index()).max()
    print(f"\n[INFO] Full history:    {results_df.index[0].date()} → "
          f"{results_df.index[-1].date()}")
    print(f"[INFO] ML-comparable window starts: {ml_start.date()} "
          f"({(results_df.index >= ml_start).sum()} trading days)")

    results_aligned = results_df.loc[ml_start:]

    # evaluation
    print("\nGenerating plots...")
    plot_performance(results_aligned, RESULTS_DIR)
    plot_rolling_sharpe(results_aligned, RESULTS_DIR, risk_free_rate=RISK_FREE_RATE)
    plot_regime_visualization(prices["SPY"], rf_regime, RESULTS_DIR)
    plot_feature_importance(get_feature_importance(X_ml, y_ml), RESULTS_DIR)

    # performance summary
    print("\n" + "=" * 90)
    print(f"PERFORMANCE SUMMARY  (ML window: {ml_start.date()} → "
          f"{results_aligned.index[-1].date()})")
    print("=" * 90)
    summary = compare_strategies(
        {col: results_aligned[col] for col in results_aligned.columns},
        risk_free_rate=RISK_FREE_RATE,
    )
    try:
        print(summary.to_markdown())
    except ImportError:
        print(summary.to_string())

    print("\n" + "=" * 90)
    print(f"FULL HISTORY  ({results_df.index[0].date()} → {results_df.index[-1].date()}) "
          f"— non-ML strategies only")
    print("=" * 90)
    non_ml_cols = ["1_EqualWeight_BuyHold", "2_SPY_BuyHold", "3_Momentum_Only"]
    summary_full = compare_strategies(
        {col: results_df[col].dropna() for col in non_ml_cols},
        risk_free_rate=RISK_FREE_RATE,
    )
    try:
        print(summary_full.to_markdown())
    except ImportError:
        print(summary_full.to_string())


if __name__ == "__main__":
    main()