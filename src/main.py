# main.py
import pandas as pd
import os
from data_loader import load_data
from features import build_all_features
from momentum_strategy import generate_trend_signals, calculate_inverse_vol_weights
from ml_model import create_labels, train_and_predict_walk_forward, get_feature_importance
from backtest import run_backtest  # <--- THIS WAS MISSING
from evaluation import (
    calculate_metrics, 
    plot_performance, 
    plot_rolling_sharpe, 
    plot_feature_importance
)


def main():
    # Configuration
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'USO', 'ICLN']
    start_date = '2010-01-01'
    results_dir = 'results/figures/'
    
    # 1. Data Loading (Day 1)
    prices = load_data(tickers, start_date)
    # Save raw prices for modules that read from disk
    os.makedirs('data', exist_ok=True)
    prices.to_csv('data/raw_prices.csv')
    
    # 2. Feature Engineering (Day 2)
    features_df = build_all_features(prices)
    
    # 3. Strategy Construction (Day 3)
    # Generate signals based on 252-day momentum
    signals = generate_trend_signals(features_df, tickers)
    # Use Vol-Scaled weights as our "Base" strategy
    base_weights = calculate_inverse_vol_weights(signals, features_df, tickers)
    
    # 4. ML Regime Modeling
    # Create labels: 1 if positive returns over next 21 days
    labels = create_labels(prices, base_weights, horizon=21)

    # ==========================================
    # THE FIX: Proper Triple-Alignment
    # ==========================================
    # 1. Find the intersection of indices that exist in BOTH features and labels
    # This removes the early 'warm-up' period and the very end 'future' period.
    common_idx = features_df.index.intersection(labels.dropna().index)

    X_clean = features_df.loc[common_idx]
    y_clean = labels.loc[common_idx]

    print(f"Alignment complete. Training features shape: {X_clean.shape}, Labels shape: {y_clean.shape}")
    # ==========================================

    # Now continue with training
    rf_probs = train_and_predict_walk_forward(X_clean, y_clean, model_type='rf')
    lr_probs = train_and_predict_walk_forward(X_clean, y_clean, model_type='lr')
    
    # Forward fill to align with base weights index
    rf_regime = rf_probs.reindex(base_weights.index).ffill().fillna(0)
    lr_regime = lr_probs.reindex(base_weights.index).ffill().fillna(0)
    
    # Apply overlays (Scaling by probability of 'Good' regime)
    ml_rf_weights = base_weights.multiply(rf_regime, axis=0)
    ml_lr_weights = base_weights.multiply(lr_regime, axis=0)


    # 5. Backtesting (The Comparison Set)
    daily_rets = prices.pct_change().dropna()
    
    results = pd.DataFrame({
        'SPY_Benchmark': daily_rets['SPY'],
        'Base_Mom_VolScaled': run_backtest(daily_rets, base_weights),
        'ML_RF_Enhanced': run_backtest(daily_rets, ml_rf_weights),
        'ML_LR_Enhanced': run_backtest(daily_rets, ml_lr_weights)
    }).dropna()

    # 6. Final Evaluation
    plot_performance(results, 'results/figures/')
    plot_rolling_sharpe(results)
    
    # Feature Importance (RF only)
    feat_imp = get_feature_importance(X_clean, y_clean)
    plot_feature_importance(feat_imp, 'results/figures/')

if __name__ == "__main__":
    main()