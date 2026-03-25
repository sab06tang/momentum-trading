import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def create_labels(prices: pd.DataFrame, eq_weights: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """
    Creates binary labels: 1 if the equal-weight momentum strategy generates 
    a positive return over the next `horizon` days, 0 otherwise.
    """
    print(f"Creating forward {horizon}-day return labels...")
    
    # Calculate daily returns of the assets
    daily_returns = prices.pct_change()
    
    # Calculate daily strategy returns (weights from T applied to returns on T+1)
    # Our features are already shifted, so eq_weights at T represents data up to T-1.
    # Therefore, we multiply eq_weights at T by daily_returns at T.
    strat_daily_returns = (eq_weights * daily_returns).sum(axis=1)
    
    # Calculate forward 21-day returns
    # .shift(-horizon) looks INTO THE FUTURE to create the label for today
    forward_returns = strat_daily_returns.rolling(window=horizon).sum().shift(-horizon)
    
    # Binarize: 1 if profitable, 0 if not
    labels = (forward_returns > 0).astype(int)
    
    # The last `horizon` days will be NaN for forward returns, so we mark them as NaN
    labels.iloc[-horizon:] = np.nan
    
    return labels

def train_and_predict_walk_forward(X: pd.DataFrame, y: pd.Series, model_type: str = 'rf'):
    """
    Enhanced to support both 'rf' and 'lr' with appropriate scaling.
    """
    print(f"Training walk-forward {model_type.upper()} model...")
    predictions = pd.Series(index=X.index, dtype=float)
    tscv = TimeSeriesSplit(n_splits=10) 
    scaler = StandardScaler()

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = y.iloc[train_index]
        
        # Fit scaler ONLY on training data to prevent leakage
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        else:
        # UPDATED: Remove penalty='l2' to follow the new scikit-learn API 
        # The default is already L2, and C=1.0 controls the strength.
            model = LogisticRegression(max_iter=1000, random_state=42)
            
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        predictions.iloc[test_index] = probs
        
    return predictions

def get_feature_importance(X: pd.DataFrame, y: pd.Series):
    """
    Helper for Day 7: Extracting what drives the model.
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y) # For global importance, fit on all valid data
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False)

if __name__ == "__main__":
    features_path = 'data/features.csv'
    prices_path = 'data/raw_prices.csv'
    weights_path = 'data/weights_equal.csv'
    
    if all(os.path.exists(p) for p in [features_path, prices_path, weights_path]):
        # Load data
        X = pd.read_csv(features_path, index_col=0, parse_dates=True)
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        eq_weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        
        # Align indices just in case
        common_idx = X.index.intersection(prices.index).intersection(eq_weights.index)
        X = X.loc[common_idx]
        prices = prices.loc[common_idx]
        eq_weights = eq_weights.loc[common_idx]
        
        # Create target variable (y)
        y = create_labels(prices, eq_weights, horizon=21)
        
        # Drop the last 21 days where we don't have future returns to train on
        valid_idx = y.dropna().index
        X_clean = X.loc[valid_idx]
        y_clean = y.loc[valid_idx]
        
        print(f"Training on {len(X_clean)} days of data...")
        
        # Train Random Forest
        rf_predictions = train_and_predict_walk_forward(X_clean, y_clean, model_type='rf')
        
        # Train Logistic Regression
        lr_predictions = train_and_predict_walk_forward(X_clean, y_clean, model_type='lr')
        
        # Save predictions
        preds_df = pd.DataFrame({
            'rf_regime_prob': rf_predictions,
            'lr_regime_prob': lr_predictions
        }, index=X_clean.index)
        
        # Forward fill the predictions to the original index length (so we have a prediction for the very last days to trade on tomorrow)
        preds_df = preds_df.reindex(X.index).ffill()
        
        preds_df.to_csv('data/ml_predictions.csv')
        print("ML Predictions saved to data/ml_predictions.csv")
        
    else:
        print("Error: Missing required CSV files in data/ directory. Run previous steps.")