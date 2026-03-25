import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Calculates annualized return, annualized volatility, Sharpe ratio, and Max Drawdown.
    """
    # Drop NaNs just in case
    returns = returns.dropna()
    n_days = len(returns)
    
    if n_days == 0:
        return {}
        
    # Cumulative Return
    cum_returns = (1 + returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    
    # Annualized Metrics (assuming 252 trading days)
    ann_return = (1 + total_return) ** (252 / n_days) - 1
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    
    # Maximum Drawdown
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns / rolling_max) - 1
    max_dd = drawdowns.min()
    
    return {
        'Total Return': f"{total_return * 100:.2f}%",
        'Annualized Return': f"{ann_return * 100:.2f}%",
        'Annualized Volatility': f"{ann_vol * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_dd * 100:.2f}%"
    }

def plot_performance(returns_df: pd.DataFrame, save_dir: str):
    """
    Generates and saves the Equity Curve and Drawdown plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Plot Equity Curve
    cum_returns = (1 + returns_df).cumprod() * 100 # Start at $100
    
    plt.figure(figsize=(12, 6))
    for col in cum_returns.columns:
        plt.plot(cum_returns.index, cum_returns[col], label=col, linewidth=1.5)
        
    plt.title('Strategy Ablation Study: Cumulative Equity Curve (Log Scale)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.yscale('log') # Log scale is standard for long-term quant plots
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'equity_curve.png'), dpi=300)
    plt.close()
    
    # 2. Plot Drawdowns
    plt.figure(figsize=(12, 4))
    for col in cum_returns.columns:
        rolling_max = cum_returns[col].cummax()
        drawdown = (cum_returns[col] / rolling_max) - 1
        plt.plot(drawdown.index, drawdown * 100, label=col, linewidth=1)
        
    plt.title('Strategy Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'drawdowns.png'), dpi=300)
    plt.close()

def plot_feature_importance(importance_series: pd.Series, save_dir: str):
    """
    Visualizes which features (Mom, Vol, Dispersion) drive the ML regime filter.
    """
    plt.figure(figsize=(10, 6))
    importance_series.head(15).plot(kind='barh', color='teal')
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.xlabel('Relative Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
    plt.close()

def plot_rolling_sharpe(returns_df: pd.DataFrame, window: int = 252):
    """
    Plots 1-year rolling Sharpe to show if ML improves stability.
    """
    # Annualized Rolling Sharpe: (Mean / Std) * sqrt(252)
    rolling_sharpe = (returns_df.rolling(window).mean() / returns_df.rolling(window).std()) * np.sqrt(252)
    
    plt.figure(figsize=(12, 6))
    for col in rolling_sharpe.columns:
        plt.plot(rolling_sharpe[col], label=col)
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title(f'Rolling {window}-Day Sharpe Ratio (Comparison)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/rolling_sharpe.png')
    plt.close()

if __name__ == "__main__":
    returns_path = 'data/backtest_returns.csv'
    figures_dir = 'results/figures/'
    
    if os.path.exists(returns_path):
        print("Loading backtest returns...")
        returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        # Calculate and print metrics for each strategy
        print("\n" + "="*50)
        print("PERFORMANCE METRICS (Ablation Study)")
        print("="*50)
        
        metrics_list = []
        for strat in returns_df.columns:
            metrics = calculate_metrics(returns_df[strat])
            metrics['Strategy'] = strat
            metrics_list.append(metrics)
            
        # Display as a clean DataFrame
        metrics_df = pd.DataFrame(metrics_list).set_index('Strategy')
        print(metrics_df.to_string())
        print("="*50 + "\n")
        
        # Generate plots
        print("Generating plots...")
        plot_performance(returns_df, figures_dir)
        print(f"Plots successfully saved to {figures_dir}")
        
    else:
        print("Error: backtest_returns.csv not found. Run backtest.py first.")