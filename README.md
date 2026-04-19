# Cross-Asset Momentum Trading: Cross-Sectional, Time-Series, and ML Regime Filter

A quantitative trading framework comparing three systematic strategies across a
multi-asset universe: cross-sectional momentum, time-series momentum with a
trailing stop-loss, and a machine learning regime overlay. Built to
institutional standards: walk-forward validation, explicit signal lagging,
transaction cost modelling, and a clean ablation study.

---

## Results (January 2011 – March 2026)

| Strategy | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD | Calmar |
|:---|---:|---:|---:|---:|---:|---:|
| Equal-Weight Buy & Hold | 8.4% | 12.0% | 0.57 | 0.70 | -25.3% | 0.33 |
| SPY Buy & Hold | 13.4% | 17.1% | 0.70 | 0.81 | -33.7% | 0.40 |
| **Cross-Sectional Momentum** | **12.3%** | **13.2%** | **0.79** | **1.00** | **-24.8%** | **0.50** |
| Momentum + Logistic Reg. | 7.9% | 11.2% | 0.55 | 0.60 | -24.8% | 0.32 |
| Momentum + Random Forest | 8.9% | 11.9% | 0.60 | 0.68 | -24.8% | 0.36 |
| TS Momentum + Stop-Loss | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Sanity Check | 7.0% | 15.1% | 0.39 | 0.42 | -33.3% | 0.21 |

> **Key finding:** Cross-sectional momentum achieves the highest risk-adjusted
> performance (Sharpe 0.79, Sortino 1.00), outperforming SPY on every
> risk-adjusted metric and incurring a smaller maximum drawdown (−24.8% vs
> −33.7%). The ML regime filter reduces volatility but also reduces returns
> sufficiently to lower the Sharpe ratio, which is a valid and expected result in a
> predominantly trending market (discussed below). Time-series momentum results
> are pending a full backtest run.

---

## Strategy Architecture

Three strategies are compared in the ablation study:

1. **Cross-Sectional Momentum** — ranks assets relative to each other using composite z-scores across four lookback windows (1m/3m/6m/12m). Long the relatively strongest assets, flat on the weakest.
2. **Time-Series Momentum with Stop-Loss** — evaluates each asset independently; long if its own 12m return is positive. A 10% trailing stop-loss (based on 252-day rolling drawdown) overrides the signal to limit drawdown exposure.
3. **ML Regime Overlay** — applied on top of cross-sectional momentum; scales position sizes down when a classifier predicts an unfavorable forward return regime.

---

### Universe
Five liquid ETFs providing cross-asset exposure:

| Ticker | Asset Class | Role |
|:---|:---|:---|
| SPY | US Large-Cap Equity | Risk-on core |
| QQQ | US Technology Equity | High-momentum growth |
| TLT | Long-Duration Treasuries | Flight-to-quality / diversifier |
| GLD | Gold | Inflation hedge / tail risk |
| USO | Crude Oil | Commodity momentum |

### Signal Generation

**Cross-sectional momentum:** Composite score averaging cross-sectional z-scores
across four lookback windows (1m / 3m / 6m / 12m). Cross-sectional z-scoring at
each date ensures the signal captures *relative* momentum strength across assets,
not just absolute direction. Binary long/flat signal: long when composite z-score > 0,
flat (cash) otherwise.

**Time-series momentum:** Each asset is evaluated independently — long if its own
12m return is positive, flat otherwise. A 10% trailing stop-loss based on the
252-day rolling drawdown overrides the momentum signal to exit positions during
sharp drawdowns, regardless of the trend signal.

### Position Sizing
Inverse-volatility weighting using 63-day realised volatility, capped at 40%
per asset to prevent excessive concentration in low-vol assets (typically TLT).
Weights are re-normalised after capping; residual allocation sits in cash.

### ML Regime Overlay
A binary classifier predicts whether the momentum strategy's forward 21-day
return will exceed the expanding-window historical median. When the model
signals an unfavorable regime, position sizes are scaled to 50% or 0% of their
momentum-only values.

Two models trained in parallel:
- **Logistic Regression** — interpretable baseline with L2 regularisation
- **Random Forest** — nonlinear benchmark (200 trees, max depth 3,
  balanced class weights)

Both trained with strict **walk-forward (time-series) cross-validation** —
no shuffling, no future data in any training fold.

### Rebalancing & Execution
- Monthly rebalancing at business month-end close
- Daily weight drift between rebalances (weights evolve with market moves)
- 10 bps one-way transaction cost applied at each rebalance
- 1-day execution lag: signals computed at close of day *t*, positions entered
  at close of day *t+1*

---

## Methodology

### Lookahead Bias Prevention
Every layer of the pipeline enforces strict temporal separation:

```
prices[t]           → features[t]       (data through close of day t only)
features[t]         → signals[t]        (momentum score using features[t])
signals[t]          → weights[t]        (inv-vol sizing at close of day t)
weights[t]          → label[t]          (fwd return t+1..t+21 — no overlap)
features[t]+label[t]→ ml_probs[t]       (walk-forward OOS only)
weights[t]×regime[t]→ portfolio[t]      
portfolio[t]        → return[t+1]       (shift(1) execution lag in backtest)
```

No row in any training set contains information from its own test window.

### ML Validation: Walk-Forward with Gap Purging
`TimeSeriesSplit(n_splits=10, gap=21, test_size=252)` — the `gap=21` parameter
is critical. Without it, the last 21 training labels overlap with the test
period (they include returns from days *inside* the test window), creating
direct data leakage. The gap purges this overlap exactly.

Each test fold covers approximately one calendar year, giving 10 independent
out-of-sample evaluation periods across the full history.

### Label Construction: Expanding-Window Median Split
Labels use an **expanding-window median** rather than a fixed return threshold.
A fixed threshold (e.g. −2%) produces severe class imbalance in trending
markets (>95% positive labels in a post-2010 bull market), causing the model to
learn a degenerate always-positive predictor. The expanding median adapts to
each market regime and consistently produces ~50/50 class balance
(observed: 53% positive), enabling genuine discriminative learning.

### Feature Engineering
36 features across four categories:

| Category | Features |
|:---|:---|
| Momentum | Rolling returns at 21d, 63d, 126d, 252d × 5 assets = 20 features |
| Moving Average | 50d/200d MA distance × 5 assets = 5 features |
| Volatility | 21d and 63d annualised vol × 5 assets = 10 features |
| Drawdown | 252d drawdown × 5 assets = 5 features |
| Cross-asset | 21d rolling dispersion of daily returns = 1 feature |

All features computed with `min_periods` equal to the full window length —
no partial-window values during warmup. The top five features by RF importance:
GLD MA distance, QQQ 63d volatility, GLD 12m momentum, QQQ 1m momentum,
SPY 1y drawdown.

---

## Ablation Study: Isolating the ML Contribution

The ablation is ordered by increasing complexity:

```
Random (sanity floor)   →  Sharpe 0.39
Equal-weight passive    →  Sharpe 0.57  (+0.18 vs random)
SPY buy & hold          →  Sharpe 0.70
Momentum only           →  Sharpe 0.79  (+0.22 vs SPY — momentum premium)
Momentum + Logistic     →  Sharpe 0.55  (−0.24 vs momentum only)
Momentum + RF           →  Sharpe 0.60  (−0.19 vs momentum only)
```

**Interpretation:** The ML overlay does not add value over the 2011–2026 sample
period. This is likely because of the following three reasons:

1. **Bull market bias.** The ML filter reduces gross exposure (active 83–94% of
   days). In a period where momentum almost always pays, reducing exposure
   mechanically reduces returns without a commensurate reduction in drawdown
   (max drawdown is identical across all momentum variants at −24.8%, because
   drawdowns occur within regimes labelled "favorable" by the model).

2. **Regime signal lead time.** A 21-day forward return label trained to predict
   the *median* of the distribution is a difficult classification problem in
   near-efficient markets. The model has low predictive accuracy OOS, and
   a noisy filter hurts more than it helps.

3. **Transaction cost asymmetry.** Regime-scaled weights change the rebalance
   target at each monthly reset. Even with 5-day probability smoothing, the ML
   strategies incur approximately 15× more rebalance turnover than the
   momentum-only strategy (~7x vs ~0.5x annualised one-way).

Based on the above analysis, I conclude that **the momentum signal itself is the alpha
source**, and that ML regime filtering requires either a longer history spanning
multiple full cycles, or a better-specified prediction target (e.g. tail-risk
drawdown events rather than median forward return).

---

## Project Structure

```
momentum-trading/
├── src/
│   ├── data_loader.py          # yfinance download, adjusted prices, validation
│   ├── features.py             # momentum, MA, vol, drawdown, dispersion features
│   ├── momentum_strategy.py    # cross-sectional z-score signals, inv-vol weighting
│   ├── ts_momentum_strategy.py # time-series momentum with trailing stop-loss
│   ├── ml_model.py             # label creation, walk-forward training, importances
│   ├── backtest.py             # vectorised backtest, daily drift, TC modelling
│   ├── evaluation.py           # metrics, ablation table, all visualisations
│   └── main.py                 # pipeline orchestration
└── results/
    └── figures/
        ├── equity_curve.png
        ├── drawdowns.png
        ├── rolling_sharpe.png
        ├── regime_visualization.png
        └── feature_importance.png
```

---


## Key Design Decisions

**Cross-sectional vs. time-series momentum — what's the difference?**  
Cross-sectional momentum asks "which asset is strongest *relative to the others*?" and always holds the relatively best assets. Time-series momentum asks "is this asset trending *in absolute terms*?" — it's possible for all assets to be simultaneously long or simultaneously flat. The stop-loss in the time-series variant adds a third override: exit even a trending asset if it has already fallen 10% from its recent high.

**Why long/flat, not long/short?**  
Shorting individual asset ETFs introduces significant borrow costs and
diverges from how most institutional cross-asset momentum strategies are
implemented. Long/flat with cash better represents the strategy's risk profile
and avoids massive drag during equity bull markets.

**Why inverse-volatility weighting?**  
Inverse-vol normalises each position's risk contribution without requiring a
full covariance matrix estimate (which is noisy in a 5-asset universe).

**Why composite z-score across lookbacks?**  
Single-lookback momentum is sensitive to the chosen window. Averaging
cross-sectional z-scores across 1m/3m/6m/12m is more robust and reflects
the standard multi-horizon approach in the academic momentum literature.

**Why expanding-window median labels?**  
A fixed return threshold produces severely imbalanced labels in trending
markets, causing degenerate classifiers. The expanding median self-calibrates
to the current market regime while maintaining strict temporal separation.

---

## Limitations and Future Work

- **Sample period:** 2011–2026 is predominantly a US equity bull market.
  Performance across a full cycle (including 2000–2002, 2007–2009) would
  require extending the universe to indices with longer histories.

- **Survivorship bias:** The five tickers were selected with knowledge of their
  existence through 2026. A production implementation would use a point-in-time
  universe construction.

- **ML signal quality:** The regime classifier's predictive accuracy is not
  reported. Adding precision/recall curves and calibration plots per fold would
  better characterise where the model adds and destroys value.

- **Alternative ML targets:** Predicting tail-risk events (e.g. rolling max
  drawdown > 5% in next 21 days) rather than the median return split may
  produce a filter that adds value even in trending markets by protecting
  specifically against sharp drawdowns.

- **Transaction cost sensitivity:** A TC sweep (0–30 bps) would show the
  breakeven cost at which each strategy becomes unviable — relevant for
  comparing futures vs. ETF implementation.