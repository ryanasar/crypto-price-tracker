# Crypto Price Predictor

Idea: build an ML pipeline to predict crypto
price movements. The goal is to learn ML engineering and quant finance, not to make money (unless it works well haha).

**Stack:** Python, pandas, XGBoost, ccxt
**Data:** 4 years of hourly BTC/USD and ETH/USD from Coinbase

---

## Repo layout

```
src/
  data_collection.py   # fetch OHLCV from Coinbase
  features.py          # engineered features
  targets.py           # prediction labels
  train.py             # walk-forward XGBoost training
  backtest.py          # strategy simulation + metrics
  main.py              # run the whole pipeline
data/                  # (gitignored) raw parquet files
```

# v1 — XGBoost baseline

## What it does

Predicts whether BTC/ETH price will move up in the next hour using
several calculated features from OHLCV data. Trains one model for
magnitude and one for direction,
then combines them into a trading signal.

## Architecture

1. **Data collection** — pulls hourly OHLCV from Coinbase via ccxt,
   saves to Parquet, validates for gaps.
2. **Features** — ~25 features per asset: multi-horizon returns,
   rolling volatility, volume ratios, RSI/MACD/Bollinger, and cyclical
   time encodings. All calculated specific to asset, then pooled.
3. **Targets** — binary labels on next-hour return. Two versions:
   magnitude (`return > 0.2%`) and direction (`return > 0`).
4. **Training** — XGBoost classifier, walk-forward CV with 24 monthly
   folds and an anchored 2-year initial training window.
5. **Backtest** — long-only with hysteresis thresholds, 0.2% per-side
   fees, proper position shift alignment to prevent look-ahead bias.

## Results
### Without Fees
```
Magnitude AUC: 0.6176
Direction AUC: 0.5429

=== BTC_USD ===
Total return:        24.06%
Annualized ret:      11.50%
Annualized vol:       8.99%
Sharpe:               1.279
Max drawdown:        -5.73%
Calmar:               2.007
Num trades:             176
Fraction of time long:    1.53%
Win rate:            58.87%
Avg win:            0.5590%
Avg loss:          -0.5949%
Expectancy:         0.0844%

  Buy-and-hold baseline:
  Total return:      28.57%
  Annualized ret:    13.53%
  Annualized vol:    47.63%
  Sharpe:             0.284
  Max drawdown:     -50.21%

=== ETH_USD ===
Total return:        -8.74%
Annualized ret:      -4.51%
Annualized vol:      15.87%
Sharpe:              -0.284
Max drawdown:       -18.92%
Calmar:              -0.239
Num trades:             314
Fraction of time long:    3.15%
Win rate:            55.21%
Avg win:            0.5675%
Avg loss:          -0.7266%
Expectancy:        -0.0121%

  Buy-and-hold baseline:
  Total return:     -22.21%
  Annualized ret:   -11.91%
  Annualized vol:    68.47%
  Sharpe:            -0.174
  Max drawdown:     -65.33%
```

### With Fees (0.2%)
```
Magnitude AUC: 0.5762
Direction AUC: 0.5429

=== BTC_USD ===
Total return:        -9.61%
Annualized ret:      -4.97%
Annualized vol:       8.09%
Sharpe:              -0.615
Max drawdown:       -16.14%
Calmar:              -0.308
Num trades:             166
Fraction of time long:    1.57%
Win rate:            53.11%
Avg win:            0.4684%
Avg loss:          -0.5397%
Expectancy:        -0.0042%

  Buy-and-hold baseline:
  Total return:      28.57%
  Annualized ret:    13.53%
  Annualized vol:    47.63%
  Sharpe:             0.284
  Max drawdown:     -50.21%

=== ETH_USD ===
Total return:       -29.36%
Annualized ret:     -16.10%
Annualized vol:      12.84%
Sharpe:              -1.254
Max drawdown:       -31.03%
Calmar:              -0.519
Num trades:             224
Fraction of time long:    2.17%
Win rate:            53.32%
Avg win:            0.4894%
Avg loss:          -0.6833%
Expectancy:        -0.0581%

  Buy-and-hold baseline:
  Total return:     -22.21%
  Annualized ret:   -11.91%
  Annualized vol:    68.47%
  Sharpe:            -0.174
  Max drawdown:     -65.33%
```

## Reflection

**What worked:** walk-forward validation kept me honest about
generalization, and training two models separately let me see that
the 0.617 AUC was mostly volatility forecasting, not directional
prediction (up/down ratio ~1.04 across all volatility quartiles).

**What didn't:** the signal isn't strong enough to overcome retail
fees. At an estimated 0.2% per side fee, expected return is below the cost. No threshold config made it profitable.

**What I learned:**
- Model AUC doesn't directly translate to trading profit
- Fees are the hardest part of retail crypto trading
- Writing my own backtest caught bugs (like an 801M% return from a
  bad merge)
- Normalization is crucial — MACD in dollar units nearly broke the
  model until I normalized it by close price

**Limitations:**
- Only OHLCV-derived features (no order book, funding, on-chain)
- Long-only, no leverage
- Thresholds chosen by inspection rather than formal tuning

**Next:**
- Add more features, not beyond just from OHLCV data
- Make a better trading strategy that chooses size as well
- More coins to train and predict
    - Maybe grouped by size/type

---

# v2

_(In progress)_

---
