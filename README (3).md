# Gap Prediction Dataset — Tech Stocks 2021–2023

This dataset was built for a machine learning project at NYCU (National Yang Ming Chiao Tung University), Spring 2026. The goal is to predict whether a stock will open with a significant gap the next morning — meaning the opening price moves more than 1% away from the previous close.

This is different from just predicting if a stock goes up or down. Gaps are caused by things that happen when the market is closed (earnings, macro news, geopolitical events), so they're harder to predict and more intersting from a financial point of view.

---

## What's in this dataset

The dataset covers **10 major tech stocks** over a **3-year period** (February 2021 to December 2023) :

```
AAPL, MSFT, NVDA, GOOGL, META, TSLA, AMD, INTC, ASML, TSM
```

We chose this time window on purpose — it includes the 2021 bull run, the 2022 crash driven by Fed rate hikes, and the 2023 recovery. So the model sees different market regimes during training, which makes it more robust.

Total rows after cleaning : **~7300** (one row per stock per trading day)

---

## How the data was collected

Raw OHLCV data (Open, High, Low, Close, Volume) was downloaded using the `yfinance` Python library, which is free and pulls data directly from Yahoo Finance.

From there we computed 16 technical indicators as features. The first few rows of each stock were dropped because indicators like MACD or RSI need some past data to compute (warmup period) — this is standard practice.

---

## Features (16 columns)

### Standard technical indicators
| Feature | What it captures |
|---|---|
| `rsi_14` | Is the stock overbought or oversold ? |
| `macd` | Trend direction (short vs long-term EMA) |
| `macd_signal` | Trigger line for MACD crossovers |
| `macd_hist` | Strength of the trend move |
| `bb_pct_b` | Where price sits inside Bollinger Bands (0 = bottom, 1 = top) |
| `atr_14` | How much price moves on average (volatility, not direction) |
| `stoch_k` | Where close sits relative to recent high/low range |
| `stoch_d` | Smoothed version of stoch_k |
| `return_1d` | Daily return (yesterday's close to today's close) |
| `return_5d` | 5-day momentum |
| `volatility_5d` | Short-term realized volatility |
| `volatility_20d` | Medium-term realized volatility |
| `volume_ratio` | Today's volume vs 20-day average — detects unusual activity |

### Gap-specific features
These three were built specifically for this problem. The idea is that the way a stock closes can give indirect signals about what might happen overnight.

| Feature | What it captures |
|---|---|
| `prev_gap_pct` | Did it gap yesterday ? Gaps tend to cluster during earnings season |
| `close_to_high` | How far the close is from the daily high (closed strong = buyers in control until the end) |
| `close_to_low` | How far the close is from the daily low |

---

## Label

```
label = 1  if  |open(t+1) - close(t)| / close(t)  >  0.01
label = 0  otherwise
```

A **gap day** is when the next morning's open price is more than 1% away from yesterday's close. In our dataset, gap days represent approximately **37% of all observations** — though it varies a lot by stock (AAPL ~22%, TSLA ~52%).

---

## Sample rows

Here are a few rows from the dataset to give an idea of what it looks like :

| date | ticker | close | rsi_14 | volatility_20d | prev_gap_pct | close_to_high | label |
|---|---|---|---|---|---|---|---|
| 2021-02-02 | AAPL | 131.28 | 58.58 | 0.0182 | 0.0031 | 0.0095 | 0 |
| 2021-02-03 | AAPL | 130.26 | 54.35 | 0.0179 | -0.0021 | 0.0001 | 1 |
| 2021-02-04 | AAPL | 133.62 | 61.61 | 0.0177 | 0.0024 | 0.0001 | 0 |
| 2021-02-05 | NVDA | 534.89 | 71.23 | 0.0251 | 0.0187 | 0.0034 | 1 |

---

## Files in this repo

```
prices_with_indicators.csv   → the full dataset (7300+ rows, 27 columns)
feature_columns.txt          → list of the 16 features used by the models
GAP_PREDICTION_FINAL.ipynb   → the full notebook (data collection + 5 experiments)
README.md                    → this file
```

---

## How to reproduce

```python
# install dependencies
pip install yfinance pandas numpy scikit-learn imbalanced-learn xgboost matplotlib

# then just run the notebook from top to bottom
# it will re-download the data and regenerate all figures and results
```

---

## Why this project

This came from my preparation for finance interviews (structuring and trading desks). A lot of interview questions touch on gap risk and how to think about overnight moves. I wanted to see how far you can get with just technical indicators — no sentiment, no news, no options data. The answer is: you can get something (~0.647 AUROC) but there's a clear ceiling because the real driver of gaps is information that becomes available after market close.

The most intersting finding is that raising the confidence threshold to 70%+ pushes precision above 0.73, meaning the model is quite reliable when it's confident — it just can't predict on every single day.

---

## References

- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011
- Ran Aroussi, *yfinance*, https://github.com/ranaroussi/yfinance
- Caporale & Plastun, *Price Gaps: Another Market Anomaly?*, Investment Analysts Journal, 2019
