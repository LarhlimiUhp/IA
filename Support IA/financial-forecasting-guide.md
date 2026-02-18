# Field Guide: Forecasting Multivariate Financial Time-Series

A compact yet comprehensive reference covering conceptual explanations, practical workflow advice, and pointers to concrete models and code — from raw data to a deployable forecaster or trading signal.

---

## 1. Define the Problem Clearly

| Dimension | Examples |
|-----------|----------|
| **Target(s)** | prices, log-returns, volatility, drawdown risk, spreads, factors, PnL |
| **Forecast horizon** | intraday minutes, EOD, weekly, t+20 |
| **Purpose** | trading signal, VaR, asset–liability management, hedging, scenario generation |
| **Evaluation metric** | RMSE/MAE, direction-of-change, Sharpe, hit-ratio, CVaR |
| **Exogenous inputs** | macro releases, rates, sentiment, order-book features, ESG scores |

---

## 2. Data Pipeline

### 2.1 Collection

- Prices/quotes: Bloomberg, Refinitiv, crypto APIs
- Fundamentals: Compustat
- Macro: FRED
- News feeds and alternative data

### 2.2 Cleaning

- De-duplication
- Corporate-action adjustments
- Timezone alignment
- Forward-fill or interpolation of small gaps
- Remove overnight or auction prints if intraday

### 2.3 Feature Engineering

- **Price transforms**: returns, log-returns, √vol, momentum, slopes
- **Technical indicators**: RSI, ATR, Bollinger width, OBV
- **Statistical**: rolling mean/σ, z-scores, PCA factors, clustering labels
- **Cross-asset lags**: e.g. USDJPY returns lagged 1 h → Nikkei futures
- **Calendar**: day-of-week, quarter-end, FOMC days, option expiry
- **Text / sentiment**: news polarity scores, social-media buzz

### 2.4 Normalisation

- **Stationarity**: difference, percentage change, Box-Cox, log
- **Scaling**: z-score, min–max, robust scaler (**fit only on train!**)

### 2.5 Train / Validation / Test Split

- Non-shuffled, walk-forward or expanding-window CV
- Example: 80% train → walk-forward 10% val → final 10% out-of-sample

---

## 3. Model Families

### A. Classical Econometrics

| # | Model | Use Case |
|---|-------|----------|
| 1 | **VAR / VARX / Bayesian VAR** | Good baseline, interpretable impulse responses |
| 2 | **VECM (Johansen)** | Cointegrated price series / spreads |
| 3 | **State-space & Kalman** | Time-varying loadings, factor models (e.g. Dynamic Nelson-Siegel) |
| 4 | **(M)GARCH** (CCC, DCC, BEKK, GO) | Volatility & correlation forecasting |
| 5 | **Markov-Switching / Regime VAR** | Bull/bear or low/high-vol regime capture |

### B. Machine Learning, Shallow

| # | Model | Notes |
|---|-------|-------|
| 1 | Penalised regression (Lasso-/Ridge-VAR) | Many series |
| 2 | Tree models (RF, XGBoost, LightGBM) | Lagged features |
| 3 | SVM with RBF/poly kernels | Mid-frequency classification |
| 4 | k-NN / LOF | Anomaly detection (not forecasting) |

### C. Deep Learning

| # | Model | Notes |
|---|-------|-------|
| 1 | **RNN family** (LSTM, GRU, BiLSTM, Stacked) | Long-memory sequences |
| 2 | **Seq2seq + attention; TCN** | Temporal Convolutional Networks |
| 3 | **Transformers for TS** | TFT, Informer, Autoformer, FEDformer |
| 4 | **Hybrid CNN-LSTM** | CNN extracts local patterns, LSTM captures long memory |
| 5 | **Probabilistic DL** | DeepAR, DeepVAR (GluonTS), N-BEATS, N-HiTS |

### D. Ensemble / Hybrid

- Combine statistical VAR with LSTM residuals
- Stack XGBoost on top of deep net features
- Average forecasts via simple mean, Bayesian model averaging, or meta-learner

---

## 4. How to Choose a Model

1. **Start simple** — baseline: naïve random-walk, last value, or VAR(1)
2. **Check data volume & dimensionality**:
   - Daily returns of 5 indices (~5×10³ obs) → VAR or Bayesian VAR
   - 1-second BTC/ETH prices for 2 years (~6×10⁷ obs) → scalable DL (TCN or Transformer)
3. **Need prediction intervals?** → probabilistic models (DeepAR, Bayesian VAR, GARCH)
4. **Latency constraints**: real-time may preclude heavy transformers
5. **Regulatory / interpretability**: use linear factor models or SHAP on tree ensembles

---

## 5. Workflow / Example (Python)

### A. Data & Features

```python
import yfinance as yf
import pandas as pd
import numpy as np

tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
prices = yf.download(tickers, start='2015-01-01')['Adj Close']
rets   = np.log(prices).diff().dropna()

# Lagged features up to 5 days
X = pd.concat(
    [rets.shift(i).add_suffix(f'_lag{i}') for i in range(1, 6)],
    axis=1
).dropna()
y = rets.loc[X.index]
```

### B. Split (expanding window)

```python
split    = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
```

### C. Fit a VAR

```python
from statsmodels.tsa.api import VAR

model   = VAR(endog=y_train)
var_res = model.fit(maxlags=5, ic='aic')
forecast = var_res.forecast(y_train.values[-5:], steps=len(y_test))
pred_df  = pd.DataFrame(forecast, index=y_test.index, columns=y_test.columns)
```

### D. Evaluate

```python
from sklearn.metrics import mean_squared_error

rmse             = np.sqrt(mean_squared_error(y_test.values, pred_df.values))
directional_acc  = ((np.sign(pred_df) == np.sign(y_test)).mean()).mean()

print(f'RMSE={rmse:.5f}', f'DirAcc={directional_acc:.3%}')
```

### E. Swap in an LSTM (Keras)

```python
import tensorflow as tf

timesteps, features = 20, len(tickers)

def create_xy(df, steps):
    X, y = [], []
    for i in range(len(df) - steps):
        X.append(df.iloc[i:i+steps].values)
        y.append(df.iloc[i+steps].values)
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_xy(rets, timesteps)
split = int(0.8 * len(X_lstm))

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(timesteps, features), return_sequences=False),
    tf.keras.layers.Dense(features)
])
model.compile(optimizer='adam', loss='mse')
model.fit(
    X_lstm[:split], y_lstm[:split],
    epochs=20,
    validation_data=(X_lstm[split:], y_lstm[split:])
)
```

### F. Probabilistic with GluonTS DeepAR

```python
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

train_ds = ListDataset(
    [{"target": rets[col].values, "start": str(rets.index[0])} for col in rets.columns],
    freq="1D"
)

est = DeepAREstimator(freq="1D", prediction_length=10, trainer=Trainer(epochs=50))
predictor = est.train(train_ds)
```

---

## 6. Special Topics

| Topic | Approach |
|-------|----------|
| **Cointegration / Pairs** | Johansen test → VECM → hedge-ratio trading |
| **Cross-section + time** | Panel LASSO, Deep Cross-Sectional LSTM |
| **Volatility focus** | Multivariate DCC-GARCH or log-variance LSTM |
| **Regime switching** | Hidden Markov (`hmmlearn`) or MS-VAR |
| **Copulas** | Fit marginal GARCH, model tail dependence for stress scenarios |
| **Reinforcement learning** | Treat allocation as action; environment driven by forecast model |
| **Transfer learning** | Pre-train on long history / similar asset class, then fine-tune |
| **Online learning** | Update weights incrementally (River, skmultiflow) |

---

## 7. Backtesting & Risk

- Run signal generation fully **out-of-sample**; no peeking into future volumes/spreads
- Include realistic costs: slippage, commission, borrow fees, funding
- Measure turnover, drawdowns, max adverse excursion
- Apply **White's reality check** or **Deflated Sharpe** to avoid data-snooping bias

---

## 8. Where to Go Next

### Papers & Books

- *Forecasting Volatility and Correlation* — Engle (multivariate GARCH)
- *Time Series and Panel Data Econometrics* — Vogelsang & Lütkepohl
- *Advances in Financial Machine Learning* — Marcos López de Prado
- *Temporal Fusion Transformers* — Lim et al. (2021)

### Key Libraries

| Category | Libraries |
|----------|-----------|
| Econometrics | `statsmodels`, `arch` |
| Time-series ML | `sktime`, `darts`, `kats`, `tsfresh` |
| Probabilistic DL | `gluonts`, `prophet`, `tensorflow-probability`, `pyflux` |

### Competitions & Datasets

- Two Sigma Financial Modeling Challenge (Kaggle)
- FLA-FX, CryptoForecasting (ICML workshops)
- Numerai

---

## Key Takeaways

1. Start with a **crystal-clear objective** and evaluation metric
2. Build a **disciplined pipeline**: data quality > model sophistication
3. Use **simple statistical baselines** first; escalate to deep nets only if data size, horizon, or nonlinearities truly demand it
4. **Backtest rigorously** with walk-forward CV and cost-aware metrics
5. Keep models **updateable** — markets change faster than our code
