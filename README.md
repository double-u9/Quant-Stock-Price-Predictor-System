# Quantitative Forecasting System

A multi-model machine learning pipeline for equity direction forecasting. Given a ticker and a historical window, it trains five model architectures, combines them into a stacked ensemble, backtests the strategy with realistic costs, and produces a next-period probability forecast.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Models](#models)
- [Feature Engineering](#feature-engineering)
- [Ensemble & Calibration](#ensemble--calibration)
- [Backtesting](#backtesting)
- [Statistical Validity](#statistical-validity)
- [Walk-Forward Validation](#walk-forward-validation)
- [Configuration Reference](#configuration-reference)
- [Outputs](#outputs)
- [Prediction (Inference Only)](#prediction-inference-only)
- [SHAP Explainability](#shap-explainability)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)

---

## Overview

The system predicts whether a stock's price will be higher or lower over a configurable horizon (default: next day). It is built around three core principles:

**Leakage-free by construction.** All feature engineering, scaling, and feature selection runs independently on each split. No future data touches the training window at any stage.

**Statistically honest.** Every backtest is accompanied by a bootstrap Sharpe confidence interval and the Deflated Sharpe Ratio (DSR), which accounts for multiple testing across model versions and hyperparameter trials.

**Production-ready checkpoints.** Models are saved in format-safe serialisation (safetensors for PyTorch, native JSON for XGBoost, joblib for sklearn) with SHA-256 integrity sidecar files. No pickle anywhere in the inference path.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train on AAPL with defaults (~15–25 min on CPU, ~5 min on GPU)
python main.py

# Train with per-model hyperparameter optimisation (+30–60 min)
python main.py --hpo --hpo-trials 25

# Train across multiple tickers
python main.py --multi-ticker

# Run a deeper walk-forward (trains deep models in each fold)
python main.py --deep-wf

# Predict next day without retraining (requires trained checkpoints)
python predict.py --ticker AAPL
python predict.py --ticker AAPL --format json
```

All results, charts, checkpoints, and logs are written to `results/`.

---

## Project Structure

```
├── main.py               # 10-step training pipeline — entry point
├── predict.py            # Standalone inference from saved checkpoints
├── config.py             # All hyperparameters and paths (frozen dataclasses)
├── config.yaml           # Human-editable overrides (loaded at startup)
│
├── data_loader.py        # OHLCV download, validation, and cleaning
├── features.py           # ~110-feature engineering engine + target builder
├── models.py             # LSTM, GRU, Transformer, TCN, XGBoost architectures
├── trainer.py            # Training loop, HPO, AMP, gradient monitoring
├── ensemble.py           # Equal-weight, AUC-weighted, stacking, calibrated
├── evaluation.py         # Classification + trading metrics, DSR, cost tests
├── walk_forward.py       # Purged expanding-window cross-validation
├── visualisation.py      # 11 diagnostic charts
├── logger.py             # Structured logging, step timing, JSON events
├── shap_analysis.py      # SHAP explainability for XGBoost
│
├── requirements.txt
├── config.yaml
└── results/
    ├── checkpoints/      # Saved model weights + integrity hashes
    ├── charts/           # 11 PNG diagnostic charts
    ├── logs/             # Rotating log files (full + errors-only)
    └── shap/             # SHAP importance CSV + summary plot
```

---

## Pipeline Walkthrough

`main.py` runs ten sequential steps. Each is logged with elapsed time.

| Step | Name | What happens |
|------|------|--------------|
| 1 | Data & Feature Engineering | Downloads OHLCV from Yahoo Finance, validates and cleans it, then builds features independently on each split to prevent leakage |
| 2 | Class Imbalance & Weights | Computes `pos_weight` from training labels; injected into XGBoost and PyTorch loss |
| 3 | Walk-Forward Validation | Purged expanding-window CV across 8 folds; establishes out-of-sample baseline before final training |
| 4 | Feature Selection | Mutual information + XGBoost importance scored in inner chronological folds; top-60 features selected; single final scaler fitted |
| 5 | Scaling & Sequences | RobustScaler applied to all splits; 50-bar sliding windows created for deep models |
| 6 | Train Models | XGBoost + LSTM + GRU + Transformer + TCN, each with optional per-model HPO |
| 7 | Ensemble | Four combination strategies fitted on held-out val slices; all checkpoints saved |
| 8 | SHAP + Evaluation + Charts | Feature attribution, full classification and trading metrics, 11 charts, regime-conditional breakdown, diagnostic health checks |
| 9 | Tomorrow's Prediction | Final probability forecast from each model and the ensemble |
| 10 | Save Results | `prediction_log.csv` and `summary.json` written to `results/` |

---

## Models

All deep models take sequences of shape `(batch, 50, n_features)` and output a single logit. All are strictly causal — no future bars within the input window are visible at any time step.

### LSTM (`LSTMPredictor`)
Unidirectional two-layer LSTM with LayerNorm input projection and causal temporal attention. Attention re-weights earlier time steps by learned relevance without attending to future positions.

- Hidden size: 64
- Layers: 2
- Dropout: 0.30
- Attention head: 64 → 32 → 16 → 1

### GRU (`GRUPredictor`)
Same architecture as LSTM with GRU cells. Slightly lower dropout (0.25). Typically trains faster and generalises comparably.

### Causal Transformer (`TransformerPredictor`)
Patch-embedding Transformer. The 50-bar sequence is divided into non-overlapping patches of size 5 (10 patches total), embedded linearly, then passed through a causal Transformer encoder with sinusoidal positional encoding.

- Model dimension: 64
- Attention heads: 4
- Encoder layers: 2
- Feed-forward dimension: 128

### TCN (`TCNPredictor`)
Temporal Convolutional Network with exponentially increasing dilation (1, 2, 4, 8 across 4 levels). Each block uses causal convolutions with padding trimmed to prevent future leakage, InstanceNorm, GELU activation, and residual connections.

- Channels: 32
- Levels: 4
- Kernel size: 3

### XGBoost
Gradient boosted trees on the flat (non-sequential) feature vector. Regularised for short financial datasets.

- Trees: 600
- Max depth: 3
- Learning rate: 0.02
- `scale_pos_weight`: computed from training labels
- Early stopping on last 10% of training data

---

## Feature Engineering

`build_features()` in `features.py` engineers approximately 110 features from OHLCV data. All rolling calculations use only past bars — no forward-looking windows.

**Trend & momentum**
- Log returns at lags 1, 2, 3, 5, 10, 20 days
- SMA at windows 5, 10, 20, 50, 100, 200; price-vs-SMA ratios; SMA crossovers
- EMA at spans 9, 12, 21, 26; MACD, signal, histogram, normalised MACD
- RSI at periods 7, 14, 21 with distance-from-50 and momentum features
- Rate of change at periods 5, 10, 20

**Volatility**
- Garman-Klass volatility at 5 and 20 days
- Historical volatility (annualised) at 5, 10, 20, 60 days with percentile rank
- High/low volatility ratio (5/20 and 10/60 day pairs)
- Bollinger Bands width, position, and squeeze at windows 10 and 20
- ATR (7, 14, 21 day) normalised by price; Keltner channel position

**Volume & microstructure**
- Volume ratio vs rolling average at 5, 10, 20 days; volume trend
- On-balance volume normalised by EMA; Amihud illiquidity ratio
- Force index at 2 and 13 days; VWAP deviation

**Candle structure**
- Body size, body percentage, upper and lower shadow fractions, high-low range

**Statistical**
- Rolling autocorrelation at lags 1 and 5 (20-bar and 30-bar windows)
- Return sign entropy (20-bar window)
- Z-score vs 20, 60, 120-day mean; price acceleration; volatility acceleration

**Regime & range**
- Rolling Sharpe ratio and mean-reversion score (60-day window)
- 52-week high/low distance and position within range
- Donchian channel position and width (10 and 20 days)

**Oscillators**
- Stochastic %K and %D, Williams %R, CCI (20-day)
- Elder Ray Bull/Bear Power

### Target modes

The prediction target is controlled by `target_mode` in config:

| Mode | Column | Description |
|------|--------|-------------|
| `binary` | `target_{h}d` | 1 if Close rises over horizon h, else 0 |
| `return` | `target_ret_{h}d` | Raw log return (regression) |
| `excess_return` | `target_exc_bin_{h}d` | 1 if return exceeds rolling median |
| `rank` | `target_rank_bin_{h}d` | 1 if percentile rank > 0.5 in rolling window |

Default is `binary`. For single-ticker use, `excess_return` is recommended. For multi-ticker use, `rank` is recommended.

---

## Ensemble & Calibration

The ensemble (`ensemble.py`) combines base model probabilities using four strategies, all fitted on held-out data to avoid contamination:

**Training data split** (strict chronological order):
- Full training set → base model weights
- Validation first half → stacking meta-learner
- Validation second half → AUC weights + Platt calibration
- Test set → final reporting only

**Strategies:**

`predict_equal` — Simple NaN-aware mean across all models. Always available; baseline for the other strategies.

`predict_auc_weighted` — Softmax weighting over diversity-penalised validation AUCs. Models that are highly correlated with a better model receive a penalty, encouraging the weighting to favour diverse signal sources.

`predict_stacking` — Logistic regression meta-learner trained on the stack slice. Can assign negative weights, allowing it to down-weight models that are systematically contrarian to the correct signal.

`predict_calibrated_auc` — Platt scaling is applied per model before AUC-weighted combination, correcting systematic over- or under-confidence.

---

## Backtesting

`trading_metrics()` in `evaluation.py` implements a time-correct backtest with realistic costs.

**Execution model:** Signal generated from Close[t] is executed at Open[t+1]. No same-bar execution — there is no look-ahead in the P&L calculation.

**Position sizing:** Fractional Kelly with a conservative fraction of 0.20. Kelly fraction is computed from the rolling win/loss ratio of actual strategy P&L (not raw price returns), so it accounts for direction correctly.

**Confidence filter:** Only trade when `|p − 0.5| ≥ confidence_threshold`. Default is 0.0 (trade every signal). Raise this to filter for high-conviction signals once positive alpha is confirmed.

**Cost model:** Three-component one-way cost applied on every position change:
- Commission: 1.0 bps
- Slippage (half-spread + timing): 2.0 bps
- Market impact (square-root law): `λ × √(trade_size / ADV)`, default ~0.3 bps for a 1000-share trade against 10M ADV

Reversals (long→short) pay the full cost twice: once to close, once to open.

**Stop-loss:** Per-trade entry price tracked bar-by-bar. If unrealised loss exceeds `stop_loss_pct` (default 5%), the position closes at the next open. The number of stops fired is reported as `n_stopped`.

**Reported metrics:**

| Category | Metrics |
|----------|---------|
| Risk-adjusted | Sharpe, Sortino, Calmar, Omega ratio |
| Drawdown | Max DD, max DD duration, max DD recovery, average DD |
| Returns | Total return, annualised return |
| Tail risk | VaR 95%, CVaR 95% |
| Trades | Win rate, profit factor, trade count, stops fired, avg holding period, turnover |
| Signal quality | % confident bars, Information Coefficient (rank IC) |
| Benchmark | Alpha, Beta, Information Ratio, excess return vs buy-and-hold |

---

## Statistical Validity

Two statistical tests accompany every backtest to guard against overfitting:

**Bootstrap Sharpe CI** (`sharpe_bootstrap_ci`): Block bootstrap with block size estimated from lag-1 autocorrelation. Produces a 95% confidence interval for the annualised Sharpe. A Sharpe of 1.0 on 252 observations has a CI of approximately [−0.3, +2.3] — statistically indistinguishable from zero.

**Deflated Sharpe Ratio** (`deflated_sharpe_ratio`): Accounts for the number of model versions, HPO trials, and walk-forward folds evaluated against the same test set. A DSR > 0.95 is required before treating performance as evidence of genuine edge. The verdict is printed in the run summary.

---

## Walk-Forward Validation

`walk_forward.py` runs an expanding-window cross-validation before the final model is trained, giving an unbiased estimate of out-of-sample performance.

**Expanding train window:** Each fold trains on all data from bar 0 to fold boundary, then tests on the next window. The train set grows each fold — no data is discarded.

**Purge gap:** A purge of `max(purge_days, seq_len + horizon)` bars separates training from test in each fold. This ensures no training sequence overlaps with any test label, preventing autocorrelation leakage at fold boundaries.

**Per-fold output:** Each fold reports accuracy, AUC, Information Coefficient, Sharpe, Sortino, win rate, max drawdown, the gap actually used, and whether the gap satisfies the sequence-aware minimum.

**Baseline comparison:** The simple 5/20-day MA crossover is evaluated in each fold. Walk-forward XGBoost accuracy is reported as both an absolute and as a delta versus this baseline.

---

## Configuration Reference

Edit `config.yaml` to override defaults without touching Python code. All sections are optional — omitted keys use the values defined in `config.py`.

```yaml
data:
  ticker: "AAPL"              # Primary ticker for single-asset mode
  period: "10y"               # yfinance period string (e.g. 5y, 7y, 10y)
  train_ratio: 0.70           # Fraction of dates for training
  val_ratio: 0.15             # Fraction of dates for validation
  seq_len: 50                 # Bars per input sequence for deep models
  primary_horizon_idx: 0      # 0=next day, 1=next week (5d), 2=next month (21d)
  target_mode: "binary"       # binary | return | excess_return | rank
  feature_selection_k: 60     # Top-K features to keep (null = keep all)
  purge_days: 5               # Embargo days between folds in walk-forward
  tickers: [AAPL, MSFT, GOOGL, AMZN, NVDA]  # Universe for multi-ticker mode
  cs_norm_method: "zscore"    # Cross-sectional norm: zscore | rank | none
  align_method: "inner"       # inner=common dates only, outer=all dates

model:
  epochs: 80
  batch_size: 64
  lr: 0.0003
  patience: 15                # Early stopping patience (epochs)
  threshold: 0.50             # Probability threshold for binary prediction
  lstm_hidden: 64
  lstm_layers: 2
  tf_d_model: 64
  tf_nhead: 4
  tf_layers: 2
  tcn_channels: 32
  tcn_levels: 4
  n_walks: 8                  # Walk-forward folds

backtest:
  commission_bps: 1.0         # One-way broker commission
  slippage_bps: 2.0           # One-way half-spread + timing slippage
  impact_coeff: 0.1           # Market impact coefficient λ
  adv_shares: 10000000        # Average daily volume for impact calculation
  trade_size_shares: 1000     # Notional trade size for impact calculation
  confidence_threshold: 0.0   # Min |p - 0.5| to execute a signal
  use_kelly: true             # Use fractional Kelly position sizing
  kelly_fraction: 0.20        # Conservative Kelly multiplier
  stop_loss_pct: 0.05         # Max unrealised loss before forced exit

experiment:
  run_hpo: false              # Per-model Optuna hyperparameter search
  hpo_trials: 20              # Optuna trials per model
  deep_wf: false              # Train deep models in each walk-forward fold
  run_feature_selection: true
  run_shap: true
  run_baseline: true          # Always compare vs MA-crossover baseline
  use_multi_ticker: false

paths:
  results_dir: "results"
```

**CLI flags** (override config.yaml):

```bash
--hpo / --no-hpo          Enable or disable hyperparameter optimisation
--hpo-trials N            Number of Optuna trials per model
--deep-wf / --no-deep-wf  Enable or disable deep models in walk-forward
--multi-ticker            Train across all tickers in DATA.tickers
```

---

## Outputs

After a completed run, `results/` contains:

```
results/
├── checkpoints/
│   ├── lstm.pt                # LSTM weights (safetensors format)
│   ├── lstm.pt.sha256         # SHA-256 integrity hash
│   ├── gru.pt
│   ├── gru.pt.sha256
│   ├── transformer.pt
│   ├── transformer.pt.sha256
│   ├── tcn.pt
│   ├── tcn.pt.sha256
│   ├── xgboost.json           # XGBoost native JSON (no pickle)
│   ├── scaler.joblib          # RobustScaler (joblib)
│   ├── scaler.joblib.sha256
│   ├── ensemble.joblib        # Fitted Ensemble object (joblib)
│   ├── ensemble.joblib.sha256
│   └── features.json          # Selected feature names (plain JSON)
│
├── charts/
│   ├── 01_price_dashboard.png     # OHLC, volume, SMA overlays
│   ├── 02_training_curves.png     # Train/val loss per model
│   ├── 03_confusion_matrices.png  # Per-model confusion matrices
│   ├── 04_roc_curves.png          # ROC curves with AUC annotations
│   ├── 05_feature_importance.png  # XGBoost top-25 feature importances
│   ├── 06_walk_forward.png        # Per-fold accuracy + Sharpe
│   ├── 07_strategy_returns.png    # Cumulative equity curves, all models
│   ├── 08_dashboard.png           # Summary metrics grid
│   ├── 09_regime.png              # Market regime timeline
│   ├── 10_calibration.png         # Reliability diagrams per model
│   └── 11_confidence_deciles.png  # Hit rate by prediction confidence
│
├── shap/
│   ├── shap_importance.csv        # Mean |SHAP| per feature
│   ├── shap_summary.png           # Beeswarm plot, top-20 features
│   └── shap_waterfall_0.png       # Single-sample waterfall
│
├── logs/
│   ├── run_YYYYMMDD_HHMMSS.log         # Full DEBUG log with rotation
│   └── run_YYYYMMDD_HHMMSS_errors.log  # WARNING+ only
│
├── prediction_log.csv    # Appended row per run: directions + metrics
└── summary.json          # Full structured run summary
```

---

## Prediction (Inference Only)

To generate a forecast without retraining, run `predict.py` against saved checkpoints:

```bash
# Default ticker and checkpoint directory from config
python predict.py

# Specific ticker
python predict.py --ticker MSFT

# Custom checkpoint directory
python predict.py --ticker AAPL --checkpt-dir /path/to/checkpoints

# JSON output (for programmatic use)
python predict.py --ticker AAPL --format json
```

`predict.py` loads checkpoints using safe, format-specific loaders:
- Deep models: `load_checkpoint()` with safetensors + SHA-256 verification
- XGBoost: `model.load_model()` from native JSON
- Scaler: `joblib.load()` with SHA-256 verification
- Features: `json.load()` (plain text, zero deserialisation risk)
- Ensemble: `joblib.load()` with SHA-256 verification

If any checkpoint file has been modified since training, the SHA-256 check raises a `RuntimeError` before loading.

---

## SHAP Explainability

When `experiment.run_shap: true` (default), the pipeline runs SHAP TreeExplainer on the XGBoost model after evaluation:

- `shap_importance.csv` — mean absolute SHAP value per feature, sorted descending
- `shap_summary.png` — beeswarm plot showing feature direction and magnitude for the top 20 features
- `shap_waterfall_0.png` — waterfall decomposition of the model's prediction for the first test sample

To run SHAP analysis separately:

```python
from shap_analysis import shap_xgb, save_shap_importance, shap_summary_plot

shap_vals = shap_xgb(xgb_model, X_test, feature_names, max_samples=500)
shap_df   = save_shap_importance(shap_vals, feature_names, output_dir='results/shap')
shap_summary_plot(shap_vals, feature_names, output_dir='results/shap')
```

---

## Troubleshooting

**`Empty partition after leakage-free split`**  
The dataset is too short for the configured split ratios. Use `period: 10y` or increase the period. The minimum is approximately 600 clean bars after filtering.

**`0 trades executed` / `Sharpe = -1,000,000`**  
`confidence_threshold` is too high. Set it to `0.0` in `config.yaml` to trade every signal. Once you confirm the strategy has positive alpha, raise it incrementally (0.02 → 0.05 → 0.10).

**SHAP step skipped**  
Install the SHAP library: `pip install shap>=0.44.0`.

**Models predict only one direction**  
Check `pos_weight` in `summary.json`. If it is very high (e.g. > 5), the dataset is heavily imbalanced. Try a longer period to get more balanced class counts.

**`INTEGRITY CHECK FAILED` on checkpoint load**  
The `.pt` or `.joblib` file has been modified since training. Delete the `results/checkpoints/` directory and retrain.

**Out of memory on GPU**  
Reduce `batch_size` in config (try 32). Alternatively, reduce `lstm_hidden` and `tf_d_model` to 32.

**Walk-forward is very slow**  
Set `deep_wf: false` in config (default). This runs only XGBoost in each fold, which completes in seconds. Deep-model walk-forward is informative but trains 5 × n_walks full models.

---

## Requirements

```
torch>=2.0.0
xgboost>=1.7.0
scikit-learn>=1.3.0
scipy>=1.9.0
numpy>=1.24.0
pandas>=2.0.0
yfinance>=0.2.28
matplotlib>=3.7.0
optuna>=3.0.0
shap>=0.44.0
PyYAML>=6.0
psutil>=5.9.0
safetensors>=0.4.0     # recommended for secure checkpoint saving
joblib>=1.3.0
```

Python 3.9 or later is required. CUDA is optional — the pipeline runs on CPU with all features available, though deep model training is significantly slower.
