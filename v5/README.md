# Quantitative Forecasting System — V2 Production Guide

A multi-model ML system for next-day equity direction forecasting.
Models: BiLSTM, BiGRU, Causal Transformer, TCN, XGBoost, and a stacked Ensemble.

**V2 addresses all critical issues identified in the V1 audit.**

---

## What Changed in V2

### 🔴 1. Data Leakage — FIXED
V1 called `build_features(full_df)` before splitting. Rolling/EWM features
saw future bars during warm-up — pure look-ahead bias.

V2: Raw OHLCV split first, `build_features()` called independently per partition.

### 🔴 2. Class Imbalance — FIXED
V1 used default weights (1.0), causing models to predict one direction trivially.

V2: `get_pos_weight(y_train)` auto-computed and injected into XGBoost
`scale_pos_weight` and PyTorch `BCEWithLogitsLoss(pos_weight=...)`.

### 🔴 3. Per-Model HPO — FIXED
V1 ran HPO only on LSTM, used that lr for all models.

V2: `hpo_search()` called independently per architecture when `--hpo` is active.

### 🟠 4. Feature Selection — NEW
MI + XGBoost importance (rank-blended) prunes ~110 → top-60 features.

### 🟠 5. SHAP Explainability — NEW
TreeExplainer on XGBoost: `shap_summary.png`, `shap_importance.csv`, waterfall plots.

### 🟠 6. Multi-Ticker Support — NEW
`build_multi_ticker_dataset()` trains across multiple stocks for regime diversity.

### 🟡 7. Realistic Backtest — IMPROVED
Separate `slippage_bps=3.0` from `transaction_cost_bps=5.0` (was combined).

---

## Quick Start

```bash
pip install -r requirements.txt

# Basic training (single ticker)
python main.py

# With per-model HPO (recommended, +30 min)
python main.py --hpo --hpo-trials 25

# Multi-ticker (edit DATA.tickers in config.py first)
python main.py --multi-ticker

# Predict tomorrow
python predict.py --ticker AAPL --format json
```

---

## Project Structure

```
├── config.py          # V5: tickers, feature_selection_k, slippage_bps
├── main.py            # V8: 9-step pipeline — all fixes integrated
├── predict.py         # V2: loads features.pkl for exact feature alignment
├── features.py        # V7: leakage-free, class weights, feature selection
├── models.py          # V6: pos_weight in all models
├── trainer.py         # V2: per-model HPO, pos_weight in loss
├── walk_forward.py    # V2: class weights per fold
├── shap_analysis.py   # NEW: SHAP explainability
├── visualisation.py   # V2: 12 charts including SHAP
├── ensemble.py        # V6: unchanged
├── evaluation.py      # V8: slippage from config
├── data_loader.py     # V1: unchanged
├── logger.py          # unchanged
└── results/
    ├── checkpoints/   # + features.pkl (NEW — required by predict.py)
    ├── shap/          # NEW: shap_importance.csv, shap_summary.png
    ├── charts/        # 12 PNGs
    └── logs/
```

---

## Key Config Options (config.yaml)

```yaml
data:
  ticker: AAPL
  period: 7y                  # longer period recommended with leakage-free split
  tickers: [AAPL, MSFT, GOOGL]
  feature_selection_k: 60

backtest:
  transaction_cost_bps: 5.0
  slippage_bps: 3.0
  confidence_threshold: 0.55
  stop_loss_pct: 0.05

experiment:
  run_hpo: false
  hpo_all_models: true
  run_feature_selection: true
  run_shap: true
  use_multi_ticker: false
```

---

## Troubleshooting

**`Empty partition after leakage-free split`** → Use `period: 7y` or `10y`.

**SHAP skipped** → `pip install shap>=0.44.0`

**Models still predict one direction** → Check `pos_weight` in `summary.json`.
Should be > 1.0 if UP is minority class.
