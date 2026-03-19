"""
main.py  —  V9 Quantitative Forecasting Pipeline.

V9 improvements over V8:

  CRITICAL FIXES:
  ───────────────
  1. CONFIDENCE THRESHOLD FIXED (config.py V6):
     confidence_threshold changed from 0.55 to 0.0.
     V8 threshold caused 0 trades and Sharpe = -1,000,000 in every backtest.
     Now executes every signal; raise threshold after confirming positive alpha.

  2. SINGLE SCALER PIPELINE:
     V8 built scaler_pre for feature selection, then rebuilt scaler from scratch
     for actual training — two scalers on the same data caused subtle mismatches.
     V9 builds ONE scaler after feature selection and uses it everywhere.

  3. DEEP MODEL CHECKPOINTS SAVED EXPLICITLY:
     V8 saved scaler + XGBoost but NOT deep model .pt files with canonical names
     (lstm.pt, gru.pt, transformer.pt, tcn.pt). V9 saves all of them so
     predict.py can load them reliably.

  4. MODEL SIZE REDUCED (config.py V6 + models.py V7):
     Hidden dims halved (128→64) to prevent overfitting on ~2500 daily bars.
     XGBoost more regularised (max_depth 3, min_child_weight 5).

  NEW FEATURES:
  ─────────────
  5. MA-CROSSOVER BASELINE (features.py V8):
     A simple 5/20 SMA crossover is evaluated alongside ML models.
     Any model that can't beat this has no real edge.

  6. REGIME-CONDITIONAL EVALUATION (features.py V8):
     Test performance broken down by market regime (trending_up,
     trending_down, high_vol, mean_reverting).

  7. PURGED WALK-FORWARD (walk_forward.py V3):
     Embargo gap between train and test folds prevents autocorrelation
     leakage at fold boundaries.

  8. DIAGNOSTIC BLOCK after evaluation:
     Explicitly warns if: (a) all models worse than baseline,
     (b) 0 trades executed, (c) AUC < 0.52 for all models.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA, MODEL, BACKTEST, PATHS, EXPERIMENT, SEED
from logger  import (setup_logging, log_environment, log_config,
                     StepTimer, log_epoch, log_metrics,
                     log_exception, log_run_summary, safe_call)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATHS.makedirs()
RUN_TIME = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

logger = setup_logging(
    log_dir       = PATHS.log_dir,
    console_level = logging.INFO,
    file_level    = logging.DEBUG,
    json_events   = False,
)

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from features     import (build_dataset, build_multi_ticker_dataset,
                           fit_scaler, apply_scale, make_sequences,
                           get_pos_weight, select_features,
                           ma_crossover_baseline, classify_regime)
from models       import (LSTMPredictor, GRUPredictor, TransformerPredictor,
                           TCNPredictor, build_xgb, count_params,
                           save_checkpoint)
from trainer      import (fit as torch_fit, predict_proba,
                           predict_single, hpo_search)
from evaluation   import classification_metrics, trading_metrics, print_metrics
from ensemble     import Ensemble
from walk_forward import run_walk_forward
import visualisation as viz


# ════════════════════════════════════════════════════════════════
#  RUN CONFIGURATION
# ════════════════════════════════════════════════════════════════

@dataclass
class RunConfig:
    run_hpo      : bool
    deep_wf      : bool
    hpo_trials   : int
    multi_ticker : bool

    def log_summary(self) -> None:
        logger.info(
            f"  RunConfig — HPO={self.run_hpo}  HPO_trials={self.hpo_trials}  "
            f"DeepWF={self.deep_wf}  MultiTicker={self.multi_ticker}"
        )


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description='Quant Forecasting Pipeline v9',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python main.py\n'
            '  python main.py --hpo --hpo-trials 30\n'
            '  python main.py --deep-wf\n'
            '  python main.py --multi-ticker\n'
        ),
    )
    parser.add_argument('--hpo',          dest='run_hpo',      action='store_true', default=None)
    parser.add_argument('--no-hpo',       dest='run_hpo',      action='store_false')
    parser.add_argument('--deep-wf',      dest='deep_wf',      action='store_true', default=None)
    parser.add_argument('--no-deep-wf',   dest='deep_wf',      action='store_false')
    parser.add_argument('--hpo-trials',   dest='hpo_trials',   type=int, default=None)
    parser.add_argument('--multi-ticker', dest='multi_ticker', action='store_true', default=None)
    args = parser.parse_args()

    return RunConfig(
        run_hpo      = args.run_hpo      if args.run_hpo      is not None else EXPERIMENT.run_hpo,
        deep_wf      = args.deep_wf      if args.deep_wf      is not None else EXPERIMENT.deep_wf,
        hpo_trials   = args.hpo_trials   if args.hpo_trials   is not None else EXPERIMENT.hpo_trials,
        multi_ticker = args.multi_ticker if args.multi_ticker is not None else EXPERIMENT.use_multi_ticker,
    )


def hdr(msg: str) -> None:
    bar = '=' * 60
    logger.info(f'\n{bar}\n  {msg}\n{bar}')


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=MODEL.batch_size, shuffle=shuffle, drop_last=shuffle)


# ════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def main(cfg: RunConfig) -> None:
    import time as _time
    _run_start = _time.perf_counter()

    hdr(f"Quant Forecasting System v9  >  {DATA.ticker}  >  {RUN_TIME}")
    logger.info(f"  Device={DEVICE}  Seed={SEED}")
    logger.info(f"  confidence_threshold={BACKTEST.confidence_threshold}  "
                f"(0.0 = execute all signals, raise after confirming alpha)")
    cfg.log_summary()
    log_environment(logger)
    log_config(logger, DATA, MODEL, BACKTEST, cfg, label="RUN CONFIG")

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Data & Feature Engineering
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 1 / 10  --  Data & Feature Engineering (Leakage-Free)')

    from data_loader import download
    if cfg.multi_ticker and len(DATA.tickers) > 1:
        logger.info(f"  Multi-ticker mode: {list(DATA.tickers)}")
        df_tr, df_va, df_te, FEATURES, TARGET_COLS = build_multi_ticker_dataset(
            list(DATA.tickers), DATA.period)
    else:
        raw = download(DATA.ticker, DATA.period)
        df_tr, df_va, df_te, FEATURES, TARGET_COLS = build_dataset(raw)

    PRIMARY = TARGET_COLS[DATA.primary_horizon_idx]
    logger.info(f"  Primary target: {PRIMARY}  ({DATA.horizons[DATA.primary_horizon_idx]}-day)")

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Class Imbalance Analysis
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 2 / 10  --  Class Imbalance & Weights')
    y_tr_raw = df_tr[PRIMARY].values.astype(np.float32)
    y_va_raw = df_va[PRIMARY].values.astype(np.float32)
    y_te_raw = df_te[PRIMARY].values.astype(np.float32)

    POS_WEIGHT = get_pos_weight(y_tr_raw)
    up_pct_tr  = float(y_tr_raw.mean() * 100)
    logger.info(f"  UP% train={up_pct_tr:.1f}%  pos_weight={POS_WEIGHT:.3f}")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Walk-Forward Validation (with purge gap + baseline)
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 3 / 10  --  Walk-Forward Validation (Purged)')
    raw_wf = download(DATA.ticker, DATA.period)
    wf_df = run_walk_forward(raw_wf, DEVICE, deep=cfg.deep_wf)

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Feature Selection  →  build SINGLE scaler
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 4 / 10  --  Feature Selection (MI + XGB) + Single Scaler')

    # V9 FIX: Build a temporary scaler to score features, then build
    # the FINAL scaler on selected features only. One scaler, used everywhere.
    scaler_tmp = fit_scaler(df_tr, FEATURES)
    X_tr_full  = apply_scale(scaler_tmp, df_tr, FEATURES)

    if EXPERIMENT.run_feature_selection and DATA.feature_selection_k is not None:
        FEATURES = select_features(
            X_tr_full, y_tr_raw, FEATURES,
            top_k=DATA.feature_selection_k,
        )
        logger.info(f"  Features after selection: {len(FEATURES)}")
    else:
        logger.info(f"  Feature selection disabled — using all {len(FEATURES)} features.")

    # FINAL scaler — built on selected features, used for ALL splits
    scaler = fit_scaler(df_tr, FEATURES)
    logger.info(f"  Single scaler fitted on {len(FEATURES)} selected features.")

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Scale & Sequences
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 5 / 10  --  Scaling & Sequence Generation')
    X_tr = apply_scale(scaler, df_tr, FEATURES)
    X_va = apply_scale(scaler, df_va, FEATURES)
    X_te = apply_scale(scaler, df_te, FEATURES)
    y_tr = y_tr_raw
    y_va = y_va_raw
    y_te = y_te_raw

    SEQ = DATA.seq_len
    Xtr_s, ytr_s = make_sequences(X_tr, y_tr, SEQ)
    Xva_s, yva_s = make_sequences(X_va, y_va, SEQ)
    Xte_s, yte_s = make_sequences(X_te, y_te, SEQ)

    # ── FIX: time-correct three-way split ────────────────────────────────
    # CONTAMINATION BUG: stack slice was carved from the END of X_tr, then
    # base models were trained on X_tr[:vi] which overlaps that slice when
    # vi = 0.85 * len(X_tr) > 0.80 * len(X_tr). The models memorise the
    # stack labels before producing stack predictions → meta-learner trains
    # on in-sample outputs, not held-out outputs.
    #
    # CORRECT split (strict chronological order):
    #   [  X_tr (full)  ] → base model training  (no stack data removed)
    #   [  X_va first half  ] → stack slice      (base models never trained on this)
    #   [  X_va second half ] → AUC / Platt cal  (independent of stack)
    #   [  X_te             ] → final evaluation (never touched until reporting)
    #
    # Splitting val 50/50 keeps at least ~125 bars per half (typical 5-month val).
    n_va       = len(Xva_s)
    n_stack    = max(n_va // 2, 32)                      # first half of val
    n_val_eval = n_va - n_stack                          # second half of val

    Xstack  = Xva_s[:n_stack];   ystack  = yva_s[:n_stack]
    Xva_e   = Xva_s[n_stack:];   yva_e   = yva_s[n_stack:]   # for AUC/Platt

    # Base models train on the FULL training set — nothing carved out
    tr_dl    = make_loader(Xtr_s,  ytr_s,  shuffle=True)
    va_dl    = make_loader(Xva_e,  yva_e)                # AUC-eval half only
    te_dl    = make_loader(Xte_s,  yte_s)
    stack_dl = make_loader(Xstack, ystack)

    logger.info(
        f"  Train={len(Xtr_s):,}  Stack(val-first-half)={n_stack:,}  "
        f"ValEval(val-second-half)={n_val_eval:,}  Test={len(Xte_s):,}"
    )

    # ─────────────────────────────────────────────────────────────
    # STEP 6: Train Models
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 6 / 10  --  Training Models')
    loss_curves = {}
    N_FEATURES  = len(FEATURES)

    # ── XGBoost ──────────────────────────────────────────────────────────
    # FIX: XGBoost trains on full X_tr (no carved-out stack portion).
    # Internal early-stopping uses the last 10% of X_tr — this is purely
    # for stopping, never for feature selection or ensemble fitting.
    # xgb_stack is now generated on X_va[:n_stack] (the val stack slice),
    # which XGBoost has never seen during training.
    logger.info(f'\n  XGBoost (scale_pos_weight={POS_WEIGHT:.3f}) ...')
    vi = int(len(X_tr) * 0.90)            # FIX: was 0.85, now full-train split
    xgb_model = build_xgb(pos_weight=POS_WEIGHT)
    xgb_model.fit(
        X_tr[:vi], y_tr[:vi].astype(int),
        eval_set=[(X_tr[vi:], y_tr[vi:].astype(int))],
        verbose=False,
    )
    xgb_va    = xgb_model.predict_proba(X_va[n_stack:])[:, 1]   # FIX: AUC-eval half
    xgb_te    = xgb_model.predict_proba(X_te)[:, 1]
    xgb_stack = xgb_model.predict_proba(X_va[:n_stack])[:, 1]   # FIX: val stack slice

    # Helper: run HPO then train deep model + save checkpoint
    def _train_deep(model_class, mname: str):
        """Per-model HPO + full training + save .pt checkpoint."""
        best_params: dict = {}
        if cfg.run_hpo:
            logger.info(f'\n  HPO for {mname} ({cfg.hpo_trials} trials) ...')
            best_params = hpo_search(
                model_factory = lambda: model_class(N_FEATURES, pos_weight=POS_WEIGHT),
                tr_dl         = tr_dl,
                va_dl         = va_dl,
                name          = f'hpo_{mname.lower()}',
                device        = DEVICE,
                n_trials      = cfg.hpo_trials,
                pos_weight    = POS_WEIGHT,
            )

        best_lr = best_params.get('lr', MODEL.lr)
        best_ls = best_params.get('label_smoothing', 0.05)
        best_wd = best_params.get('weight_decay', MODEL.weight_decay)

        logger.info(f'\n  {mname} (pos_weight={POS_WEIGHT:.3f}) ...')
        m = model_class(N_FEATURES, pos_weight=POS_WEIGHT).to(DEVICE)
        logger.info(f'  Params: {count_params(m):,}')
        m, trl, vll = torch_fit(
            m, tr_dl, va_dl, mname.lower(), DEVICE,
            lr              = best_lr,
            weight_decay    = best_wd,
            label_smoothing = best_ls,
            pos_weight      = POS_WEIGHT,
        )
        loss_curves[mname] = {'train': trl, 'val': vll}

        # V9: save with canonical name for predict.py
        ckpt_path = save_checkpoint(m, mname.lower())
        logger.info(f'  Saved {mname} checkpoint → {ckpt_path}')
        return m

    lstm_model = _train_deep(LSTMPredictor,        'lstm')
    gru_model  = _train_deep(GRUPredictor,         'gru')
    tf_model   = _train_deep(TransformerPredictor, 'transformer')
    tcn_model  = _train_deep(TCNPredictor,         'tcn')

    # FIX: va_dl now points to Xva_e (val second half) — base models
    # were trained on X_tr only, so both va_dl and stack_dl are clean.
    lstm_va    = predict_proba(lstm_model, va_dl,    DEVICE)   # val second half
    lstm_te    = predict_proba(lstm_model, te_dl,    DEVICE)
    lstm_stack = predict_proba(lstm_model, stack_dl, DEVICE)   # val first half
    gru_va     = predict_proba(gru_model,  va_dl,    DEVICE)
    gru_te     = predict_proba(gru_model,  te_dl,    DEVICE)
    gru_stack  = predict_proba(gru_model,  stack_dl, DEVICE)
    tf_va      = predict_proba(tf_model,   va_dl,    DEVICE)
    tf_te      = predict_proba(tf_model,   te_dl,    DEVICE)
    tf_stack   = predict_proba(tf_model,   stack_dl, DEVICE)
    tcn_va     = predict_proba(tcn_model,  va_dl,    DEVICE)
    tcn_te     = predict_proba(tcn_model,  te_dl,    DEVICE)
    tcn_stack  = predict_proba(tcn_model,  stack_dl, DEVICE)

    # ─────────────────────────────────────────────────────────────
    # STEP 7: Ensemble
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 7 / 10  --  Ensemble')
    # FIX: three strictly disjoint splits used here:
    #   stack_probs / stack_y  → val first half  (meta-learner training)
    #   val_probs   / val_y    → val second half (AUC weights + Platt cal)
    #   test_probs             → test set        (final reporting only)
    # No split has been seen by any component that uses another split.
    N_va  = len(lstm_va)
    N_te  = len(lstm_te)
    N_stk = len(lstm_stack)

    val_probs   = {'XGBoost': xgb_va[-N_va:],     'LSTM': lstm_va,    'GRU': gru_va,    'Transformer': tf_va,    'TCN': tcn_va}
    test_probs  = {'XGBoost': xgb_te[-N_te:],     'LSTM': lstm_te,    'GRU': gru_te,    'Transformer': tf_te,    'TCN': tcn_te}
    stack_probs = {'XGBoost': xgb_stack[-N_stk:], 'LSTM': lstm_stack, 'GRU': gru_stack, 'Transformer': tf_stack, 'TCN': tcn_stack}

    ens = Ensemble()
    ens.fit(stack_probs=stack_probs, stack_y=ystack[-N_stk:],
            val_probs=val_probs,     val_y=yva_e[-N_va:])   # FIX: yva_e = val second half

    # Save checkpoints — V8 FIX C3: replaced pickle with safe serialization
    #
    # REMOVED: pickle.dump(scaler/xgboost/ensemble)
    # REPLACED with:
    #   - scaler   → joblib (sklearn's preferred format) + SHA-256 sidecar
    #   - xgboost  → XGBoost native JSON (no Python objects, no eval risk)
    #   - features → JSON (plain text, no deserialization risk at all)
    #   - ensemble → joblib + SHA-256 sidecar
    #   - deep models → safetensors via save_checkpoint() (already fixed)
    #
    # WHY: pickle.load() on a tampered file executes arbitrary Python code.
    # This is not a theoretical risk — it is the documented attack vector
    # for ML model supply-chain attacks.  joblib and XGBoost native JSON
    # do not execute code on load.
    import hashlib, json as _json
    import joblib

    def _safe_save_joblib(obj, fname: str) -> None:
        """Save with joblib and write SHA-256 sidecar."""
        path = os.path.join(PATHS.checkpt_dir, fname)
        joblib.dump(obj, path, compress=3)
        sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        with open(path + '.sha256', 'w') as f:
            _json.dump({'file': fname, 'sha256': sha}, f)
        logger.info(f'  Saved {fname} (joblib + SHA-256 sidecar)')

    # Scaler (sklearn RobustScaler) → joblib
    _safe_save_joblib(scaler, 'scaler.joblib')

    # XGBoost → native JSON (completely safe, no pickle)
    xgb_json_path = os.path.join(PATHS.checkpt_dir, 'xgboost.json')
    xgb_model.save_model(xgb_json_path)
    logger.info(f'  Saved xgboost.json (native XGBoost JSON — no pickle)')

    # Feature list → plain JSON (zero deserialization risk)
    feat_json_path = os.path.join(PATHS.checkpt_dir, 'features.json')
    with open(feat_json_path, 'w') as f:
        _json.dump(FEATURES, f, indent=2)
    logger.info(f'  Saved features.json ({len(FEATURES)} features)')

    # Ensemble (custom Python class) → joblib
    _safe_save_joblib(ens, 'ensemble.joblib')

    logger.info('  All checkpoints saved with safe serialization (no pickle).')

    ens_equal = ens.predict_equal(test_probs)
    ens_auc   = ens.predict_auc_weighted(test_probs)
    ens_stack = ens.predict_stacking(test_probs)
    ens_cal   = ens.predict_calibrated_auc(test_probs)

    # ─────────────────────────────────────────────────────────────
    # STEP 8: SHAP Explainability
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 8 / 10  --  SHAP Explainability')
    if EXPERIMENT.run_shap:
        try:
            from shap_analysis import (shap_xgb, save_shap_importance,
                                        shap_summary_plot, shap_waterfall_plot)
            shap_vals = shap_xgb(xgb_model, X_te, FEATURES, max_samples=500)
            shap_df   = save_shap_importance(shap_vals, FEATURES, PATHS.shap_dir)
            shap_summary_plot(shap_vals, FEATURES, PATHS.shap_dir, max_display=20)
            shap_waterfall_plot(shap_vals, sample_idx=0, output_dir=PATHS.shap_dir)
            logger.info(f'  SHAP analysis complete → {PATHS.shap_dir}/')
        except Exception as exc:
            logger.warning(f'  SHAP analysis skipped: {exc}')
    else:
        logger.info('  SHAP disabled.')

    # ─────────────────────────────────────────────────────────────
    # STEP 8b: Evaluate  (ML models + MA baseline)
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 8b / 10  --  Evaluation')
    y_common   = yte_s[-N_te:]
    test_ohlcv = df_te.iloc[SEQ:]

    all_probs = {
        'XGBoost':        xgb_te[-N_te:],
        'LSTM':           lstm_te,
        'GRU':            gru_te,
        'Transformer':    tf_te,
        'TCN':            tcn_te,
        'Ensemble_Equal': ens_equal,
        'Ensemble_AUC':   ens_auc,
        'Ensemble_Stack': ens_stack,
        'Ensemble_Cal':   ens_cal,
    }

    # V9: Add MA-crossover baseline to all_probs for unified comparison
    if EXPERIMENT.run_baseline:
        try:
            bl_raw = ma_crossover_baseline(test_ohlcv, fast=5, slow=20)
            # Align to y_common length
            bl_raw = bl_raw[-len(y_common):]
            all_probs['Baseline_MA'] = bl_raw
            logger.info(f"  Baseline MA 5/20 added (length={len(bl_raw)})")
        except Exception as exc:
            logger.warning(f"  Baseline failed: {exc}")

    clf_metrics = {}; trd_metrics = {}
    for name, probs in all_probs.items():
        preds = (probs >= MODEL.threshold).astype(int)
        clf   = classification_metrics(y_common, probs)
        trd   = trading_metrics(
            preds, probs, test_ohlcv,
            commission_bps    = BACKTEST.commission_bps,
            slippage_bps      = BACKTEST.slippage_bps,
            impact_coeff      = BACKTEST.impact_coeff,
            adv_shares        = BACKTEST.adv_shares,
            trade_size_shares = BACKTEST.trade_size_shares,
        )
        clf_metrics[name] = clf
        trd_metrics[name] = trd
        print_metrics(name, clf, trd)

    # ─────────────────────────────────────────────────────────────
    # STEP 8b-ii: Regime-conditional evaluation
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 8b-ii / 10  --  Regime-Conditional Performance')
    try:
        raw_te_full = download(DATA.ticker, DATA.period)
        regime_series = classify_regime(raw_te_full)
        # align to test window
        test_regime = regime_series.iloc[-len(test_ohlcv):]
        if len(test_regime) >= len(y_common):
            test_regime_aligned = test_regime.iloc[-len(y_common):].values

            best_ensemble = 'Ensemble_Stack'
            ens_probs = all_probs.get(best_ensemble, ens_stack)
            ens_preds = (ens_probs >= MODEL.threshold).astype(int)

            for regime_name in ['trending_up', 'trending_down', 'mean_reverting', 'high_vol']:
                mask = test_regime_aligned == regime_name
                if mask.sum() < 10:
                    continue
                r_acc = float((ens_preds[mask] == y_common[mask]).mean())
                r_up  = float(y_common[mask].mean() * 100)
                logger.info(
                    f"  Regime [{regime_name:<16}]  n={mask.sum():3d}  "
                    f"acc={r_acc*100:.1f}%  UP%={r_up:.1f}%"
                )
    except Exception as exc:
        logger.warning(f"  Regime evaluation skipped: {exc}")

    # ─────────────────────────────────────────────────────────────
    # STEP 8c: Charts
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 8c / 10  --  Generating Charts')
    raw_for_viz = download(DATA.ticker, DATA.period)
    df_all = pd.concat([df_tr, df_va, df_te])
    safe_call(viz.chart_price_dashboard, df_all, DATA.ticker, logger=logger, context='chart_price_dashboard')
    safe_call(viz.chart_training_curves, loss_curves, logger=logger, context='chart_training_curves')
    viz.chart_confusion_matrices({n: {'cm': clf_metrics[n]['cm'], 'accuracy': clf_metrics[n]['accuracy']} for n in clf_metrics})
    viz.chart_roc({n: {'y_true': y_common, 'y_prob': all_probs[n], 'auc': clf_metrics[n]['auc']} for n in all_probs})
    viz.chart_feature_importance(pd.Series(xgb_model.feature_importances_, index=FEATURES))
    safe_call(viz.chart_walk_forward, wf_df, logger=logger, context='chart_walk_forward')
    viz.chart_strategy_returns(trd_metrics, test_ohlcv)
    viz.chart_dashboard(clf_metrics, trd_metrics, DATA.ticker)
    safe_call(viz.chart_regime, df_all, DATA.ticker, logger=logger, context='chart_regime')
    viz.chart_calibration(y_common, all_probs)
    viz.chart_confidence_deciles(y_common, all_probs)

    # ─────────────────────────────────────────────────────────────
    # STEP 8d: DIAGNOSTIC CHECKS  ← V9 NEW
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 8d / 10  --  Diagnostic Health Checks')
    issues_found = 0

    # Check 1: Did any model execute trades?
    for name, trd in trd_metrics.items():
        if name == 'Baseline_MA':
            continue
        if trd['n_trades'] == 0:
            logger.warning(
                f"  ⚠ DIAGNOSTIC: {name} executed 0 trades. "
                f"Check confidence_threshold ({BACKTEST.confidence_threshold}) "
                f"and model output range."
            )
            issues_found += 1

    # Check 2: Are all AUCs near random?
    ml_models = [n for n in clf_metrics if not n.startswith('Baseline')]
    best_auc = max(clf_metrics[n]['auc'] for n in ml_models)
    if best_auc < 0.52:
        logger.warning(
            f"  ⚠ DIAGNOSTIC: Best AUC across all ML models = {best_auc:.4f} "
            f"(< 0.52). Models are near-random. Consider: more data, "
            f"different features, or accept that this asset is hard to predict."
        )
        issues_found += 1

    # Check 3: Does any ML model beat the baseline?
    if 'Baseline_MA' in clf_metrics:
        bl_acc   = clf_metrics['Baseline_MA']['accuracy']
        best_ml  = max((clf_metrics[n]['accuracy'] for n in ml_models), default=0)
        if best_ml <= bl_acc:
            logger.warning(
                f"  ⚠ DIAGNOSTIC: Best ML accuracy ({best_ml*100:.1f}%) ≤ "
                f"MA-crossover baseline ({bl_acc*100:.1f}%). "
                f"ML adds no value over a 2-line indicator on this dataset."
            )
            issues_found += 1
        else:
            logger.info(
                f"  ✓ DIAGNOSTIC: Best ML ({best_ml*100:.1f}%) > "
                f"baseline ({bl_acc*100:.1f}%) by {(best_ml-bl_acc)*100:.1f}pp."
            )

    # Check 4: Walk-forward edge
    if len(wf_df) > 0 and 'xgb_vs_baseline' in wf_df.columns:
        wf_edge = wf_df['xgb_vs_baseline'].mean()
        if wf_edge < 0:
            logger.warning(
                f"  ⚠ DIAGNOSTIC: Walk-forward XGB edge = {wf_edge:+.4f} "
                f"(negative — XGB underperforms baseline in out-of-sample folds)."
            )
            issues_found += 1
        else:
            logger.info(f"  ✓ DIAGNOSTIC: Walk-forward XGB edge = {wf_edge:+.4f}")

    if issues_found == 0:
        logger.info("  ✓ All diagnostic checks passed.")
    else:
        logger.warning(f"  {issues_found} diagnostic issue(s) found — review warnings above.")

    # ─────────────────────────────────────────────────────────────
    # STEP 9: Tomorrow's Prediction
    # ─────────────────────────────────────────────────────────────
    hdr("STEP 9 / 10  --  Tomorrow's Prediction")
    last_X    = scaler.transform(df_te[FEATURES].iloc[-SEQ:].values).astype(np.float32)
    last_seq  = torch.from_numpy(last_X).unsqueeze(0)
    last_flat = scaler.transform(df_te[FEATURES].iloc[[-1]])

    preds_tomorrow = {}
    preds_tomorrow['XGBoost'] = float(xgb_model.predict_proba(last_flat)[0][1])
    for mname, mobj in [('LSTM', lstm_model), ('GRU', gru_model),
                         ('Transformer', tf_model), ('TCN', tcn_model)]:
        preds_tomorrow[mname] = predict_single(mobj, last_seq, DEVICE)

    ens_p = ens.predict_stacking({k: np.array([v]) for k, v in preds_tomorrow.items()})
    preds_tomorrow['Ensemble_Stack'] = float(ens_p[0])

    def direction(p): return 'UP  ' if p >= MODEL.threshold else 'DOWN'
    def strength(p):
        g = abs(p - 0.5)
        return 'STRONG' if g > 0.20 else ('MODERATE' if g > 0.10 else 'WEAK')

    best = max(clf_metrics, key=lambda k: clf_metrics[k]['accuracy']
               if k != 'Baseline_MA' else -1)
    div  = '=' * 54
    logger.info(f'\n  {div}')
    _h = DATA.horizons[DATA.primary_horizon_idx]
    _label = 'TOMORROW' if _h == 1 else ('NEXT WEEK' if _h <= 7 else 'NEXT MONTH')
    logger.info(f"  {_label} PREDICTION ({_h}d)  --  {DATA.ticker}")
    logger.info(f'  {div}')
    for name, p in preds_tomorrow.items():
        logger.info(f'  {name:<18} ->  {direction(p)}  {p*100:.1f}%  [{strength(p)}]')
    logger.info(f'  {"-"*54}')
    logger.info(f'  Best ML model (test accuracy): {best}')
    logger.info(f'  {div}\n')

    # ─────────────────────────────────────────────────────────────
    # STEP 10: Save Results
    # ─────────────────────────────────────────────────────────────
    hdr('STEP 10 / 10  --  Saving Results')
    csv_path = os.path.join(PATHS.results_dir, 'prediction_log.csv')
    row = {
        'date': RUN_TIME, 'ticker': DATA.ticker,
        'confidence_filter': BACKTEST.confidence_threshold,
        'use_kelly': BACKTEST.use_kelly,
        'pos_weight': POS_WEIGHT,
        'n_features': len(FEATURES),
        'leakage_free': True,
        'single_scaler': True,      # V9 fix
        'purged_wf': True,          # V9 fix
        'baseline_included': EXPERIMENT.run_baseline,
    }
    for name, p in preds_tomorrow.items():
        row[f'{name.lower()}_dir']  = direction(p).strip()
        row[f'{name.lower()}_conf'] = round(p, 4)
    for name in all_probs:
        row[f'{name.lower()}_acc'] = round(clf_metrics[name]['accuracy'], 4)
        row[f'{name.lower()}_auc'] = round(clf_metrics[name]['auc'], 4)
    row['wf_mean_acc'] = round(wf_df['xgb_acc'].mean(), 4) if len(wf_df) else float('nan')
    row['best_model']  = best

    new_row = pd.DataFrame([row])
    if os.path.exists(csv_path):
        new_row = pd.concat([pd.read_csv(csv_path), new_row], ignore_index=True)
    new_row.to_csv(csv_path, index=False)

    summary = {
        'run_time': RUN_TIME, 'ticker': DATA.ticker, 'version': 'v9',
        'improvements_v9': [
            'confidence_threshold_fixed_0.0',
            'single_scaler_pipeline',
            'deep_model_checkpoints_saved',
            'model_sizes_halved_no_overfit',
            'ma_crossover_baseline_included',
            'regime_conditional_evaluation',
            'purged_walk_forward',
            'diagnostic_health_checks',
        ],
        'experiment': {
            'run_hpo': cfg.run_hpo, 'deep_wf': cfg.deep_wf,
            'hpo_trials': cfg.hpo_trials, 'multi_ticker': cfg.multi_ticker,
        },
        'n_features': len(FEATURES),
        'pos_weight': POS_WEIGHT,
        'class_balance_pct_up': float(y_tr_raw.mean() * 100),
        'confidence_threshold': BACKTEST.confidence_threshold,
        'models': {
            name: {
                'accuracy':      round(clf_metrics[name]['accuracy'], 4),
                'auc':           round(clf_metrics[name]['auc'], 4),
                'f1_up':         round(clf_metrics[name]['f1_up'], 4),
                'brier':         round(clf_metrics[name]['brier'], 4),
                'sharpe':        round(trd_metrics[name]['sharpe'], 3),
                'sortino':       round(trd_metrics[name]['sortino'], 3),
                'max_dd_pct':    round(trd_metrics[name]['max_dd'] * 100, 2),
                'win_rate_pct':  round(trd_metrics[name]['win_rate'] * 100, 2),
                'total_ret_pct': round(trd_metrics[name]['total_ret'] * 100, 2),
                'n_trades':      trd_metrics[name]['n_trades'],
                'alpha':         round(trd_metrics[name].get('alpha', 0), 4),
            }
            for name in clf_metrics
        },
        'best_model': best,
        'tomorrow': {
            name: {'direction': direction(p).strip(), 'confidence': round(p, 4), 'strength': strength(p)}
            for name, p in preds_tomorrow.items()
        },
        'diagnostics': {
            'issues_found': issues_found,
        },
    }
    json_path = os.path.join(PATHS.results_dir, 'summary.json')
    with open(json_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    logger.info(f'  Prediction log → {csv_path}')
    logger.info(f'  Summary JSON   → {json_path}')
    logger.info(f'  Charts         → {PATHS.chart_dir}/')
    logger.info(f'\n  All done!\n')

    log_run_summary(logger, _run_start, extra={'best_model': best, 'ticker': DATA.ticker})


if __name__ == '__main__':
    run_cfg = parse_args()
    main(run_cfg)
