"""
walk_forward.py  —  V10 Walk-Forward Cross-Validation.

V10 redesign over V3
────────────────────
1. TRUE EXPANDING WINDOW:
   V3 used non-overlapping fixed test chunks with train always starting
   from 40% into the dataset. V10 starts training from bar 0 and grows
   the train window fold-by-fold. Each fold adds one test chunk to the
   end of the prior train window.

   Timeline:
     Fold 1: train[0..T1)  purge  test[T1+purge..T1+purge+W)
     Fold 2: train[0..T2)  purge  test[T2+purge..T2+purge+W)
     ...
     Fold K: train[0..TK)  purge  test[TK+purge..TK+purge+W)
   where T_{k+1} = T_k + W  (train expands by one test window per fold).

2. PURGE GAP >= SEQ_LEN + HORIZON (sequence-aware embargo):
   V3 used purge_days=5 but seq_len=50. A training sequence ending at
   bar T uses bars [T-seq_len+1..T] as input and the label at T+horizon.
   The first safe test bar is T + horizon + 1.
   FIX: effective_purge = max(purge_days, seq_len + horizon)

3. PER-FOLD METRICS (extended):
   V3 reported acc / auc / ic / baseline_acc.
   V10 also reports per fold:
     - train/test date ranges
     - sharpe, sortino, win_rate, max_dd (from long/short signal)
     - fold_gap_days (actual purge used)
     - is_clean (True iff effective_purge >= seq_len + horizon)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import DATA, MODEL, PATHS, SEED
from features import (build_features, add_targets, fit_scaler, apply_scale,
                       make_sequences, get_pos_weight, ma_crossover_baseline)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  TRADING METRIC HELPERS
# ════════════════════════════════════════════════════════════════

def _fold_sharpe(pnl: np.ndarray, ann_factor: float = 252.0) -> float:
    if len(pnl) < 2 or pnl.std() < 1e-10:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(ann_factor))


def _fold_sortino(pnl: np.ndarray, ann_factor: float = 252.0) -> float:
    downside = pnl[pnl < 0]
    if len(downside) < 2 or downside.std() < 1e-10:
        return 0.0
    return float(pnl.mean() / downside.std() * np.sqrt(ann_factor))


def _fold_max_dd(pnl: np.ndarray) -> float:
    cum = np.cumprod(1.0 + np.clip(pnl, -0.99, None))
    roll_max = np.maximum.accumulate(cum)
    dd = (cum - roll_max) / (roll_max + 1e-9)
    return float(dd.min())


# ════════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ════════════════════════════════════════════════════════════════

def run_walk_forward(
    raw: pd.DataFrame,
    device: torch.device,
    deep: bool = False,
    n_walks: int = MODEL.n_walks,
    purge_days: int = DATA.purge_days,
    seq_len: int = DATA.seq_len,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward cross-validation with sequence-aware purge.

    Split structure per fold k  (k = 1 .. n_walks)
    ───────────────────────────────────────────────
      train : raw.iloc[0 : train_end_k]
      purge : raw.iloc[train_end_k : train_end_k + effective_purge]
      test  : raw.iloc[train_end_k + effective_purge :
                       train_end_k + effective_purge + test_window]

    effective_purge = max(purge_days, seq_len + horizon)

    Guarantees
    ----------
    - train_end strictly increases each fold  (expanding window)
    - test windows are non-overlapping and contiguous
    - no sequence in test overlaps with any training label

    Parameters
    ----------
    purge_days : minimum embargo gap in bars (calendar days for daily data).
    seq_len    : model lookback used to compute effective_purge.
    horizon    : label lookahead (1 = next-bar target).

    Returns
    -------
    pd.DataFrame with one row per successful fold.
    """
    # ── Sequence-aware purge ─────────────────────────────────────────────
    effective_purge = max(purge_days, seq_len + horizon)
    if effective_purge > purge_days:
        logger.info(
            f"  [WalkForward] Purge extended: purge_days={purge_days} < "
            f"seq_len+horizon={seq_len}+{horizon}. "
            f"Using effective_purge={effective_purge}."
        )

    n = len(raw)

    # ── Expanding window layout ──────────────────────────────────────────
    # min_train: enough history to build features and sequences
    min_train   = max(300, seq_len * 4)
    test_window = max((n - min_train - effective_purge) // n_walks, 20)

    # train_end for fold k: starts at min_train, grows by test_window each fold
    fold_train_ends = [
        min_train + k * test_window
        for k in range(n_walks)
        if min_train + k * test_window + effective_purge + test_window <= n
    ]

    if not fold_train_ends:
        logger.warning(
            f"  [WalkForward] No valid folds — dataset too short. "
            f"n={n}, min_train={min_train}, test_window={test_window}, "
            f"effective_purge={effective_purge}."
        )
        return pd.DataFrame()

    logger.info(
        f"  [WalkForward] {len(fold_train_ends)} folds  "
        f"test_window={test_window}  effective_purge={effective_purge}  "
        f"(purge_days={purge_days}, seq_len={seq_len}, horizon={horizon})"
    )

    results = []

    for fold_idx, train_end in enumerate(fold_train_ends):
        fold_num   = fold_idx + 1
        test_start = train_end + effective_purge
        test_end   = test_start + test_window
        is_clean   = effective_purge >= (seq_len + horizon)

        def _date(idx):
            if hasattr(raw.index, 'date'):
                try:
                    return str(raw.index[min(idx, n - 1)].date())
                except Exception:
                    pass
            return str(idx)

        raw_tr_fold = raw.iloc[:train_end].copy()
        raw_te_fold = raw.iloc[test_start:test_end].copy()

        if len(raw_tr_fold) < min_train or len(raw_te_fold) < 10:
            logger.warning(
                f"  Fold {fold_num}: too few rows "
                f"(train={len(raw_tr_fold)}, test={len(raw_te_fold)}) — skipping."
            )
            continue

        try:
            # ── Feature engineering (independent per fold) ────────────────
            df_tr = build_features(raw_tr_fold)
            df_te = build_features(raw_te_fold)
            df_tr, target_cols = add_targets(df_tr)
            df_te, _           = add_targets(df_te)
            df_tr = df_tr.replace([np.inf, -np.inf], np.nan).dropna()
            df_te = df_te.replace([np.inf, -np.inf], np.nan).dropna()

            primary  = target_cols[DATA.primary_horizon_idx]
            sma_cols = {f'sma_{w}' for w in DATA.sma_windows}
            BASE     = ({'Open', 'High', 'Low', 'Close', 'Volume'}
                        | set(target_cols) | sma_cols)
            feats    = [c for c in df_tr.columns if c not in BASE]

            X_tr = df_tr[feats].values.astype(np.float32)
            y_tr = df_tr[primary].values.astype(np.float32)
            X_te = df_te[feats].values.astype(np.float32)
            y_te = df_te[primary].values.astype(np.float32)

            pw     = get_pos_weight(y_tr)
            up_pct = float(y_tr.mean() * 100)

            logger.info(
                f"  Fold {fold_num}/{len(fold_train_ends)}  "
                f"train=[{_date(0)}..{_date(train_end-1)}] ({len(X_tr)}d)  "
                f"gap={effective_purge}d  "
                f"test=[{_date(test_start)}..{_date(test_end-1)}] ({len(X_te)}d)  "
                f"UP={up_pct:.0f}%  clean={is_clean}"
            )

            # ── XGBoost ───────────────────────────────────────────────────
            from xgboost import XGBClassifier
            from sklearn.metrics import roc_auc_score
            from scipy.stats import spearmanr

            xgb_params = dict(MODEL.xgb_params)
            xgb_params['scale_pos_weight'] = pw
            xgb_fold = XGBClassifier(**xgb_params)

            vi = int(len(X_tr) * 0.85)
            xgb_fold.fit(
                X_tr[:vi], y_tr[:vi].astype(int),
                eval_set=[(X_tr[vi:], y_tr[vi:].astype(int))],
                verbose=False,
            )

            xgb_prob = xgb_fold.predict_proba(X_te)[:, 1]
            xgb_pred = (xgb_prob >= MODEL.threshold).astype(int)
            xgb_acc  = float((xgb_pred == y_te.astype(int)).mean())

            try:
                xgb_auc = float(roc_auc_score(y_te.astype(int), xgb_prob))
            except Exception:
                xgb_auc = 0.5

            try:
                close_ret = raw_te_fold['Close'].pct_change().dropna().values
                n_ic = min(len(xgb_prob), len(close_ret))
                sp = spearmanr(xgb_prob[:n_ic], close_ret[:n_ic])
                xgb_ic = float(
                    getattr(sp, 'statistic', getattr(sp, 'correlation', 0.0))
                )
            except Exception:
                xgb_ic = 0.0

            # ── Per-fold trading metrics ──────────────────────────────────
            try:
                close_arr    = raw_te_fold['Close'].values
                bar_rets     = np.diff(close_arr) / (close_arr[:-1] + 1e-9)
                n_pnl        = min(len(xgb_pred), len(bar_rets))
                # Long/short: +1 when pred=UP, -1 when pred=DOWN
                signal_pnl   = (
                    np.where(xgb_pred[:n_pnl] == 1, 1.0, -1.0)
                    * bar_rets[:n_pnl]
                )
                xgb_sharpe   = _fold_sharpe(signal_pnl)
                xgb_sortino  = _fold_sortino(signal_pnl)
                xgb_win_rate = float((signal_pnl > 0).mean())
                xgb_max_dd   = _fold_max_dd(signal_pnl)
            except Exception as exc:
                logger.warning(f"  Fold {fold_num}: trading metrics failed: {exc}")
                xgb_sharpe = xgb_sortino = xgb_win_rate = xgb_max_dd = float('nan')

            # ── MA-crossover baseline ─────────────────────────────────────
            try:
                bl_signal = ma_crossover_baseline(raw_te_fold, fast=5, slow=20)
                bl_pred   = bl_signal.astype(int)
                n_common  = min(len(bl_pred), len(y_te))
                bl_acc    = float(
                    (bl_pred[:n_common] == y_te[:n_common].astype(int)).mean()
                )
            except Exception as exc:
                logger.warning(f"  Baseline failed fold {fold_num}: {exc}")
                bl_acc = float('nan')

            row = {
                # provenance
                'fold'            : fold_num,
                'train_start'     : _date(0),
                'train_end'       : _date(train_end - 1),
                'test_start'      : _date(test_start),
                'test_end'        : _date(test_end - 1),
                'fold_gap_days'   : effective_purge,
                'is_clean'        : is_clean,
                'tr_rows'         : len(X_tr),
                'te_rows'         : len(X_te),
                # classification
                'xgb_acc'         : xgb_acc,
                'xgb_auc'         : xgb_auc,
                'xgb_ic'          : xgb_ic,
                # trading
                'xgb_sharpe'      : xgb_sharpe,
                'xgb_sortino'     : xgb_sortino,
                'xgb_win_rate'    : xgb_win_rate,
                'xgb_max_dd'      : xgb_max_dd,
                # baseline
                'baseline_acc'    : bl_acc,
                'xgb_vs_baseline' : (
                    xgb_acc - bl_acc if not np.isnan(bl_acc) else float('nan')
                ),
                # label stats
                'up_pct'          : up_pct,
                'pos_weight'      : pw,
            }

            logger.info(
                f"  Fold {fold_num}  "
                f"acc={xgb_acc*100:.1f}%  auc={xgb_auc:.3f}  ic={xgb_ic:+.4f}  "
                f"sharpe={xgb_sharpe:+.2f}  sortino={xgb_sortino:+.2f}  "
                f"win={xgb_win_rate*100:.1f}%  maxdd={xgb_max_dd*100:.1f}%  "
                f"vs_bl={row['xgb_vs_baseline']:+.3f}"
            )

            # ── Optional deep LSTM per fold ───────────────────────────────
            if deep:
                try:
                    from models import LSTMPredictor
                    from trainer import fit as torch_fit, predict_proba

                    scaler_fold = fit_scaler(df_tr, feats)
                    Xs_tr = apply_scale(scaler_fold, df_tr, feats)
                    Xs_te = apply_scale(scaler_fold, df_te, feats)

                    SEQ = seq_len
                    Xtr_s, ytr_s = make_sequences(Xs_tr, y_tr, SEQ)
                    Xte_s, yte_s = make_sequences(Xs_te, y_te, SEQ)

                    if len(Xtr_s) < 32 or len(Xte_s) < 4:
                        raise ValueError("Fold too small for sequences")

                    tr_dl_f = DataLoader(
                        TensorDataset(
                            torch.from_numpy(Xtr_s), torch.from_numpy(ytr_s)
                        ),
                        batch_size=MODEL.batch_size, shuffle=True, drop_last=True,
                    )
                    te_dl_f = DataLoader(
                        TensorDataset(
                            torch.from_numpy(Xte_s), torch.from_numpy(yte_s)
                        ),
                        batch_size=MODEL.batch_size,
                    )

                    lstm_fold = LSTMPredictor(len(feats), pos_weight=pw).to(device)
                    lstm_fold, _, _ = torch_fit(
                        lstm_fold, tr_dl_f, te_dl_f,
                        name=f'wf_lstm_fold{fold_num}', device=device,
                        epochs=40, patience=8, pos_weight=pw,
                    )
                    lstm_prob  = predict_proba(lstm_fold, te_dl_f, device)
                    lstm_pred  = (lstm_prob >= MODEL.threshold).astype(int)
                    lstm_acc   = float((lstm_pred == yte_s.astype(int)).mean())
                    try:
                        lstm_auc = float(roc_auc_score(yte_s.astype(int), lstm_prob))
                    except Exception:
                        lstm_auc = 0.5

                    row['lstm_acc'] = lstm_acc
                    row['lstm_auc'] = lstm_auc
                    logger.info(
                        f"    LSTM acc={lstm_acc*100:.1f}%  auc={lstm_auc:.3f}"
                    )

                except Exception as exc:
                    logger.warning(f"  Deep WF fold {fold_num} failed: {exc}")

            results.append(row)

        except Exception as exc:
            logger.warning(f"  Walk-forward fold {fold_num} failed: {exc}")

    if not results:
        logger.warning("  Walk-forward: no successful folds.")
        return pd.DataFrame()

    df_wf = pd.DataFrame(results)

    # ── Per-fold display table ────────────────────────────────────────────
    display_cols = [
        'fold', 'train_end', 'test_start', 'test_end',
        'tr_rows', 'te_rows', 'fold_gap_days', 'is_clean',
        'xgb_acc', 'xgb_auc', 'xgb_sharpe', 'xgb_sortino',
        'xgb_win_rate', 'xgb_max_dd', 'xgb_ic',
        'baseline_acc', 'xgb_vs_baseline',
    ]
    display_cols = [c for c in display_cols if c in df_wf.columns]
    logger.info(
        "\n  Per-Fold Results:\n" + df_wf[display_cols].to_string(index=False)
    )

    # ── Aggregate summary ─────────────────────────────────────────────────
    n_clean = int(df_wf['is_clean'].sum())
    n_total = len(df_wf)

    def _fmt(col):
        vals = df_wf[col].dropna()
        return f"{vals.mean():.3f} ± {vals.std():.3f}" if len(vals) else "n/a"

    edge_mean = df_wf['xgb_vs_baseline'].dropna().mean()
    logger.info(
        f"\n  Walk-Forward Summary  "
        f"({n_total} folds, {n_clean} clean, "
        f"effective_purge={effective_purge}d):\n"
        f"  Accuracy   : {_fmt('xgb_acc')}  (baseline: {_fmt('baseline_acc')})\n"
        f"  AUC        : {_fmt('xgb_auc')}\n"
        f"  IC         : {_fmt('xgb_ic')}\n"
        f"  Sharpe     : {_fmt('xgb_sharpe')}\n"
        f"  Sortino    : {_fmt('xgb_sortino')}\n"
        f"  Win Rate   : {_fmt('xgb_win_rate')}\n"
        f"  Max DD     : {_fmt('xgb_max_dd')}\n"
        f"  vs Baseline: {_fmt('xgb_vs_baseline')}  "
        f"({'POSITIVE' if edge_mean > 0 else 'NEGATIVE — ML adds no value'})"
    )

    if n_clean < n_total:
        logger.warning(
            f"  {n_total - n_clean} fold(s) is_clean=False. "
            f"Raise purge_days to at least {seq_len + horizon} to fix."
        )

    return df_wf
