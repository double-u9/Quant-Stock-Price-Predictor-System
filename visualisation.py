"""
visualisation.py  —  Chart Generation (stub + full implementation).

This module was not included in the v1 zip but is imported by main.py.
All chart functions are safe to call — they log a warning and return
gracefully if matplotlib or required data is unavailable.

V2 additions:
  - chart_shap_importance(): bar chart of top SHAP features.
  - All charts now use the dark theme from COLORS config.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import COLORS, PATHS

logger = logging.getLogger(__name__)

# ── Matplotlib setup ──────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    _MPL = True
except ImportError:
    _MPL = False
    logger.warning("  matplotlib not installed — all charts will be skipped.")


def _save(fig, name: str, subdir: str = '') -> None:
    """Save a figure to the chart directory."""
    if not _MPL:
        return
    out_dir = os.path.join(PATHS.chart_dir, subdir) if subdir else PATHS.chart_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    try:
        fig.savefig(path, dpi=120, bbox_inches='tight',
                    facecolor=COLORS['bg'])
        logger.info(f"  Chart saved → {path}")
    except Exception as exc:
        logger.warning(f"  chart save failed ({name}): {exc}")
    finally:
        plt.close(fig)


def _dark_fig(nrows=1, ncols=1, figsize=(12, 6)):
    """Create a dark-themed figure."""
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize,
                           facecolor=COLORS['bg'])
    if hasattr(ax, '__iter__'):
        axes = ax.flatten() if hasattr(ax, 'flatten') else list(ax)
        for a in axes:
            a.set_facecolor(COLORS['panel'])
            a.tick_params(colors=COLORS['neutral'])
            a.xaxis.label.set_color(COLORS['neutral'])
            a.yaxis.label.set_color(COLORS['neutral'])
            for spine in a.spines.values():
                spine.set_edgecolor(COLORS['neutral'])
    else:
        ax.set_facecolor(COLORS['panel'])
        ax.tick_params(colors=COLORS['neutral'])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS['neutral'])
    return fig, ax


# ════════════════════════════════════════════════════════════════
#  CHART 1 — Price Dashboard
# ════════════════════════════════════════════════════════════════

def chart_price_dashboard(df: pd.DataFrame, ticker: str) -> None:
    if not _MPL or df is None or len(df) == 0:
        return
    try:
        fig, axes = _dark_fig(nrows=2, ncols=1, figsize=(14, 8))
        ax1, ax2  = axes

        ax1.plot(df.index, df['Close'], color=COLORS['white'], lw=1.0, label='Close')
        for w, col in [(20, COLORS['lstm']), (50, COLORS['xgb']), (200, COLORS['ensemble'])]:
            col_name = f'sma_{w}'
            if col_name in df.columns:
                ax1.plot(df.index, df[col_name], color=col, lw=0.8,
                         alpha=0.7, label=f'SMA {w}')
        ax1.set_title(f'{ticker} — Price & SMAs', color=COLORS['white'])
        ax1.legend(fontsize=7, labelcolor=COLORS['neutral'])
        ax1.set_ylabel('Price', color=COLORS['neutral'])

        if 'Volume' in df.columns:
            ax2.bar(df.index, df['Volume'], color=COLORS['neutral'], alpha=0.4, width=1)
            ax2.set_ylabel('Volume', color=COLORS['neutral'])
            ax2.set_title('Volume', color=COLORS['white'])

        fig.suptitle(f'{ticker} Dashboard', color=COLORS['white'], fontsize=13)
        _save(fig, '01_price_dashboard.png')
    except Exception as exc:
        logger.warning(f"  chart_price_dashboard failed: {exc}")


# ════════════════════════════════════════════════════════════════
#  CHART 2 — Training Curves
# ════════════════════════════════════════════════════════════════

def chart_training_curves(loss_curves: Dict) -> None:
    if not _MPL or not loss_curves:
        return
    try:
        n    = len(loss_curves)
        fig, axes = _dark_fig(nrows=1, ncols=n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        else:
            axes = list(axes)

        for ax, (name, curves) in zip(axes, loss_curves.items()):
            tr = curves.get('train', [])
            va = curves.get('val', [])
            ax.plot(tr, color=COLORS['lstm'],     lw=1.2, label='Train')
            ax.plot(va, color=COLORS['ensemble'], lw=1.2, label='Val')
            ax.set_title(name, color=COLORS['white'])
            ax.set_xlabel('Epoch', color=COLORS['neutral'])
            ax.set_ylabel('Loss',  color=COLORS['neutral'])
            ax.legend(fontsize=8, labelcolor=COLORS['neutral'])

        fig.suptitle('Training Curves', color=COLORS['white'], fontsize=12)
        _save(fig, '02_training_curves.png')
    except Exception as exc:
        logger.warning(f"  chart_training_curves failed: {exc}")


# ════════════════════════════════════════════════════════════════
#  CHART 3 — Confusion Matrices
# ════════════════════════════════════════════════════════════════

def chart_confusion_matrices(metrics: Dict) -> None:
    if not _MPL or not metrics:
        return
    try:
        n   = len(metrics)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8),
                                  facecolor=COLORS['bg'])
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for ax, (name, m) in zip(axes_flat, metrics.items()):
            ax.set_facecolor(COLORS['panel'])
            cm  = m.get('cm', np.zeros((2, 2)))
            acc = m.get('accuracy', 0.0)
            im  = ax.imshow(cm, cmap='Blues', aspect='auto')
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['DOWN', 'UP'],   color=COLORS['neutral'])
            ax.set_yticklabels(['DOWN', 'UP'],   color=COLORS['neutral'])
            ax.set_xlabel('Predicted',           color=COLORS['neutral'])
            ax.set_ylabel('Actual',              color=COLORS['neutral'])
            ax.set_title(f'{name}\nacc={acc*100:.1f}%', color=COLORS['white'], fontsize=9)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(int(cm[i, j])), ha='center', va='center',
                            color=COLORS['white'], fontsize=10)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        fig.suptitle('Confusion Matrices', color=COLORS['white'], fontsize=12)
        _save(fig, '03_confusion_matrices.png')
    except Exception as exc:
        logger.warning(f"  chart_confusion_matrices failed: {exc}")


# ════════════════════════════════════════════════════════════════
#  CHART 4 — ROC Curves
# ════════════════════════════════════════════════════════════════

def chart_roc(metrics: Dict) -> None:
    if not _MPL or not metrics:
        return
    try:
        from sklearn.metrics import roc_curve
        fig, ax = _dark_fig(figsize=(7, 7))

        colors = [COLORS['lstm'], COLORS['xgb'], COLORS['tf'],
                  COLORS['tcn'], COLORS['ensemble'],
                  '#aaaaff', '#ffaaaa', '#aaffaa', '#ffaaff']

        for (name, m), color in zip(metrics.items(), colors):
            y_true = m.get('y_true')
            y_prob = m.get('y_prob')
            auc    = m.get('auc', 0.0)
            if y_true is None or y_prob is None:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax.plot(fpr, tpr, color=color, lw=1.2,
                    label=f'{name} (AUC={auc:.3f})')

        ax.plot([0, 1], [0, 1], '--', color=COLORS['neutral'], lw=0.8)
        ax.set_xlabel('False Positive Rate', color=COLORS['neutral'])
        ax.set_ylabel('True Positive Rate',  color=COLORS['neutral'])
        ax.set_title('ROC Curves', color=COLORS['white'])
        ax.legend(fontsize=7, labelcolor=COLORS['neutral'])
        _save(fig, '04_roc_curves.png')
    except Exception as exc:
        logger.warning(f"  chart_roc failed: {exc}")


# ════════════════════════════════════════════════════════════════
#  CHART 5 — Feature Importance
# ════════════════════════════════════════════════════════════════

def chart_feature_importance(importance: pd.Series, top_n: int = 25) -> None:
    if not _MPL or importance is None or len(importance) == 0:
        return
    try:
        top = importance.nlargest(top_n)
        fig, ax = _dark_fig(figsize=(8, 6))
        top.plot(kind='barh', ax=ax, color=COLORS['xgb'], alpha=0.8)
        ax.set_title(f'Top {top_n} XGBoost Feature Importances', color=COLORS['white'])
        ax.set_xlabel('Importance', color=COLORS['neutral'])
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=8)
        _save(fig, '05_feature_importance.png')
    except Exception as exc:
        logger.warning(f"  chart_feature_importance failed: {exc}")


# ════════════════════════════════════════════════════════════════
#  CHART 6 — Walk-Forward
# ════════════════════════════════════════════════════════════════

def chart_walk_forward(wf_df: pd.DataFrame) -> None:
    if not _MPL or wf_df is None or len(wf_df) == 0:
        return
    try:
        fig, axes = _dark_fig(nrows=1, ncols=3, figsize=(14, 4))
        ax1, ax2, ax3 = axes

        folds = wf_df['fold'].values
        ax1.bar(folds, wf_df['xgb_acc'] * 100, color=COLORS['xgb'], alpha=0.8)
        ax1.axhline(50, color=COLORS['neutral'], ls='--', lw=0.8)
        ax1.set_title('Accuracy per Fold', color=COLORS['white'])
        ax1.set_ylabel('Accuracy (%)', color=COLORS['neutral'])
        ax1.set_ylim(40, 70)

        ax2.bar(folds, wf_df['xgb_auc'], color=COLORS['lstm'], alpha=0.8)
        ax2.axhline(0.5, color=COLORS['neutral'], ls='--', lw=0.8)
        ax2.set_title('AUC per Fold', color=COLORS['white'])
        ax2.set_ylabel('AUC', color=COLORS['neutral'])
        ax2.set_ylim(0.3, 0.75)

        if 'xgb_ic' in wf_df.columns:
            colors_ic = [COLORS['up'] if v >= 0 else COLORS['down']
                         for v in wf_df['xgb_ic']]
            ax3.bar(folds, wf_df['xgb_ic'], color=colors_ic, alpha=0.8)
            ax3.axhline(0, color=COLORS['neutral'], ls='--', lw=0.8)
            ax3.set_title('IC per Fold', color=COLORS['white'])
            ax3.set_ylabel('Information Coefficient', color=COLORS['neutral'])

        fig.suptitle('Walk-Forward Validation', color=COLORS['white'])
        _save(fig, '06_walk_forward.png')
    except Exception as exc:
        logger.warning(f"  chart_walk_forward failed: {exc}")


# ════════════════════════════════════════════════════════════════
#  CHART 7 — Strategy Returns
# ════════════════════════════════════════════════════════════════

def chart_strategy_returns(trd_metrics: Dict, ohlcv: pd.DataFrame) -> None:
    if not _MPL or not trd_metrics:
        return
    try:
        fig, ax = _dark_fig(figsize=(14, 6))
        colors  = [COLORS['lstm'], COLORS['xgb'], COLORS['tf'],
                   COLORS['tcn'], COLORS['ensemble'],
                   '#aaaaff', '#ffaaaa', '#aaffaa', '#ffaaff']

        # Buy & hold
        if 'Close' in ohlcv.columns and len(ohlcv) > 1:
            bh = ohlcv['Close'].values
            bh_curve = bh / bh[0]
            ax.plot(range(len(bh_curve)), bh_curve,
                    color=COLORS['neutral'], lw=1.0, ls='--', label='Buy & Hold')

        for (name, m), color in zip(trd_metrics.items(), colors):
            cum = m.get('cum')
            if cum is None or len(cum) == 0:
                continue
            ax.plot(range(len(cum)), cum, color=color, lw=1.0,
                    alpha=0.8, label=f'{name}')

        ax.set_title('Strategy Equity Curves', color=COLORS['white'])
        ax.set_xlabel('Bars', color=COLORS['neutral'])
        ax.set_ylabel('Cumulative Return', color=COLORS['neutral'])
        ax.legend(fontsize=7, labelcolor=COLORS['neutral'])
        ax.axhline(1.0, color=COLORS['neutral'], ls=':', lw=0.6)
        _save(fig, '07_strategy_returns.png')
    except Exception as exc:
        logger.warning(f"  chart_strategy_returns failed: {exc}")


# ════════════════════════════════════════════════════════════════
#  CHART 8 — Summary Dashboard
# ════════════════════════════════════════════════════════════════

def chart_dashboard(clf_metrics: Dict, trd_metrics: Dict, ticker: str) -> None:
    if not _MPL or not clf_metrics:
        return
    try:
        names   = list(clf_metrics.keys())
        accs    = [clf_metrics[n]['accuracy'] * 100 for n in names]
        aucs    = [clf_metrics[n]['auc'] for n in names]
        sharpes = [trd_metrics[n]['sharpe'] for n in names if n in trd_metrics]

        fig, axes = _dark_fig(nrows=1, ncols=3, figsize=(15, 5))
        ax1, ax2, ax3 = axes

        x = range(len(names))
        ax1.bar(x, accs, color=COLORS['lstm'], alpha=0.8)
        ax1.axhline(50, color=COLORS['neutral'], ls='--', lw=0.8)
        ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax1.set_title('Test Accuracy (%)', color=COLORS['white'])

        ax2.bar(x, aucs, color=COLORS['xgb'], alpha=0.8)
        ax2.axhline(0.5, color=COLORS['neutral'], ls='--', lw=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax2.set_title('AUC', color=COLORS['white'])

        if sharpes:
            colors_s = [COLORS['up'] if s > 0 else COLORS['down'] for s in sharpes]
            ax3.bar(range(len(sharpes)), sharpes, color=colors_s, alpha=0.8)
            ax3.set_xticks(range(len(sharpes)))
            ax3.set_xticklabels(names[:len(sharpes)], rotation=45, ha='right', fontsize=7)
            ax3.axhline(0, color=COLORS['neutral'], ls='--', lw=0.8)
            ax3.set_title('Sharpe Ratio', color=COLORS['white'])

        fig.suptitle(f'{ticker} — Model Performance Dashboard', color=COLORS['white'])
        _save(fig, '08_dashboard.png')
    except Exception as exc:
        logger.warning(f"  chart_dashboard failed: {exc}")


# ════════════════════════════════════════════════════════════════
#  CHART 9 — Market Regime
# ════════════════════════════════════════════════════════════════

def chart_regime(df: pd.DataFrame, ticker: str) -> None:
    if not _MPL or df is None or 'regime_trend' not in df.columns:
        return
    try:
        fig, axes = _dark_fig(nrows=2, ncols=1, figsize=(14, 7))
        ax1, ax2  = axes

        ax1.plot(df.index, df['Close'], color=COLORS['white'], lw=0.8)
        ax1.set_title(f'{ticker} Close', color=COLORS['white'])

        rt = df['regime_trend']
        colors_r = [COLORS['up'] if v > 0 else COLORS['down'] for v in rt]
        ax2.bar(df.index, rt, color=colors_r, alpha=0.7, width=1)
        ax2.axhline(0, color=COLORS['neutral'], ls='--', lw=0.6)
        ax2.set_title('Regime Trend Score', color=COLORS['white'])

        _save(fig, '09_regime.png')
    except Exception as exc:
        logger.warning(f"  chart_regime failed: {exc}")


# ════════════════════════════════════════════════════════════════
#  CHART 10 — Calibration
# ════════════════════════════════════════════════════════════════

def chart_calibration(y_true: np.ndarray, all_probs: Dict) -> None:
    if not _MPL or y_true is None:
        return
    try:
        from sklearn.calibration import calibration_curve
        fig, ax = _dark_fig(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], '--', color=COLORS['neutral'], lw=0.8, label='Perfect')

        colors = [COLORS['lstm'], COLORS['xgb'], COLORS['tf'],
                  COLORS['tcn'], COLORS['ensemble']]
        for (name, probs), color in zip(all_probs.items(), colors):
            try:
                prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
                ax.plot(prob_pred, prob_true, 'o-', color=color, lw=1.2,
                        ms=4, label=name)
            except Exception:
                pass

        ax.set_xlabel('Mean Predicted Probability', color=COLORS['neutral'])
        ax.set_ylabel('Fraction of Positives',      color=COLORS['neutral'])
        ax.set_title('Calibration Curves', color=COLORS['white'])
        ax.legend(fontsize=7, labelcolor=COLORS['neutral'])
        _save(fig, '10_calibration.png')
    except Exception as exc:
        logger.warning(f"  chart_calibration failed: {exc}")


# ════════════════════════════════════════════════════════════════
#  CHART 11 — Confidence Deciles
# ════════════════════════════════════════════════════════════════

def chart_confidence_deciles(y_true: np.ndarray, all_probs: Dict) -> None:
    if not _MPL or y_true is None:
        return
    try:
        fig, ax = _dark_fig(figsize=(10, 5))
        colors  = [COLORS['lstm'], COLORS['xgb'], COLORS['tf'],
                   COLORS['tcn'], COLORS['ensemble']]
        width   = 0.8 / max(len(all_probs), 1)

        for i, ((name, probs), color) in enumerate(zip(all_probs.items(), colors)):
            try:
                p = pd.Series(probs)
                t = pd.Series(y_true[:len(probs)])
                try:
                    deciles = pd.qcut(p, 10, labels=False, duplicates='drop')
                except Exception:
                    deciles = pd.qcut(p, 5, labels=False, duplicates='drop')
                hit_rate = t.groupby(deciles).mean()
                offset   = (i - len(all_probs) / 2) * width
                ax.bar(hit_rate.index + offset, hit_rate.values * 100,
                       width=width * 0.9, color=color, alpha=0.7, label=name)
            except Exception:
                pass

        ax.axhline(50, color=COLORS['neutral'], ls='--', lw=0.8)
        ax.set_xlabel('Confidence Decile', color=COLORS['neutral'])
        ax.set_ylabel('UP Hit Rate (%)',   color=COLORS['neutral'])
        ax.set_title('Hit Rate by Confidence Decile', color=COLORS['white'])
        ax.legend(fontsize=7, labelcolor=COLORS['neutral'])
        _save(fig, '11_confidence_deciles.png')
    except Exception as exc:
        logger.warning(f"  chart_confidence_deciles failed: {exc}")


# ════════════════════════════════════════════════════════════════
#  CHART 12 — SHAP Feature Importance  (NEW in V2)
# ════════════════════════════════════════════════════════════════

def chart_shap_importance(shap_df: pd.DataFrame, top_n: int = 20) -> None:
    """Bar chart of mean |SHAP| values from shap_analysis.save_shap_importance()."""
    if not _MPL or shap_df is None or len(shap_df) == 0:
        return
    try:
        top = shap_df.head(top_n)
        fig, ax = _dark_fig(figsize=(8, 6))
        ax.barh(top['feature'], top['mean_abs_shap'],
                color=COLORS['ensemble'], alpha=0.85)
        ax.set_title(f'Top {top_n} Features — Mean |SHAP|', color=COLORS['white'])
        ax.set_xlabel('Mean |SHAP value|', color=COLORS['neutral'])
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=8)
        _save(fig, '12_shap_importance.png')
    except Exception as exc:
        logger.warning(f"  chart_shap_importance failed: {exc}")
