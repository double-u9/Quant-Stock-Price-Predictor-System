"""
shap_analysis.py  —  V1 SHAP Explainability Engine.

Provides model-agnostic feature importance using SHAP (SHapley Additive
exPlanations). Unlike XGBoost's built-in feature importance (which only
measures split frequency/gain), SHAP gives:

  1. Direction: does this feature push predictions toward UP or DOWN?
  2. Magnitude: how much does it shift the probability?
  3. Interaction: conditional on other features?
  4. Per-sample: which features drove THIS specific prediction?

Usage:
    from shap_analysis import (
        shap_xgb, shap_summary_plot, shap_dependence_plot,
        shap_waterfall_plot, save_shap_importance
    )
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import PATHS

logger = logging.getLogger(__name__)


def shap_xgb(
    xgb_model,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 500,
) -> Optional[object]:
    """
    Compute SHAP values for an XGBoost model.

    Uses TreeExplainer (exact, fast) since XGBoost is a tree ensemble.
    Limits to max_samples rows to keep computation tractable.

    Parameters
    ----------
    xgb_model    : fitted XGBClassifier.
    X            : feature matrix (N, F), already scaled.
    feature_names: feature column names.
    max_samples  : max rows to explain (random subsample if N > max_samples).

    Returns
    -------
    shap.Explanation object, or None if shap is not installed.
    """
    try:
        import shap
    except ImportError:
        logger.warning("  shap not installed — run: pip install shap")
        return None

    if len(X) > max_samples:
        idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X   = X[idx]

    try:
        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer(
            pd.DataFrame(X, columns=feature_names)
        )
        logger.info(f"  SHAP: computed for {len(X)} samples, {len(feature_names)} features.")
        return shap_values
    except Exception as exc:
        logger.warning(f"  SHAP computation failed: {exc}")
        return None


def save_shap_importance(
    shap_values,
    feature_names: List[str],
    output_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute mean |SHAP| per feature and save to CSV.

    Returns DataFrame sorted by mean absolute SHAP value (most important first).
    Saved to {shap_dir}/shap_importance.csv.
    """
    if shap_values is None:
        return None

    output_dir = output_dir or PATHS.shap_dir
    os.makedirs(output_dir, exist_ok=True)

    try:
        # shap_values.values shape: (N, F) for binary classification
        vals = shap_values.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]   # take positive class for binary

        mean_abs = np.abs(vals).mean(axis=0)
        df = pd.DataFrame({
            'feature'         : feature_names,
            'mean_abs_shap'   : mean_abs,
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        csv_path = os.path.join(output_dir, 'shap_importance.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"  SHAP importance saved → {csv_path}")
        logger.info(f"  Top-10 SHAP features: {df['feature'].head(10).tolist()}")
        return df

    except Exception as exc:
        logger.warning(f"  save_shap_importance failed: {exc}")
        return None


def shap_summary_plot(
    shap_values,
    feature_names: List[str],
    output_dir: Optional[str] = None,
    max_display: int = 20,
) -> None:
    """
    Generate and save SHAP beeswarm summary plot.

    The beeswarm shows:
      - x-axis: SHAP value (positive = pushes UP prediction)
      - y-axis: features ranked by mean |SHAP|
      - color: feature value (red = high, blue = low)

    Saved to {shap_dir}/shap_summary.png.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  shap/matplotlib not installed — skipping summary plot.")
        return

    if shap_values is None:
        return

    output_dir = output_dir or PATHS.shap_dir
    os.makedirs(output_dir, exist_ok=True)

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        out_path = os.path.join(output_dir, 'shap_summary.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight',
                    facecolor='#0d1117')
        plt.close()
        logger.info(f"  SHAP summary plot saved → {out_path}")
    except Exception as exc:
        logger.warning(f"  shap_summary_plot failed: {exc}")


def shap_waterfall_plot(
    shap_values,
    sample_idx: int = 0,
    output_dir: Optional[str] = None,
) -> None:
    """
    Waterfall plot for a single prediction — shows which features
    contributed to that specific forecast and by how much.

    Saved to {shap_dir}/shap_waterfall_{sample_idx}.png.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if shap_values is None or sample_idx >= len(shap_values):
        return

    output_dir = output_dir or PATHS.shap_dir
    os.makedirs(output_dir, exist_ok=True)

    try:
        shap.waterfall_plot(shap_values[sample_idx], show=False, max_display=15)
        out_path = os.path.join(output_dir, f'shap_waterfall_{sample_idx}.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        logger.info(f"  SHAP waterfall saved → {out_path}")
    except Exception as exc:
        logger.warning(f"  shap_waterfall_plot failed: {exc}")


def shap_dependence_plot(
    shap_values,
    feature_names: List[str],
    feature: str,
    output_dir: Optional[str] = None,
) -> None:
    """
    Dependence plot for one feature: shows how its SHAP value varies
    with its actual value, revealing non-linear relationships.

    Saved to {shap_dir}/shap_dep_{feature}.png.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if shap_values is None or feature not in feature_names:
        return

    output_dir = output_dir or PATHS.shap_dir
    os.makedirs(output_dir, exist_ok=True)

    try:
        fi = feature_names.index(feature)
        shap.dependence_plot(fi, shap_values.values, shap_values.data,
                              feature_names=feature_names,
                              interaction_index='auto', show=False)
        safe_name = feature.replace('/', '_').replace(' ', '_')
        out_path  = os.path.join(output_dir, f'shap_dep_{safe_name}.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  SHAP dependence plot ({feature}) → {out_path}")
    except Exception as exc:
        logger.warning(f"  shap_dependence_plot({feature}) failed: {exc}")
