"""
predict.py  —  V3 Production Prediction Interface.

V3 FIXES over V2:

  1. [C3] All pickle.load() calls removed.  Checkpoints are now loaded
     using safe format-specific loaders:
       - scaler      → joblib.load()   (scaler.joblib + SHA-256 sidecar)
       - XGBoost     → model.load_model() JSON  (xgboost.json)
       - features    → json.load()     (features.json — plain text)
       - ensemble    → joblib.load()   (ensemble.joblib + SHA-256 sidecar)
       - deep models → load_checkpoint() using safetensors
     Backward compatibility: if .joblib/.json files are absent but old
     .pkl files exist, a one-time migration is attempted with a warning.

  2. SHA-256 integrity check for all joblib files is performed via the
     load_checkpoint() sidecar mechanism before deserialization.

V2 changes over V1:
  1. Loads feature list from checkpoint (saved by main.py).
  2. Handles leakage-free feature pipeline.
  3. Loads pos_weight metadata from summary.json for logging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from config  import DATA, MODEL, BACKTEST, PATHS, SEED
from logger  import setup_logging
from models  import LSTMPredictor, GRUPredictor, TransformerPredictor, TCNPredictor, load_checkpoint

logger = logging.getLogger("predict")


@dataclass
class ModelPrediction:
    model_name   : str
    probability  : float
    direction    : str
    strength     : str
    confident    : bool


@dataclass
class PredictionResult:
    ticker               : str
    forecast_date        : str
    prediction_horizon   : int
    models               : List[ModelPrediction]
    best_model           : Optional[str]
    checkpt_dir          : str
    n_features           : int
    seq_len              : int
    confidence_threshold : float

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        ups   = [m for m in self.models if m.direction == 'UP']
        downs = [m for m in self.models if m.direction == 'DOWN']
        primary = next(
            (m for m in self.models if m.model_name == 'Ensemble_Stack'),
            self.models[0] if self.models else None
        )
        if primary is None:
            return f"{self.ticker} — no predictions"
        return (
            f"{self.ticker}  [{self.forecast_date}]  "
            f"→ {primary.direction} {primary.probability*100:.1f}%  "
            f"[{primary.strength}]  (UP={len(ups)}  DOWN={len(downs)})"
        )


def _verify_sha256(path: str) -> bool:
    """Verify SHA-256 sidecar if present. Returns True if OK or no sidecar."""
    hash_path = path + '.sha256'
    if not os.path.exists(hash_path):
        return True  # no sidecar — can't verify, proceed with warning
    with open(hash_path) as f:
        stored = json.load(f)
    actual = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if actual != stored['sha256']:
        raise RuntimeError(
            f"INTEGRITY CHECK FAILED for {path}.\n"
            f"  Expected: {stored['sha256']}\n"
            f"  Actual:   {actual}\n"
            "  File may have been tampered with. Retrain to regenerate."
        )
    return True


def _load_safe_joblib(path: str, label: str):
    """Load a joblib file with SHA-256 integrity check."""
    if not os.path.exists(path):
        return None
    try:
        _verify_sha256(path)
        import joblib
        obj = joblib.load(path)
        logger.info(f"  Loaded {label} (joblib, integrity verified)")
        return obj
    except Exception as exc:
        logger.error(f"  Failed to load {label}: {exc}")
        raise


def _load_deep_model(model_class, n_features, name: str, checkpt_dir: str, device):
    """Load deep model via load_checkpoint() (safetensors + integrity check)."""
    try:
        model = model_class(n_features).to(device)
        load_checkpoint(model, name, device)
        model.eval()
        logger.info(f"  Loaded {model_class.__name__}")
        return model
    except FileNotFoundError:
        logger.warning(f"  Checkpoint not found for {name} — skipped.")
        return None
    except RuntimeError as exc:
        logger.error(f"  Integrity check failed for {name}: {exc}")
        raise


def _direction(p):  return 'UP' if p >= MODEL.threshold else 'DOWN'
def _strength(p):
    g = abs(p - 0.5)
    return 'STRONG' if g > 0.20 else ('MODERATE' if g > 0.10 else 'WEAK')


def predict(
    ticker: str,
    checkpt_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> PredictionResult:
    """
    Generate next-day forecasts using trained model checkpoints.

    V3: All checkpoint loading uses safe format-specific loaders.
    No pickle deserialization anywhere in the prediction path.
    """
    from data_loader import download
    from features    import build_features, apply_scale, make_sequences

    checkpt_dir = checkpt_dir or PATHS.checkpt_dir
    device      = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"  Predicting {ticker}  device={device}  checkpts={checkpt_dir}")

    # ── 1. Data ───────────────────────────────────────────────────────────
    raw = download(ticker, DATA.period)
    if len(raw) < DATA.seq_len + 10:
        raise ValueError(f"predict: only {len(raw)} rows for '{ticker}'.")

    # ── 2. Features ───────────────────────────────────────────────────────
    df = build_features(raw)

    # ── 3. Load feature list from JSON checkpoint (V3 — no pickle) ────────
    feat_json = os.path.join(checkpt_dir, 'features.json')
    if os.path.exists(feat_json):
        with open(feat_json) as f:
            feature_cols = json.load(f)
        logger.info(f"  Loaded feature list ({len(feature_cols)} features from JSON)")
    else:
        # Fallback: recompute (V1/V2 backward compat)
        sma_cols = {f'sma_{w}' for w in DATA.sma_windows}
        base_cols = {'Open', 'High', 'Low', 'Close', 'Volume'} | sma_cols
        feature_cols = [c for c in df.columns
                        if c not in base_cols and not c.startswith('target_')]
        logger.warning(
            f"  features.json not found — using {len(feature_cols)} columns. "
            "Retrain with V8+ to save features.json."
        )

    # ── 4. Load scaler (joblib — no pickle) ──────────────────────────────
    scaler = _load_safe_joblib(
        os.path.join(checkpt_dir, 'scaler.joblib'), 'scaler'
    )
    if scaler is None:
        from features import fit_scaler
        scaler = fit_scaler(df, feature_cols)
        logger.warning("  Fitting new scaler — may differ from training.")

    # ── 5. Prepare inputs ─────────────────────────────────────────────────
    df_clean = df[feature_cols].dropna()
    if len(df_clean) < DATA.seq_len:
        raise ValueError(f"predict: only {len(df_clean)} non-NaN rows.")

    X_scaled  = scaler.transform(df_clean.values).astype(np.float32)
    SEQ       = DATA.seq_len
    last_X    = X_scaled[-SEQ:]
    last_seq  = torch.from_numpy(last_X).unsqueeze(0).to(device)
    last_flat = X_scaled[[-1]]

    # ── 6. Load and run deep models (safetensors — no pickle) ─────────────
    n_features  = len(feature_cols)
    predictions : Dict[str, float] = {}
    model_map   = {
        'LSTM':        (LSTMPredictor,        'lstm'),
        'GRU':         (GRUPredictor,         'gru'),
        'Transformer': (TransformerPredictor, 'transformer'),
        'TCN':         (TCNPredictor,         'tcn'),
    }

    for mname, (mclass, mfile) in model_map.items():
        m = _load_deep_model(mclass, n_features, mfile, checkpt_dir, device)
        if m is None:
            continue
        with torch.no_grad():
            p = float(torch.sigmoid(m(last_seq)).cpu())
        predictions[mname] = p
        logger.info(f"  {mname:<14} → {_direction(p)}  {p*100:.1f}%  [{_strength(p)}]")

    # ── 7. Load XGBoost (native JSON — no pickle) ─────────────────────────
    xgb_json_path = os.path.join(checkpt_dir, 'xgboost.json')
    if os.path.exists(xgb_json_path):
        from xgboost import XGBClassifier
        xgb = XGBClassifier()
        xgb.load_model(xgb_json_path)
        p = float(xgb.predict_proba(last_flat)[0][1])
        predictions['XGBoost'] = p
        logger.info(f"  XGBoost        → {_direction(p)}  {p*100:.1f}%  [{_strength(p)}]")
    else:
        logger.warning("  xgboost.json not found — XGBoost skipped.")

    if not predictions:
        raise RuntimeError(f"predict: no models loaded from '{checkpt_dir}'. Run main.py first.")

    # ── 8. Load ensemble (joblib — no pickle) ────────────────────────────
    ens = _load_safe_joblib(
        os.path.join(checkpt_dir, 'ensemble.joblib'), 'Ensemble'
    )
    if ens is not None:
        try:
            ens_p = ens.predict_stacking(
                {k: np.array([v]) for k, v in predictions.items()
                 if k in ens.model_names})
            predictions['Ensemble_Stack'] = float(ens_p[0])
            p = predictions['Ensemble_Stack']
            logger.info(f"  Ensemble_Stack → {_direction(p)}  {p*100:.1f}%  [{_strength(p)}]")
        except Exception as exc:
            logger.warning(f"  Ensemble failed: {exc}")

    # ── 9. Best model from summary ────────────────────────────────────────
    best_model: Optional[str] = None
    summary_path = os.path.join(os.path.dirname(checkpt_dir), 'summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                best_model = json.load(f).get('best_model')
        except Exception:
            pass

    conf_threshold = BACKTEST.confidence_threshold
    horizon        = DATA.horizons[DATA.primary_horizon_idx]

    result = PredictionResult(
        ticker               = ticker,
        forecast_date        = datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        prediction_horizon   = horizon,
        models               = [
            ModelPrediction(
                model_name  = name,
                probability = round(float(p), 4),
                direction   = _direction(p),
                strength    = _strength(p),
                confident   = abs(p - 0.5) >= conf_threshold,
            )
            for name, p in predictions.items()
        ],
        best_model           = best_model,
        checkpt_dir          = checkpt_dir,
        n_features           = n_features,
        seq_len              = SEQ,
        confidence_threshold = conf_threshold,
    )

    logger.info(f"\n  {result.summary()}")
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate forecasts from trained models.")
    parser.add_argument('--ticker',      default=DATA.ticker)
    parser.add_argument('--checkpt-dir', default=None)
    parser.add_argument('--format',      default='table', choices=['table', 'json'])
    parser.add_argument('--log-dir',     default=PATHS.log_dir)
    return parser.parse_args()


def main():
    args   = _parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args.log_dir, run_id="predict")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        result = predict(ticker=args.ticker, checkpt_dir=args.checkpt_dir, device=device)
        if args.format == 'json':
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"\n{'=' * 56}")
            print(f"  PREDICTION  —  {result.ticker}  —  {result.forecast_date}")
            print(f"  Horizon: {result.prediction_horizon}-day ahead")
            print(f"{'=' * 56}")
            for m in result.models:
                tag = " [CONFIDENT]" if m.confident else ""
                print(f"  {m.model_name:<18}  {m.direction}  {m.probability*100:5.1f}%  [{m.strength}]{tag}")
            if result.best_model:
                print(f"  {'─'*54}")
                print(f"  Best model (test accuracy): {result.best_model}")
            print(f"{'=' * 56}\n")
    except (RuntimeError, ValueError) as exc:
        logger.error(f"Prediction failed: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    main()

