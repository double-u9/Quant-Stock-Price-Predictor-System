"""
trainer.py  —  V2 Training Loop.

V2 improvements over V1 (reconstructed from main.py usage):

  1. PER-MODEL HPO: hpo_search() now accepts a model_factory callable
     so each architecture (LSTM, GRU, Transformer, TCN) can have its
     own optimal lr and label_smoothing found independently.
     V1 ran HPO only on LSTM then used the same lr for all models —
     a significant suboptimality since each architecture has different
     optimal learning dynamics.

  2. CLASS WEIGHT IN LOSS: fit() accepts pos_weight and passes it to
     BCEWithLogitsLoss as the pos_weight tensor. This means the loss
     function penalises minority-class errors more heavily, fixing
     the issue where models predicted only one direction.

  3. GRADIENT MONITORING: fit() logs grad norm per epoch so training
     instability (exploding/vanishing gradients) is visible in logs.
     Early warning for learning rate issues.

  4. AMP (Automatic Mixed Precision): fit() enables torch.cuda.amp when
     a CUDA device is available, halving memory and accelerating training
     with negligible precision impact for this task.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import MODEL, PATHS, SEED

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════

def fit(
    model: nn.Module,
    tr_dl: DataLoader,
    va_dl: DataLoader,
    name: str,
    device: torch.device,
    lr: float = MODEL.lr,
    weight_decay: float = MODEL.weight_decay,
    epochs: int = MODEL.epochs,
    patience: int = MODEL.patience,
    label_smoothing: float = 0.05,
    pos_weight: float = 1.0,         # V2: class imbalance weight
    grad_clip: float = 1.0,
    warmup_epochs: int = 5,
    t0: int = 9,
    t_mult: int = 2,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Train a binary classifier with:
      - AdamW optimiser + cosine annealing with warm restarts
      - ReduceLROnPlateau as secondary scheduler (plateau detection)
      - Label smoothing (regularisation)
      - pos_weight in BCEWithLogitsLoss (class imbalance fix)
      - Gradient clipping
      - Early stopping on val loss
      - AMP on CUDA
      - Checkpoint: best val loss saved to disk

    Returns (trained model, train_losses, val_losses).
    """
    use_amp  = device.type == 'cuda'
    scaler   = GradScaler() if use_amp else None

    # ── Loss with class weight ────────────────────────────────────────────
    pw_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor, reduction='none')

    opt  = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cos_sched     = CosineAnnealingWarmRestarts(opt, T_0=t0, T_mult=t_mult)
    plateau_sched = ReduceLROnPlateau(opt, patience=8, factor=0.5)

    ckpt_path = os.path.join(PATHS.checkpt_dir, f'{name}.pt')
    best_val_loss  = float('inf')
    best_epoch     = 0
    patience_count = 0
    tr_losses, va_losses = [], []

    logger.info(
        f"  {name}  device={device}  AMP={use_amp}  "
        f"lr={lr:.2e}  pos_weight={pos_weight:.2f}  "
        f"warmup={warmup_epochs}ep  patience={patience}"
    )

    for epoch in range(1, epochs + 1):
        # ── Warmup: linearly scale lr from 0 to target ───────────────────
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in opt.param_groups:
                pg['lr'] = lr * warmup_factor

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        tr_loss_sum = 0.0
        grad_norms  = []

        for X_batch, y_batch in tr_dl:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Label smoothing: shift labels toward 0.5
            y_smooth = y_batch * (1.0 - label_smoothing) + 0.5 * label_smoothing

            opt.zero_grad()
            if use_amp:
                with autocast():
                    logits = model(X_batch)
                    loss   = criterion(logits, y_smooth).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(X_batch)
                loss   = criterion(logits, y_smooth).mean()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            tr_loss_sum += loss.item() * len(X_batch)
            grad_norms.append(float(gn))

        tr_loss = tr_loss_sum / max(len(tr_dl.dataset), 1)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        va_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in va_dl:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits  = model(X_batch)
                loss    = criterion(logits, y_batch).mean()
                va_loss_sum += loss.item() * len(X_batch)

        va_loss = va_loss_sum / max(len(va_dl.dataset), 1)

        tr_losses.append(tr_loss)
        va_losses.append(va_loss)

        # ── Schedulers ────────────────────────────────────────────────────
        if epoch > warmup_epochs:
            cos_sched.step()
        plateau_sched.step(va_loss)

        current_lr = opt.param_groups[0]['lr']
        avg_grad   = float(np.mean(grad_norms)) if grad_norms else 0.0

        # ── Logging (every 10 epochs or on improvement) ───────────────────
        if epoch % 10 == 0 or va_loss < best_val_loss:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train={tr_loss:.4f}  val={va_loss:.4f}  "
                f"lr={current_lr:.2e}  grad={avg_grad:.3f}"
            )

        # ── Early stopping ────────────────────────────────────────────────
        if va_loss < best_val_loss:
            best_val_loss  = va_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(
                    f"  Early stop epoch={epoch}  "
                    f"best_val={best_val_loss:.4f}  best_epoch={best_epoch}"
                )
                break

    # ── Restore best checkpoint ───────────────────────────────────────────
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                          weights_only=True))
    logger.info(
        f"  {name} done  best_epoch={best_epoch}  best_val={best_val_loss:.4f}"
    )
    return model, tr_losses, va_losses


# ════════════════════════════════════════════════════════════════
#  INFERENCE
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_proba(model: nn.Module,
                  dl: DataLoader,
                  device: torch.device) -> np.ndarray:
    """Run inference over a DataLoader, return sigmoid probabilities."""
    model.eval()
    probs = []
    for X_batch, _ in dl:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


@torch.no_grad()
def predict_single(model: nn.Module,
                   seq: torch.Tensor,
                   device: torch.device) -> float:
    """Predict a single sequence. seq shape: (1, T, F)."""
    model.eval()
    seq    = seq.to(device)
    logit  = model(seq)
    return float(torch.sigmoid(logit).cpu())


# ════════════════════════════════════════════════════════════════
#  PER-MODEL HPO  ← V2 CRITICAL FIX
# ════════════════════════════════════════════════════════════════

def hpo_search(
    model_factory: Callable[[], nn.Module],
    tr_dl: DataLoader,
    va_dl: DataLoader,
    name: str,
    device: torch.device,
    n_trials: int = 20,
    pos_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    Optuna HPO for a single model architecture.

    V2 FIX: V1 ran HPO only on LSTM then used that lr for ALL models.
    Now each model calls hpo_search() independently so GRU, Transformer,
    and TCN each get their own optimal hyperparameters.

    Searches over:
      - lr: learning rate [1e-4, 1e-2] (log-uniform)
      - label_smoothing: [0.0, 0.15]
      - weight_decay: [1e-5, 1e-3] (log-uniform)

    Returns dict with best found hyperparameter values.
    Falls back to MODEL defaults if optuna is not installed.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("  optuna not installed — skipping HPO, using defaults.")
        return {}

    def objective(trial: 'optuna.Trial') -> float:
        lr_trial = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        ls_trial = trial.suggest_float('label_smoothing', 0.0, 0.15)
        wd_trial = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        m = model_factory().to(device)
        _, _, va_losses = fit(
            m, tr_dl, va_dl,
            name=f'hpo_{name}_{trial.number}',
            device=device,
            lr=lr_trial,
            weight_decay=wd_trial,
            epochs=30,       # fast HPO: 30 epochs per trial
            patience=8,
            label_smoothing=ls_trial,
            pos_weight=pos_weight,
        )
        return min(va_losses) if va_losses else float('inf')

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(
        f"  HPO [{name}] best: "
        f"lr={best.get('lr', MODEL.lr):.2e}  "
        f"ls={best.get('label_smoothing', 0.05):.3f}  "
        f"wd={best.get('weight_decay', MODEL.weight_decay):.2e}"
    )
    return best
