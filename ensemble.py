"""
ensemble.py  —  V6 Ensemble System.

V6 changes over V5
──────────────────
1. _validate_probs(): shared validation helper called by every predict_*
   method. Catches mismatched keys, wrong array lengths, and NaN/inf values
   BEFORE they corrupt calculations.

2. _align_probs(): truncates all arrays to the shortest common length so
   minor length mismatches (e.g. off-by-one from sequence padding) never
   cause column_stack to crash.

3. _diversity_penalty(): guards against the single-model edge case where
   np.corrcoef returns a scalar instead of a matrix (IndexError crash).

4. fit(): empty-dict guard with clear ValueError; bare except replaced by
   logged warnings that identify which model failed and why.

5. predict_auc_weighted() / predict_stacking(): guard for unfit state
   (self._meta is None, self.auc_weights is None).

6. predict_equal() / predict_auc_weighted(): NaN-aware aggregation —
   per-bar NaN values are excluded rather than poisoning the full output.

7. Public weights property exposes the AUC weight dict for inspection.

8. _is_fitted flag: all predict_* raise RuntimeError with a clear message
   if called before fit().
"""
from __future__ import annotations

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  PLATT CALIBRATION
# ════════════════════════════════════════════════════════════════

class PlattScaler:
    """
    Platt scaling: fits a logistic regression on (raw_prob, y) to convert
    uncalibrated model probabilities into well-calibrated ones.
    """

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1.0, max_iter=500)

    def fit(self, p: np.ndarray, y: np.ndarray) -> 'PlattScaler':
        self._lr.fit(p.reshape(-1, 1), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(p.reshape(-1, 1))[:, 1]


# ════════════════════════════════════════════════════════════════
#  ENSEMBLE
# ════════════════════════════════════════════════════════════════

class Ensemble:
    """
    Combines probability predictions from multiple base models using four
    complementary strategies:

      1. equal_weight    — simple mean; baseline, always available.
      2. auc_weighted    — softmax over diversity-penalised val AUCs.
      3. stacking        — logistic meta-learner on held-out stack slice.
      4. calibrated_auc  — Platt-calibrated probabilities + AUC weights.

    All predict_* methods validate inputs before combining so mismatched
    arrays and NaN values are caught with clear error messages.
    """

    def __init__(self) -> None:
        self.model_names  : List[str]                    = []
        self.auc_weights  : Optional[np.ndarray]         = None
        self._weight_map  : Dict[str, float]             = {}
        self._meta        : Optional[LogisticRegression] = None
        self._meta_sc     : StandardScaler               = StandardScaler()
        self._calibrators : Dict[str, PlattScaler]       = {}
        self._is_fitted   : bool                         = False

    @property
    def weights(self) -> Dict[str, float]:
        """AUC weight per model name (available after fit())."""
        return dict(self._weight_map)

    # ── internal helpers ──────────────────────────────────────────────────

    def _validate_probs(self, probs: Dict[str, np.ndarray],
                        context: str = "") -> None:
        """
        Validate a probs dict before any aggregation step.

        Raises RuntimeError if fit() was not called.
        Raises ValueError for empty dict, unknown keys, wrong shapes,
        or infinite values.
        Logs warnings for NaN values (handled gracefully, not fatal).
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"Ensemble.{context}: fit() must be called before predict_*."
            )
        if not probs:
            raise ValueError(
                f"Ensemble.{context}: received empty probs dict."
            )

        unknown = set(probs.keys()) - set(self.model_names)
        if unknown:
            raise ValueError(
                f"Ensemble.{context}: unknown model names {unknown}. "
                f"Fitted models: {self.model_names}."
            )

        missing = set(self.model_names) - set(probs.keys())
        if missing:
            logger.warning(
                f"  Ensemble.{context}: models missing from probs "
                f"{missing} — they will be skipped."
            )

        for name, arr in probs.items():
            arr = np.asarray(arr)
            if arr.ndim != 1:
                raise ValueError(
                    f"Ensemble.{context}: '{name}' must be 1-D, "
                    f"got shape {arr.shape}."
                )
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            if nan_count > 0:
                logger.warning(
                    f"  Ensemble.{context}: '{name}' has {nan_count} NaN "
                    f"values — affected bars use remaining models only."
                )
            if inf_count > 0:
                raise ValueError(
                    f"Ensemble.{context}: '{name}' has {inf_count} infinite "
                    f"values. Check the model output pipeline."
                )

    def _align_probs(self, probs: Dict[str, np.ndarray],
                     names: List[str]) -> np.ndarray:
        """
        Build an (N, M) matrix, truncated to the shortest array length.

        Off-by-one differences can occur due to sequence padding in deep
        models. Truncating to the minimum is safe — arrays are ordered
        chronologically so we drop trailing rows only.

        Returns
        -------
        np.ndarray of shape (min_len, len(names)), dtype float64.
        """
        arrays  = [np.asarray(probs[n], dtype=np.float64) for n in names]
        lengths = [len(a) for a in arrays]

        if len(set(lengths)) > 1:
            min_len = min(lengths)
            logger.warning(
                f"  Ensemble._align_probs: length mismatch "
                f"{dict(zip(names, lengths))} — truncating to {min_len}."
            )
            arrays = [a[:min_len] for a in arrays]

        return np.column_stack(arrays)   # (N, M)

    def _diversity_penalty(self, probs: Dict[str, np.ndarray],
                            aucs: List[float]) -> np.ndarray:
        """
        Penalise models that are highly correlated with a better model.

        adjusted[i] = auc[i] - sum_{j: auc[j]>auc[i]}
                                max(0, corr[i,j] - 0.5) * 0.1

        Guards against the single-model case where np.corrcoef returns a
        scalar (not a matrix), which previously caused an IndexError.
        """
        names = list(probs.keys())
        n     = len(names)

        if n == 1:
            return np.array(aucs, dtype=np.float64)

        mat    = np.array([np.asarray(probs[nm], dtype=np.float64)
                           for nm in names])
        corr_m = np.corrcoef(mat)   # guaranteed (n, n) for n >= 2

        penalty = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and aucs[j] > aucs[i]:
                    penalty[i] += max(0.0, corr_m[i, j] - 0.5) * 0.1

        return np.array(aucs, dtype=np.float64) - penalty

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self,
            stack_probs : Dict[str, np.ndarray],
            stack_y     : np.ndarray,
            val_probs   : Dict[str, np.ndarray],
            val_y       : np.ndarray) -> 'Ensemble':
        """
        Fit all ensemble components on held-out data.

        Parameters
        ----------
        stack_probs : predictions on the stack slice (logistic meta-learner).
        stack_y     : true labels for the stack slice.
        val_probs   : predictions on the validation set (AUC + calibration).
        val_y       : true labels for the validation set.

        Returns self (fluent interface).
        """
        if not stack_probs:
            raise ValueError(
                "Ensemble.fit(): stack_probs is empty. "
                "At least one model is required."
            )
        if set(stack_probs.keys()) != set(val_probs.keys()):
            raise ValueError(
                "Ensemble.fit(): stack_probs and val_probs must have the "
                f"same model keys.\n"
                f"  stack: {sorted(stack_probs.keys())}\n"
                f"  val:   {sorted(val_probs.keys())}"
            )

        self.model_names = list(stack_probs.keys())

        # ── Step 1: AUC per model ─────────────────────────────────────────
        aucs: List[float] = []
        for name in self.model_names:
            try:
                a = roc_auc_score(val_y, val_probs[name])
                logger.info(f"  {name:<16} val AUC = {a:.4f}")
            except Exception as exc:
                a = 0.5
                logger.warning(
                    f"  {name:<16} AUC failed "
                    f"({type(exc).__name__}: {exc}) — defaulting to 0.5."
                )
            aucs.append(a)

        # ── Step 2: Diversity-penalised softmax weights ───────────────────
        adjusted         = self._diversity_penalty(val_probs, aucs)
        exp_a            = np.exp(4.0 * (adjusted - adjusted.mean()))
        self.auc_weights = exp_a / exp_a.sum()
        self._weight_map = dict(zip(self.model_names, self.auc_weights))
        logger.info(
            "  AUC weights: "
            + "  ".join(f"{n}={w:.3f}" for n, w in self._weight_map.items())
        )

        # ── Step 3: Platt calibration per model ───────────────────────────
        for name in self.model_names:
            try:
                self._calibrators[name] = PlattScaler().fit(
                    val_probs[name], val_y)
            except Exception as exc:
                logger.warning(
                    f"  Platt calibration failed for '{name}' "
                    f"({type(exc).__name__}: {exc}) — excluded from "
                    f"calibrated ensemble."
                )

        # ── Step 4: Stacking meta-learner ─────────────────────────────────
        X_stack         = self._align_probs(stack_probs, self.model_names)
        X_stack         = self._meta_sc.fit_transform(X_stack)
        stack_y_aligned = stack_y[-len(X_stack):]   # trim to aligned length
        self._meta      = LogisticRegression(C=0.5, max_iter=1000,
                                             random_state=42)
        self._meta.fit(X_stack, stack_y_aligned)
        logger.info(
            "  Meta-weights: "
            + "  ".join(f"{n}={w:+.4f}"
                        for n, w in zip(self.model_names,
                                        self._meta.coef_[0]))
        )

        self._is_fitted = True
        return self

    # ── predict ───────────────────────────────────────────────────────────

    def predict_equal(self, tp: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Simple equal-weight average (NaN-aware).

        Per-bar NaN values from individual models are excluded via
        np.nanmean so one bad model output cannot poison the result.
        """
        self._validate_probs(tp, "predict_equal")
        available = [n for n in self.model_names if n in tp]
        matrix    = self._align_probs(tp, available)     # (N, M)
        return np.nanmean(matrix, axis=1)

    def predict_auc_weighted(self, tp: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Weighted average using diversity-penalised AUC weights.

        Weights are renormalised if any model is absent from tp.
        NaN cells are zeroed and per-row weights are renormalised so
        the output remains a valid probability.
        """
        self._validate_probs(tp, "predict_auc_weighted")

        if self.auc_weights is None:
            raise RuntimeError(
                "Ensemble.predict_auc_weighted: auc_weights is None — "
                "call fit() first."
            )

        available = [n for n in self.model_names if n in tp]
        weights   = np.array([self._weight_map[n] for n in available],
                              dtype=np.float64)
        weights   = weights / weights.sum()

        matrix   = self._align_probs(tp, available)      # (N, M)
        nan_mask = np.isnan(matrix)
        matrix   = np.where(nan_mask, 0.0, matrix)
        w_mat    = np.where(nan_mask, 0.0, weights[np.newaxis, :])
        row_sum  = w_mat.sum(axis=1, keepdims=True)
        row_sum  = np.where(row_sum == 0, 1.0, row_sum)  # avoid /0

        return (matrix * w_mat / row_sum).sum(axis=1)

    def predict_stacking(self, tp: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Logistic meta-learner trained on the stack slice.

        The meta-learner learns the optimal linear combination of base
        model outputs, including possible negative weights.
        """
        self._validate_probs(tp, "predict_stacking")

        if self._meta is None:
            raise RuntimeError(
                "Ensemble.predict_stacking: meta-learner is None — "
                "call fit() first."
            )

        available = [n for n in self.model_names if n in tp]
        X = self._align_probs(tp, available)
        X = self._meta_sc.transform(X)
        return self._meta.predict_proba(X)[:, 1]

    def predict_calibrated_auc(self,
                                tp: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Platt-calibrated probabilities combined with AUC weights.

        Calibration corrects systematic over/under-confidence before
        weighting. Falls back to equal-weight if no calibrators exist.
        Weights are looked up by name — missing calibrators never
        misalign the weight vector.
        """
        self._validate_probs(tp, "predict_calibrated_auc")

        cal_probs   : List[np.ndarray] = []
        cal_weights : List[float]      = []
        skipped     : List[str]        = []

        for name in self.model_names:
            if name not in tp:
                skipped.append(name)
                continue
            if name in self._calibrators:
                cal_probs.append(
                    self._calibrators[name].transform(tp[name])
                )
                cal_weights.append(self._weight_map[name])
            else:
                skipped.append(name)

        if skipped:
            logger.debug(
                f"  predict_calibrated_auc: skipped {skipped} "
                f"(no calibrator fitted)."
            )

        if not cal_probs:
            logger.warning(
                "  predict_calibrated_auc: no calibrated models available — "
                "falling back to equal-weight average."
            )
            return self.predict_equal(tp)

        w       = np.array(cal_weights, dtype=np.float64)
        w       = w / w.sum()
        min_len = min(len(a) for a in cal_probs)
        matrix  = np.column_stack([a[:min_len] for a in cal_probs])

        return matrix @ w
