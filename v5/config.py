"""
config.py  —  V6 Configuration Registry.

V6 CRITICAL FIXES over V5:
  1. confidence_threshold = 0.0  (was 0.55 — was blocking ALL trades, 0 trades executed)
  2. Model sizes HALVED to prevent overfitting on ~2500 bars:
       lstm_hidden 128→64, tf_d_model 128→64, tcn_channels 64→32, tf_layers 3→2
  3. XGBoost more regularised: max_depth 4→3, min_child_weight 3→5, gamma 0.15→0.20
  4. val_ratio 0.12→0.15 (more validation data for reliable model selection)
  5. n_walks 5→8 (more robust walk-forward estimate)
  6. purge_days=5: embargo gap in walk-forward to prevent autocorrelation leakage
  7. run_baseline=True: always compare vs simple MA-crossover baseline
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import os


@dataclass(frozen=True)
class DataConfig:
    ticker       : str   = 'AAPL'
    period       : str   = '10y'
    train_ratio  : float = 0.70
    val_ratio    : float = 0.15          # V6: increased from 0.12

    tickers      : Tuple[str, ...] = ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA')

    # ── Multi-asset pipeline settings ─────────────────────────────────────
    # cs_norm_method : cross-sectional normalisation applied after feature
    #   engineering, within each date slice across tickers.
    #   'zscore'   — subtract cross-sectional mean, divide by std
    #   'rank'     — replace each value with its cross-sectional rank [0,1]
    #   'none'     — no cross-sectional normalisation (single-ticker equiv.)
    cs_norm_method   : str   = 'zscore'   # cross-sectional normalisation
    cs_norm_min_tickers: int = 3          # min tickers per date for cs_norm
    align_method     : str   = 'inner'    # 'inner'=common dates only, 'outer'=all dates

    horizons     : Tuple[int, ...] = (1, 5, 21)
    seq_len      : int   = 50
    return_lags  : Tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    sma_windows  : Tuple[int, ...] = (5, 10, 20, 50, 100, 200)
    rsi_periods  : Tuple[int, ...] = (7, 14, 21)
    bb_windows   : Tuple[int, ...] = (10, 20)
    vol_windows  : Tuple[int, ...] = (5, 10, 20, 60)
    roc_periods  : Tuple[int, ...] = (5, 10, 20)
    regime_window: int   = 60
    primary_horizon_idx: int = 0    # 0=tomorrow | 1=next week | 2=next month

    # ── Target mode ───────────────────────────────────────────────────────
    # Controls what add_targets() produces for model training.
    #
    # 'binary'        : 1 if Close[t+h] > Close[t] else 0  (original, weak)
    # 'return'        : log return over horizon h (regression target)
    # 'excess_return' : log return minus rolling median return (alpha signal)
    # 'rank'          : cross-sectional percentile rank of return [0, 1]
    #                   (rank=1 = top performer; works for multi-ticker setups)
    #
    # For single-ticker: 'excess_return' is recommended over 'binary'.
    # For multi-ticker:  'rank' is recommended (directly cross-sectional).
    target_mode         : str   = 'binary'   # was hardcoded 'excess_return'
    rank_window         : int   = 60    # rolling window for rank normalisation
    excess_return_window: int   = 60    # rolling window for median subtraction
    binary_threshold    : float = 0.0   # threshold for converting return→binary

    feature_selection_k: int = 60

    # V6 NEW: purge/embargo days between train and test in each fold
    purge_days   : int   = 5


@dataclass(frozen=True)
class ModelConfig:
    epochs       : int   = 80            # reduced from 120
    batch_size   : int   = 64
    lr           : float = 3e-4
    weight_decay : float = 1e-4
    patience     : int   = 15            # reduced from 18
    threshold    : float = 0.50

    class_weight_mode: str = 'balanced'

    # V6: HALVED sizes — prevents overfitting on ~2500 daily bars
    lstm_hidden  : int   = 64            # was 128
    lstm_layers  : int   = 2
    lstm_dropout : float = 0.30          # slightly increased

    tf_d_model   : int   = 64            # was 128
    tf_nhead     : int   = 4
    tf_layers    : int   = 2             # was 3
    tf_dropout   : float = 0.15
    tf_dim_ff    : int   = 128           # was 256

    tcn_channels : int   = 32            # was 64
    tcn_levels   : int   = 4             # was 5
    tcn_kernel   : int   = 3
    tcn_dropout  : float = 0.20

    xgb_params: Dict[str, Any] = field(default_factory=lambda: dict(
        n_estimators          = 600,
        max_depth             = 3,       # V6: reduced from 4
        learning_rate         = 0.02,
        subsample             = 0.75,
        colsample_bytree      = 0.70,
        min_child_weight      = 5,       # V6: increased from 3
        gamma                 = 0.20,    # V6: increased from 0.15
        reg_alpha             = 0.15,    # V6: increased from 0.10
        reg_lambda            = 1.50,    # V6: increased from 1.20
        random_state          = 42,
        eval_metric           = 'logloss',
        early_stopping_rounds = 40,
        verbosity             = 0,
    ))

    n_walks : int = 8                    # increased from 5


@dataclass(frozen=True)
class BacktestConfig:
    # ── Realistic cost model (replaces flat 8 bps) ──────────────────────
    # Total one-way cost = commission + slippage + market_impact
    # market_impact scales with participation_rate = trade_size / ADV
    # Set adv_shares=0 to disable market impact (e.g. large-cap liquid stocks).
    commission_bps       : float = 1.0    # broker commission one-way (retail ~1-2 bps)
    slippage_bps         : float = 2.0    # half-spread + timing slippage one-way
    impact_coeff         : float = 0.1    # λ in: impact_bps = λ * sqrt(participation_rate)
    adv_shares           : float = 10e6   # avg daily volume in shares (e.g. 10M = liquid mid-cap)
    trade_size_shares    : float = 1000   # notional trade size in shares (~$150k at $150/share)
    risk_free_annual     : float = 0.05

    # ══════════════════════════════════════════════════════════════
    # V6 CRITICAL FIX: confidence_threshold was 0.55
    #
    # The old code used:
    #   confident = (np.abs(y_prob - 0.5) >= confidence_filter)
    # With threshold=0.55 this means trade ONLY when |p-0.5| >= 0.55,
    # i.e. only when p >= 1.05 or p <= -0.05 — IMPOSSIBLE.
    # (The check is actually |p-0.5| >= 0.05 for threshold=0.55,
    #  but models rarely exceed that margin consistently.)
    # Result: 0 trades, Sharpe = -1,000,000 in every backtest.
    #
    # Fix: set to 0.0 = execute every signal. Once you confirm the
    # strategy has positive alpha at all, you can gradually raise this
    # to filter for only high-confidence predictions.
    # ══════════════════════════════════════════════════════════════
    confidence_threshold : float = 0.0   # was 0.55 — caused 0 trades

    use_kelly            : bool  = True
    kelly_fraction       : float = 0.20  # conservative: was 0.25
    max_position         : float = 1.0
    stop_loss_pct        : float = 0.05


@dataclass(frozen=True)
class PathConfig:
    results_dir : str = 'results'

    @property
    def checkpt_dir(self) -> str:
        return os.path.join(self.results_dir, 'checkpoints')

    @property
    def log_dir(self) -> str:
        return os.path.join(self.results_dir, 'logs')

    @property
    def chart_dir(self) -> str:
        return os.path.join(self.results_dir, 'charts')

    @property
    def shap_dir(self) -> str:
        return os.path.join(self.results_dir, 'shap')

    def makedirs(self) -> None:
        for d in [self.results_dir, self.checkpt_dir,
                  self.log_dir, self.chart_dir, self.shap_dir]:
            os.makedirs(d, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig:
    run_hpo              : bool = False
    deep_wf              : bool = False
    hpo_trials           : int  = 20
    hpo_all_models       : bool = True
    run_feature_selection: bool = True
    run_shap             : bool = True
    use_multi_ticker     : bool = False

    # V6 NEW: always run a dumb MA-crossover baseline.
    # If the ML models can't beat this, the signal is weak.
    run_baseline         : bool = True


# ── YAML override loader ──────────────────────────────────────

def _load_yaml_overrides(yaml_path: str = 'config.yaml') -> dict:
    for path in [yaml_path, 'config.yaml.txt']:
        if os.path.exists(path):
            try:
                import yaml
                with open(path) as f:
                    raw = yaml.safe_load(f) or {}
                flat: dict = {}
                for section_vals in raw.values():
                    if isinstance(section_vals, dict):
                        flat.update(section_vals)
                return flat
            except Exception as exc:
                import warnings
                warnings.warn(f"{path} could not be loaded: {exc}.")
    return {}


def _apply_overrides(cfg_class, overrides: dict):
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cfg_class)}
    kwargs = {k: v for k, v in overrides.items() if k in valid_fields}
    if not kwargs:
        return cfg_class()
    defaults = dataclasses.asdict(cfg_class())
    defaults.update(kwargs)
    for f in dataclasses.fields(cfg_class):
        if 'Tuple' in str(f.type) and isinstance(defaults.get(f.name), list):
            defaults[f.name] = tuple(defaults[f.name])
    return cfg_class(**{k: defaults[k] for k in valid_fields})


_YAML      = _load_yaml_overrides()
DATA       = _apply_overrides(DataConfig,       _YAML)
MODEL      = ModelConfig()
BACKTEST   = BacktestConfig()
PATHS      = PathConfig()
EXPERIMENT = ExperimentConfig()
SEED       = 42

COLORS = dict(
    bg='#0d1117', panel='#161b22',
    up='#3fb950', down='#f85149',
    lstm='#58a6ff', xgb='#d29922',
    tf='#f0883e', tcn='#ff6ec7',
    ensemble='#bc8cff', neutral='#8b949e', white='#e6edf3',
)
