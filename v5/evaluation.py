"""
evaluation.py  —  V8 Evaluation & Reporting Framework.

V8 additions over V7
────────────────────
NEW METRICS
  1.  omega_ratio()        — probability-weighted gain/loss above a threshold.
                             Complements Sharpe/Sortino for non-normal returns.
  2.  var_cvar()           — Value at Risk and Conditional VaR (Expected
                             Shortfall) at a configurable confidence level.
  3.  rolling_sharpe()     — rolling Sharpe over a configurable window.
                             Exposes regime changes hidden by point-in-time stats.
  4.  drawdown_series()    — full underwater curve, worst drawdown duration,
                             and recovery time — not just the scalar minimum.
  5.  turnover_rate()      — average daily fraction of portfolio traded.
                             High turnover inflates live costs vs backtest.
  6.  avg_holding_period() — mean bars per trade; context for cost assumptions.
  7.  benchmark_metrics()  — beta, alpha, information ratio vs buy & hold.
                             Gives a reference point for strategy added value.

REPORTING
  8.  generate_report()    — assembles a structured nested dict containing
                             all classification + trading metrics in one place,
                             ready for JSON serialisation or DataFrame export.
  9.  compare_models()     — takes a dict of {name: report_dict} and returns
                             a ranked pandas DataFrame for cross-model comparison.
  10. EquityCurve dataclass — wraps (cum, daily_strat, index) so the equity
                             curve always carries its DatetimeIndex and can be
                             plotted against dates or aligned across experiments.

ROBUSTNESS FIXES
  11. classification_metrics() — NaN guard on y_prob before sklearn calls.
  12. hit_rate_by_decile()     — fallback to 5 quantiles when all predictions
                                 have identical confidence (zero-width bins).
  13. information_coefficient() — uses .statistic attribute (scipy ≥1.9) with
                                  fallback to .correlation for older scipy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)

from config import BACKTEST, MODEL

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  ANNUALISATION CONSTANTS
# ════════════════════════════════════════════════════════════════

_ANN: Dict[str, float] = {
    "daily":   252.0,
    "weekly":   52.0,
    "monthly":  12.0,
    "hourly": 252.0 * 6.5,
}


# ════════════════════════════════════════════════════════════════
#  EQUITY CURVE CONTAINER
# ════════════════════════════════════════════════════════════════

@dataclass
class EquityCurve:
    """
    Wraps portfolio equity data with its time index.

    Carrying the DatetimeIndex with the curve enables:
      - date-aligned plotting across experiments
      - rolling window calculations with correct time labels
      - direct comparison against benchmark curves on the same axis

    Attributes
    ----------
    cum          : cumulative equity (starts at 1.0).
    daily_ret    : per-period net returns (signal × exec_ret - costs).
    index        : DatetimeIndex aligned to daily_ret.
                   May be None for non-indexed inputs (falls back to int range).
    model_name   : label for charting and reporting.
    """
    cum        : np.ndarray
    daily_ret  : np.ndarray
    index      : Optional[pd.DatetimeIndex] = None
    model_name : str = "strategy"

    def to_series(self) -> pd.Series:
        """Return cumulative equity as a named Series with the time index."""
        idx = self.index if self.index is not None \
              else pd.RangeIndex(len(self.cum))
        return pd.Series(self.cum, index=idx, name=self.model_name)

    def ret_series(self) -> pd.Series:
        """Return daily returns as a named Series with the time index."""
        idx = self.index if self.index is not None \
              else pd.RangeIndex(len(self.daily_ret))
        return pd.Series(self.daily_ret, index=idx, name=self.model_name)


# ════════════════════════════════════════════════════════════════
#  CLASSIFICATION METRICS
# ════════════════════════════════════════════════════════════════

def classification_metrics(y_true: np.ndarray,
                            y_prob: np.ndarray,
                            threshold: float = MODEL.threshold) -> Dict:
    """
    Standard classification metrics for binary direction prediction.

    V8: NaN guard on y_prob prevents cryptic sklearn errors when a
    model returns NaN probabilities (e.g. after a failed calibration).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # ── NaN guard ────────────────────────────────────────────────────────
    nan_mask = ~np.isfinite(y_prob)
    if nan_mask.any():
        n_bad = int(nan_mask.sum())
        logger.warning(
            f"classification_metrics: {n_bad} non-finite y_prob values "
            f"replaced with 0.5 (neutral). Check model output pipeline."
        )
        y_prob = y_prob.copy()
        y_prob[nan_mask] = 0.5

    preds = (y_prob >= threshold).astype(int)
    return dict(
        accuracy = accuracy_score(y_true, preds),
        auc      = roc_auc_score(y_true, y_prob),
        f1_up    = f1_score(y_true, preds, pos_label=1, zero_division=0),
        f1_down  = f1_score(y_true, preds, pos_label=0, zero_division=0),
        brier    = brier_score_loss(y_true, y_prob),
        cm       = confusion_matrix(y_true, preds),
        report   = classification_report(
                       y_true, preds,
                       target_names=['DOWN', 'UP'], zero_division=0),
    )


# ════════════════════════════════════════════════════════════════
#  INFORMATION COEFFICIENT
# ════════════════════════════════════════════════════════════════

def information_coefficient(y_prob: np.ndarray,
                             realized_ret: np.ndarray) -> float:
    """
    Rank IC: Spearman correlation of predicted probability vs
    realised close-to-close return.

    V8: uses .statistic attribute (scipy ≥ 1.9) with fallback to
    .correlation for older scipy versions, eliminating DeprecationWarning.
    """
    n = min(len(y_prob), len(realized_ret))
    if n < 5:
        return 0.0
    result = spearmanr(y_prob[:n], realized_ret[:n])
    # scipy ≥ 1.9 uses .statistic; older versions use .correlation
    corr = getattr(result, 'statistic',
                   getattr(result, 'correlation', np.nan))
    return float(corr) if np.isfinite(corr) else 0.0


# ════════════════════════════════════════════════════════════════
#  CONFIDENCE DECILE ANALYSIS
# ════════════════════════════════════════════════════════════════

def hit_rate_by_decile(y_true: np.ndarray,
                       y_prob: np.ndarray) -> pd.DataFrame:
    """
    Hit rate per confidence decile — should increase monotonically.

    V8: falls back to 5 quantiles when all predictions have identical
    confidence (pd.qcut raises with zero-width bins when confidence
    variance is zero).
    """
    df = pd.DataFrame({'y': y_true, 'p': y_prob})
    df['confidence'] = (df['p'] - 0.5).abs()
    df['pred']       = (df['p'] >= MODEL.threshold).astype(int)
    df['hit']        = (df['pred'] == df['y']).astype(int)

    # Try 10 quantiles; fall back to fewer if confidence values have low variance.
    binned = False
    for n_bins in [10, 5, 2]:
        try:
            labels = pd.qcut(df['confidence'], n_bins,
                              labels=False, duplicates='drop')
            if labels.nunique() >= 1:
                df['decile'] = labels
                binned = True
                break
        except (ValueError, TypeError):
            continue

    if not binned:
        # All confidence values identical — return a single-row summary.
        df['decile'] = 0

    result = (df.groupby('decile')['hit']
                .agg(['mean', 'count'])
                .rename(columns={'mean': 'hit_rate', 'count': 'n'}))

    if result.empty:
        # Fallback: flat aggregate when groupby yields nothing
        result = pd.DataFrame(
            {'hit_rate': [df['hit'].mean()], 'n': [len(df)]},
            index=pd.Index([0], name='decile')
        )
    return result


# ════════════════════════════════════════════════════════════════
#  RETURN-BASED RISK METRICS
# ════════════════════════════════════════════════════════════════

def sharpe_ratio(portfolio_returns: np.ndarray,
                 risk_free_rate: float = 0.0,
                 frequency: str = "daily") -> float:
    """
    Annualised Sharpe Ratio using excess returns in both
    numerator and denominator.

        Sharpe = sqrt(N) * mean(r_excess) / std(r_excess)
        r_excess = r_portfolio − rf_annual / N
    """
    if frequency not in _ANN:
        raise ValueError(f"Unknown frequency '{frequency}'. "
                         f"Options: {list(_ANN.keys())}")

    ret = np.asarray(portfolio_returns, dtype=np.float64).ravel()
    ret = ret[np.isfinite(ret)]
    if len(ret) < 2:
        return 0.0

    ann_factor = _ANN[frequency]
    excess     = ret - risk_free_rate / ann_factor
    mean_e     = excess.mean()
    std_e      = excess.std(ddof=1)

    if std_e < 1e-10:
        return float(np.sign(mean_e) * 1e6)
    return float(np.sqrt(ann_factor) * mean_e / std_e)


def omega_ratio(portfolio_returns: np.ndarray,
                threshold: float = 0.0,
                frequency: str = "daily") -> float:
    """
    Omega Ratio: probability-weighted gains / probability-weighted losses
    relative to a threshold return.

    Unlike Sharpe and Sortino, Omega captures the full shape of the
    return distribution (skewness, kurtosis) — important for financial
    returns which are typically non-normal.

        Omega(L) = E[max(r - L, 0)] / E[max(L - r, 0)]

    Values > 1 indicate the strategy returns more probability-weighted
    gain than loss above the threshold. Omega = 1 at the mean.

    Parameters
    ----------
    portfolio_returns : periodic net strategy returns.
    threshold         : minimum acceptable return per period (default 0).
                        Set to rf_per_period for a risk-adjusted version.
    frequency         : used to convert annual threshold to per-period.

    Returns
    -------
    float  Omega ratio, or inf when there are no losses below threshold.
    """
    ret = np.asarray(portfolio_returns, dtype=np.float64).ravel()
    ret = ret[np.isfinite(ret)]
    if len(ret) < 2:
        return 1.0

    gains  = np.maximum(ret - threshold, 0.0).sum()
    losses = np.maximum(threshold - ret, 0.0).sum()

    if losses < 1e-12:
        return float('inf') if gains > 0 else 1.0
    return float(gains / losses)


def var_cvar(portfolio_returns: np.ndarray,
             confidence: float = 0.95) -> Tuple[float, float]:
    """
    Value at Risk (VaR) and Conditional Value at Risk (CVaR /
    Expected Shortfall) at the given confidence level.

    VaR answers: "What is the worst loss I should expect to see in
    (1-confidence) of all periods?" e.g. 95% VaR = worst 5th percentile.

    CVaR answers: "Given that I AM in the worst (1-confidence) tail,
    what is my expected loss?" CVaR is always ≥ VaR and is preferred
    by risk managers because it is sub-additive and captures tail shape.

    Both are returned as positive numbers representing loss magnitude.

    Parameters
    ----------
    confidence : tail probability level (e.g. 0.95 for 95% VaR).

    Returns
    -------
    (var, cvar) : both as positive floats (loss magnitudes).
    """
    ret = np.asarray(portfolio_returns, dtype=np.float64).ravel()
    ret = ret[np.isfinite(ret)]
    if len(ret) < 2:
        return 0.0, 0.0

    var_pct = 1.0 - confidence
    var     = float(-np.percentile(ret, var_pct * 100))

    # CVaR = mean of returns below the VaR threshold (tail losses)
    tail   = ret[ret <= np.percentile(ret, var_pct * 100)]
    cvar   = float(-tail.mean()) if len(tail) > 0 else var

    return var, cvar


def rolling_sharpe(portfolio_returns: np.ndarray,
                   window: int = 63,
                   risk_free_rate: float = 0.0,
                   frequency: str = "daily") -> np.ndarray:
    """
    Rolling Sharpe Ratio over a trailing window.

    A single Sharpe number hides regime changes — a strategy with
    overall Sharpe=1.5 may have Sharpe=-2 in the most recent quarter.
    Rolling Sharpe exposes these periods of poor performance.

    Parameters
    ----------
    window : rolling window in periods (default 63 ≈ 1 quarter).

    Returns
    -------
    np.ndarray of length len(portfolio_returns), NaN for first window-1
    periods.
    """
    if frequency not in _ANN:
        raise ValueError(f"Unknown frequency: {frequency}")

    ann_factor = _ANN[frequency]
    rf_per     = risk_free_rate / ann_factor
    ret        = np.asarray(portfolio_returns, dtype=np.float64).ravel()
    result     = np.full(len(ret), np.nan)

    for i in range(window - 1, len(ret)):
        window_ret = ret[i - window + 1:i + 1]
        window_ret = window_ret[np.isfinite(window_ret)]
        if len(window_ret) < 2:
            continue
        exc  = window_ret - rf_per
        std  = exc.std(ddof=1)
        if std < 1e-10:
            result[i] = float(np.sign(exc.mean()) * 1e6)
        else:
            result[i] = float(np.sqrt(ann_factor) * exc.mean() / std)

    return result


# ════════════════════════════════════════════════════════════════
#  DRAWDOWN ANALYSIS
# ════════════════════════════════════════════════════════════════

@dataclass
class DrawdownStats:
    """
    Full drawdown analysis beyond the scalar maximum drawdown.

    Attributes
    ----------
    max_dd          : worst peak-to-trough decline (negative).
    max_dd_duration : length of the worst drawdown in periods.
    max_dd_recovery : periods from trough back to new equity high
                      (None if not yet recovered at end of series).
    avg_dd          : mean of all drawdown periods.
    underwater      : full underwater equity series (0 when at peak).
    """
    max_dd          : float
    max_dd_duration : int
    max_dd_recovery : Optional[int]
    avg_dd          : float
    underwater      : np.ndarray


def drawdown_series(cum: np.ndarray) -> DrawdownStats:
    """
    Compute comprehensive drawdown statistics from an equity curve.

    Parameters
    ----------
    cum : cumulative equity array (e.g. cumprod(1 + returns)).

    Returns
    -------
    DrawdownStats with full underwater curve and duration metrics.
    """
    cum  = np.asarray(cum, dtype=np.float64)
    cum  = np.maximum(cum, 1e-12)   # guard against zero/negative equity
    peak = np.maximum.accumulate(cum)
    uw   = (cum - peak) / peak      # underwater equity (≤ 0)

    max_dd = float(uw.min())
    avg_dd = float(uw[uw < 0].mean()) if (uw < 0).any() else 0.0

    # ── Find duration and recovery of worst drawdown ──────────────────────
    trough_idx = int(np.argmin(uw))

    # Duration: walk back from trough to find the peak that preceded it
    peak_idx = trough_idx
    while peak_idx > 0 and cum[peak_idx - 1] >= cum[peak_idx]:
        peak_idx -= 1
    # More precise: find the last time cum was at the peak level
    peak_val = peak[trough_idx]
    peak_candidates = np.where(cum[:trough_idx + 1] >= peak_val - 1e-10)[0]
    peak_idx = int(peak_candidates[-1]) if len(peak_candidates) > 0 else 0
    duration = trough_idx - peak_idx

    # Recovery: first bar after trough where cum reaches the prior peak
    recovery: Optional[int] = None
    for i in range(trough_idx + 1, len(cum)):
        if cum[i] >= peak_val - 1e-10:
            recovery = i - trough_idx
            break

    return DrawdownStats(
        max_dd          = max_dd,
        max_dd_duration = duration,
        max_dd_recovery = recovery,
        avg_dd          = avg_dd,
        underwater      = uw,
    )


# ════════════════════════════════════════════════════════════════
#  TRADE STATISTICS
# ════════════════════════════════════════════════════════════════

def turnover_rate(signal: np.ndarray) -> float:
    """
    Average daily portfolio turnover as a fraction of portfolio value.

    Turnover = mean |signal[t] - signal[t-1]| over all bars.

    High turnover means the model trades frequently — even small costs
    compound into significant drag. A strategy with 50% daily turnover
    and 7bps round-trip cost loses ~8.8% per year in friction alone.

    Returns
    -------
    float  mean daily turnover (0 = never trades, 1 = full portfolio
           traded every day).
    """
    sig = np.asarray(signal, dtype=np.float64)
    if len(sig) < 2:
        return 0.0
    return float(np.abs(np.diff(sig)).mean())


def avg_holding_period(signal: np.ndarray) -> float:
    """
    Mean holding period in bars per trade.

    Computed as: total_bars_in_position / number_of_trades.
    A holding period of 1 means the model never holds overnight.
    A holding period of 5 means the average trade spans one week.

    Returns
    -------
    float  average bars per trade. Returns 0.0 if no trades taken.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if len(sig) == 0:
        return 0.0

    in_position = (sig != 0.0)
    n_bars      = int(in_position.sum())
    # A trade starts every time signal transitions from 0 to non-zero
    entries = int(((sig != 0) & (np.concatenate([[0.0], sig[:-1]]) == 0)).sum())

    return float(n_bars / entries) if entries > 0 else 0.0


# ════════════════════════════════════════════════════════════════
#  BENCHMARK COMPARISON
# ════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkMetrics:
    """
    Strategy performance relative to a benchmark (buy & hold).

    Attributes
    ----------
    benchmark_total_ret   : buy & hold total return over the period.
    benchmark_ann_ret     : buy & hold annualised return.
    benchmark_sharpe      : buy & hold Sharpe ratio.
    alpha                 : strategy annualised return − benchmark ann return.
    beta                  : strategy return sensitivity to benchmark.
    information_ratio     : (strategy ann ret − benchmark ann ret) /
                            tracking error (std of daily differences).
    excess_return_vs_bh   : strategy total return − buy & hold total return.
    """
    benchmark_total_ret : float
    benchmark_ann_ret   : float
    benchmark_sharpe    : float
    alpha               : float
    beta                : float
    information_ratio   : float
    excess_return_vs_bh : float


def benchmark_metrics(strategy_ret: np.ndarray,
                       ohlcv: pd.DataFrame,
                       risk_free: float = BACKTEST.risk_free_annual) -> BenchmarkMetrics:
    """
    Compute strategy metrics relative to buy & hold benchmark.

    Parameters
    ----------
    strategy_ret : daily net strategy returns.
    ohlcv        : OHLCV dataframe used in the backtest.
    risk_free    : annual risk-free rate.

    Returns
    -------
    BenchmarkMetrics dataclass.
    """
    n = len(strategy_ret)
    close_vals = ohlcv['Close'].values
    if len(close_vals) < 2:
        return BenchmarkMetrics(0, 0, 0, 0, 1, 0, 0)

    # Buy & hold: invest at Close[0], exit at Close[-1]
    bh_ret_total = float(close_vals[-1] / close_vals[0] - 1.0)
    n_cal        = max(len(ohlcv), 1)
    bh_cum       = np.maximum(close_vals / close_vals[0], 1e-12)
    bh_ann       = float(bh_cum[-1] ** (252.0 / n_cal) - 1.0)

    # Daily buy & hold returns aligned to strategy_ret length
    bh_daily = (close_vals[1:n + 1] - close_vals[:n]) \
               / (close_vals[:n] + 1e-9)
    min_n    = min(len(strategy_ret), len(bh_daily))
    s_ret    = strategy_ret[:min_n]
    b_ret    = bh_daily[:min_n]

    bh_sharpe = sharpe_ratio(b_ret, risk_free_rate=risk_free)

    # Beta = cov(strategy, benchmark) / var(benchmark)
    bh_var = float(np.var(b_ret, ddof=1))
    if bh_var > 1e-12:
        beta = float(np.cov(s_ret, b_ret, ddof=1)[0, 1] / bh_var)
    else:
        beta = 1.0

    # Alpha = strategy ann ret - beta * benchmark ann ret
    s_cum    = np.maximum(np.cumprod(1.0 + s_ret), 1e-12)
    s_ann    = float(s_cum[-1] ** (252.0 / n_cal) - 1.0)
    alpha    = s_ann - beta * bh_ann

    # Information ratio = mean(active ret) / std(active ret) * sqrt(252)
    active   = s_ret - b_ret
    ir_denom = float(active.std(ddof=1))
    ir       = float(np.sqrt(252) * active.mean() / ir_denom) \
               if ir_denom > 1e-10 else 0.0

    bh_total_strat_window = float(
        np.cumprod(1.0 + b_ret)[-1] - 1.0
    ) if len(b_ret) > 0 else 0.0
    excess = float(s_cum[-1] - 1.0) - bh_total_strat_window

    return BenchmarkMetrics(
        benchmark_total_ret = bh_ret_total,
        benchmark_ann_ret   = bh_ann,
        benchmark_sharpe    = bh_sharpe,
        alpha               = alpha,
        beta                = beta,
        information_ratio   = ir,
        excess_return_vs_bh = excess,
    )


# ════════════════════════════════════════════════════════════════
#  KELLY POSITION SIZING
# ════════════════════════════════════════════════════════════════

def kelly_position(p: np.ndarray, ohlcv: pd.DataFrame,
                   signal_history: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fractional Kelly: f* = (p*b - q) / b, b = rolling avg_win/avg_loss.

    V8 FIX — C4:
    The previous implementation computed the win/loss ratio from raw
    AAPL price returns (np.diff(close) / close[:-1]).  This is wrong
    in two ways:

      1. DIRECTION MISMATCH: The strategy takes short positions on DOWN
         predictions.  A raw return of -1% is a WIN for a short trade
         but is counted as a LOSS by the old code.

      2. MAGNITUDE MISMATCH: The raw return is the price move, not the
         strategy's P&L.  A Kelly fraction calibrated on raw returns
         overestimates edge when the model is mostly right and
         underestimates risk when the model is mostly wrong.

    FIX:
      - When signal_history is provided (prior signal array aligned to
        ohlcv), compute trade-level P&L = signal * exec_ret and use
        that series for the win/loss rolling estimate.
      - When signal_history is None (cold start, first bar), fall back
        to raw returns with a direction-neutral estimate (b=1.0),
        equivalent to a 50/50 prior.  This is conservative and correct
        for a cold-start situation.

    Time-alignment: ohlcv.iloc[:n+1] uses only historical prices — no
    future information leaks into position sizing.
    """
    opens = (ohlcv['Open'].values if 'Open' in ohlcv.columns
             else ohlcv['Close'].values)
    n     = min(len(p), len(opens) - 1)

    if signal_history is not None and len(signal_history) >= n:
        # Use actual strategy trade returns for Kelly calibration
        sig_hist   = signal_history[:n]
        exec_ret   = (opens[1:n + 1] - opens[:n]) / (opens[:n] + 1e-9)
        trade_pnl  = pd.Series(sig_hist * exec_ret)
        win_roll   = trade_pnl.clip(lower=0).rolling(60, min_periods=20).mean()
        loss_roll  = trade_pnl.clip(upper=0).abs().rolling(60, min_periods=20).mean()
    else:
        # Cold start: use conservative neutral prior
        # b=1.0 means Kelly reduces to f* = 2p - 1, which is 0 at p=0.5
        avg_win  = np.full(n, 0.006)
        avg_loss = np.full(n, 0.006)
        b        = np.ones(n)
        pi       = p[:n]
        f        = (pi * b - (1.0 - pi)) / (b + 1e-9)
        return np.clip(f * BACKTEST.kelly_fraction, 0.0, BACKTEST.max_position)

    avg_win  = win_roll.fillna(0.006).values[:n]
    avg_loss = loss_roll.fillna(0.006).values[:n]
    b        = avg_win / (avg_loss + 1e-9)
    b        = np.where(b < 0.1, 1.0, b)
    pi       = p[:n]
    f        = (pi * b - (1.0 - pi)) / (b + 1e-9)
    return np.clip(f * BACKTEST.kelly_fraction, 0.0, BACKTEST.max_position)


# ════════════════════════════════════════════════════════════════
#  BACKTEST VALIDATION & COST MODEL
# ════════════════════════════════════════════════════════════════

def _validate_backtest_inputs(preds: np.ndarray,
                               y_prob: np.ndarray,
                               ohlcv: pd.DataFrame) -> None:
    if len(preds) == 0:
        raise ValueError("trading_metrics: preds is empty.")
    if len(preds) != len(y_prob):
        raise ValueError(
            f"trading_metrics: preds len {len(preds)} != y_prob len {len(y_prob)}.")
    missing = {'Open', 'Close'} - set(ohlcv.columns)
    if missing:
        raise ValueError(f"trading_metrics: ohlcv missing columns {missing}.")
    if len(ohlcv) < 2:
        raise ValueError("trading_metrics: ohlcv needs ≥ 2 rows.")
    invalid = np.unique(preds[~np.isin(preds, [0, 1])])
    if len(invalid) > 0:
        raise ValueError(
            f"trading_metrics: preds contains non-binary values: {invalid}.")


def _compute_trade_costs(
    signal: np.ndarray,
    commission_bps: float,
    slippage_bps: float,
    impact_coeff: float   = 0.1,
    adv_shares: float     = 1e6,
    trade_size_shares: float = 1000,
) -> np.ndarray:
    """
    Per-bar trading costs: commission + slippage + market impact.

    Formulas
    --------
    commission  : fixed broker fee, one-way
                  C = commission_bps / 10_000

    slippage    : half-spread + timing slippage, one-way
                  S = slippage_bps / 10_000

    market impact (square-root law, Almgren et al. 2005):
                  participation_rate = trade_size_shares / adv_shares
                  I = impact_coeff * sqrt(participation_rate)   [as a fraction]
                  → zero when adv_shares=0 or trade_size_shares=0

    total one-way rate:
                  rate = C + S + I

    cost per bar:
                  cost[t] = |Δsignal[t]| * rate
    Reversals (long→short) pay the rate twice — once to close, once to open.
    The extra leg is applied to the outgoing position size |signal[t-1]|.
    """
    # participation rate → market impact in bps
    if adv_shares > 0 and trade_size_shares > 0:
        participation_rate = trade_size_shares / adv_shares
        impact_bps = impact_coeff * np.sqrt(participation_rate) * 10_000
    else:
        impact_bps = 0.0

    rate = (commission_bps + slippage_bps + impact_bps) / 10_000

    prev          = np.concatenate([[0.0], signal[:-1]])
    delta         = np.abs(signal - prev)
    reversal_mask = ((np.sign(signal) != np.sign(prev)) &
                     (np.sign(signal) != 0) &
                     (np.sign(prev) != 0))

    costs = delta * rate
    # Extra close leg on reversals
    costs[reversal_mask] += np.abs(prev[reversal_mask]) * rate
    return costs


# ════════════════════════════════════════════════════════════════
#  BACKTESTING ENGINE
# ════════════════════════════════════════════════════════════════

def trading_metrics(preds: np.ndarray,
                    y_prob: np.ndarray,
                    ohlcv: pd.DataFrame,
                    commission_bps: float    = BACKTEST.commission_bps,
                    slippage_bps: float      = BACKTEST.slippage_bps,
                    impact_coeff: float      = BACKTEST.impact_coeff,
                    adv_shares: float        = BACKTEST.adv_shares,
                    trade_size_shares: float = BACKTEST.trade_size_shares,
                    risk_free: float         = BACKTEST.risk_free_annual,
                    long_only: bool          = False,
                    use_kelly: bool          = BACKTEST.use_kelly,
                    confidence_filter: float = BACKTEST.confidence_threshold,
                    stop_loss_pct: float     = BACKTEST.stop_loss_pct,
                    ) -> Dict:
    """
    Realistic backtest: next-open execution, confidence filter,
    Kelly sizing, commission + slippage, stop-loss, comprehensive metrics.

    Execution model (time-correct, no look-ahead)
    ──────────────────────────────────────────────
      Close[t]  →  pred[t] generated  →  signal[t] decided
      →  execute at Open[t+1]
      exec_ret[t] = (Open[t+1] − Open[t]) / Open[t]

    V8 FIX — C2: Stop-loss enforcement
    ────────────────────────────────────
    The previous implementation defined stop_loss_pct = 0.05 in
    BacktestConfig but NEVER referenced it in this function.  The
    reported Sharpe, drawdown, and win-rate were therefore for a strategy
    with NO risk control — materially different from what the config
    implied and what would happen in real trading.

    Implementation:
      - A per-trade entry price is tracked (Open[t+1] when signal goes
        non-zero or reverses).
      - On each bar, if the unrealised loss from entry exceeds
        stop_loss_pct, the signal is forced to 0 for that bar and the
        trade is closed at Open[t+1] of the next bar.
      - The stop fires BEFORE signal direction logic so it takes priority
        over any new prediction.
      - stop_loss_pct = 0.0 disables the stop entirely (backward compat).

    Returns
    -------
    dict with all scalar metrics plus:
      cum          : EquityCurve (equity curve + index)
      daily_strat  : np.ndarray of per-period net returns
      drawdown     : DrawdownStats (full drawdown analysis)
      rolling_sh   : np.ndarray rolling 63-day Sharpe
      benchmark    : BenchmarkMetrics (vs buy & hold)
      n_stopped    : int — number of trades closed by stop-loss
    """
    _validate_backtest_inputs(preds, y_prob, ohlcv)

    opens = (ohlcv['Open'].values if 'Open' in ohlcv.columns
             else ohlcv['Close'].values)
    n     = min(len(preds), len(opens) - 1)

    # ── Signal generation ─────────────────────────────────────────────────
    confident = (np.abs(y_prob[:n] - 0.5) >= confidence_filter)

    if use_kelly:
        pos_size = kelly_position(y_prob[:n], ohlcv.iloc[:n + 1])
        pos_size = np.where(confident, pos_size, 0.0)
        raw_signal = (np.where(preds[:n] == 1, pos_size, 0.0) if long_only
                      else np.where(preds[:n] == 1, 1.0, -1.0) * pos_size)
    else:
        if long_only:
            raw_signal = np.where((preds[:n] == 1) & confident, 1.0, 0.0)
        else:
            raw_signal = np.where(confident,
                                  np.where(preds[:n] == 1, 1.0, -1.0), 0.0)

    # ── V8: Stop-loss enforcement ─────────────────────────────────────────
    # Walk through bar by bar.  Track the entry price of the current trade.
    # If the unrealised loss from entry exceeds stop_loss_pct, force
    # signal=0 (close position) and record a stop event.
    #
    # Why bar-by-bar instead of vectorised?
    # The stop depends on the running entry price, which itself depends on
    # prior stop events — each bar's outcome affects the next bar's state.
    # A vectorised implementation would require ignoring this dependency,
    # which would understate the number of stops (and overstate performance).
    signal    = raw_signal.copy()
    n_stopped = 0

    if stop_loss_pct > 0.0:
        entry_price   = 0.0
        in_trade      = False
        stop_active   = False   # True for the one bar after a stop fires

        for t in range(n):
            if stop_active:
                # Previous bar triggered stop — we closed at Open[t].
                # Reset state; evaluate new signal normally.
                in_trade    = False
                entry_price = 0.0
                stop_active = False

            if not in_trade:
                if signal[t] != 0.0:
                    # New trade entry: fill at Open[t+1]
                    entry_price = opens[t + 1] if t + 1 < len(opens) else opens[t]
                    in_trade    = True
            else:
                # Check if direction changed (model reversed) → new entry
                prev_sign = np.sign(signal[t - 1]) if t > 0 else 0.0
                curr_sign = np.sign(signal[t])
                if curr_sign == 0.0:
                    in_trade    = False
                    entry_price = 0.0
                elif curr_sign != prev_sign:
                    # Reversal: update entry price
                    entry_price = opens[t + 1] if t + 1 < len(opens) else opens[t]
                else:
                    # Continuing trade: check stop
                    current_price = opens[t + 1] if t + 1 < len(opens) else opens[t]
                    if entry_price > 1e-9:
                        direction      = np.sign(signal[t])
                        unrealised_pnl = direction * (current_price - entry_price) / entry_price
                        if unrealised_pnl < -stop_loss_pct:
                            # Stop triggered: close position this bar
                            signal[t]   = 0.0
                            n_stopped  += 1
                            stop_active = True
                            in_trade    = False
                            entry_price = 0.0

    # ── Returns and costs ─────────────────────────────────────────────────
    exec_ret   = (opens[1:n + 1] - opens[:n]) / (opens[:n] + 1e-9)
    costs      = _compute_trade_costs(
        signal, commission_bps, slippage_bps,
        impact_coeff=impact_coeff,
        adv_shares=adv_shares,
        trade_size_shares=trade_size_shares,
    )
    net_ret    = signal * exec_ret - costs
    cum_arr    = np.maximum(np.cumprod(1.0 + net_ret), 1e-6)
    rf_per_day = risk_free / 252.0
    excess     = net_ret - rf_per_day

    # ── Attach DatetimeIndex if available ────────────────────────────────
    ohlcv_index = getattr(ohlcv, 'index', None)
    if (ohlcv_index is not None and
            isinstance(ohlcv_index, pd.DatetimeIndex) and
            len(ohlcv_index) >= n):
        ret_index = ohlcv_index[:n]
    else:
        ret_index = None

    equity_curve = EquityCurve(
        cum        = cum_arr,
        daily_ret  = net_ret,
        index      = ret_index,
    )

    # ── Scalar risk metrics ───────────────────────────────────────────────
    sharpe  = sharpe_ratio(net_ret, risk_free_rate=risk_free)
    omega   = omega_ratio(net_ret, threshold=rf_per_day)
    var_95, cvar_95 = var_cvar(net_ret, confidence=0.95)

    excess_downside = excess[excess < 0].std() + 1e-9
    sortino  = float(np.sqrt(252) * excess.mean() / excess_downside)

    dd_stats = drawdown_series(cum_arr)
    max_dd   = dd_stats.max_dd

    n_calendar = max(len(ohlcv), 1)
    ann_ret    = float(cum_arr[-1] ** (252.0 / n_calendar) - 1.0)
    calmar     = ann_ret / (abs(max_dd) + 1e-9)

    # ── Trade-level statistics ────────────────────────────────────────────
    trade_mask = signal != 0.0
    if trade_mask.any():
        trade_ret = net_ret[trade_mask]
        win_rate  = float((trade_ret > 0).mean())
        wins      = trade_ret[trade_ret > 0]
        losses    = trade_ret[trade_ret < 0]
    else:
        win_rate, wins, losses = 0.0, np.array([]), np.array([])

    gross_loss    = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = (float(wins.sum() / gross_loss)
                     if gross_loss > 1e-10
                     else (float('inf') if len(wins) > 0 else 1.0))

    n_trades   = int((np.diff(np.concatenate([[0.0], signal])) != 0).sum())
    turnover   = turnover_rate(signal)
    avg_hold   = avg_holding_period(signal)

    # ── Rolling Sharpe ────────────────────────────────────────────────────
    roll_sh = rolling_sharpe(net_ret, window=63,
                              risk_free_rate=risk_free)

    # ── Information Coefficient (aligned) ────────────────────────────────
    close_vals = ohlcv['Close'].values
    if len(close_vals) >= n + 1:
        close_ret_aligned = (close_vals[1:n + 1] - close_vals[:n]) \
                             / (close_vals[:n] + 1e-9)
        ic = information_coefficient(y_prob[:n], close_ret_aligned)
    else:
        close_ret = ohlcv['Close'].pct_change().dropna().values
        ic = information_coefficient(y_prob[:n], close_ret[:n])

    # ── Benchmark comparison ──────────────────────────────────────────────
    bm = benchmark_metrics(net_ret, ohlcv, risk_free=risk_free)

    return dict(
        # Core risk-adjusted metrics
        sharpe        = sharpe,
        sortino       = sortino,
        calmar        = calmar,
        omega         = omega,
        # Drawdown
        max_dd        = max_dd,
        max_dd_duration = dd_stats.max_dd_duration,
        max_dd_recovery = dd_stats.max_dd_recovery,
        avg_dd        = dd_stats.avg_dd,
        # Returns
        total_ret     = float(cum_arr[-1] - 1.0),
        ann_ret       = ann_ret,
        # Tail risk
        var_95        = var_95,
        cvar_95       = cvar_95,
        # Trade stats
        win_rate      = win_rate,
        profit_factor = profit_factor,
        n_trades      = n_trades,
        n_stopped     = n_stopped,          # V8 NEW: stop-loss fires
        turnover      = turnover,
        avg_hold      = avg_hold,
        # Signal quality
        pct_confident = float(confident.mean()),
        ic            = ic,
        # Benchmark
        alpha         = bm.alpha,
        beta          = bm.beta,
        information_ratio = bm.information_ratio,
        excess_vs_bh  = bm.excess_return_vs_bh,
        bh_ann_ret    = bm.benchmark_ann_ret,
        bh_sharpe     = bm.benchmark_sharpe,
        # cum = raw numpy array for backward compatibility with visualisation.py
        cum           = cum_arr,
        equity_curve  = equity_curve,
        daily_strat   = net_ret,
        drawdown      = dd_stats,
        rolling_sh    = roll_sh,
        benchmark     = bm,
    )


# ════════════════════════════════════════════════════════════════
#  REPORTING
# ════════════════════════════════════════════════════════════════

# Metrics to include in tabular comparison (scalar values only).
_REPORT_SCALARS = [
    'sharpe', 'sortino', 'calmar', 'omega',
    'max_dd', 'max_dd_duration', 'avg_dd',
    'total_ret', 'ann_ret', 'var_95', 'cvar_95',
    'win_rate', 'profit_factor', 'n_trades', 'n_stopped',   # V8: n_stopped added
    'turnover', 'avg_hold',
    'pct_confident', 'ic',
    'alpha', 'beta', 'information_ratio', 'excess_vs_bh',
    'bh_ann_ret', 'bh_sharpe',
]




# ════════════════════════════════════════════════════════════════
#  STATISTICAL SIGNIFICANCE  (V8 new — H2 fix)
# ════════════════════════════════════════════════════════════════

def sharpe_bootstrap_ci(
    portfolio_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    frequency: str = "daily",
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for the annualised Sharpe Ratio.

    WHY THIS MATTERS (H2):
    A Sharpe of 1.0 computed on 252 daily returns has a 95% CI of
    roughly [-0.3, +2.3] — statistically indistinguishable from zero.
    Reporting a point estimate without a CI gives a false sense of
    precision and is the primary way that overfitted strategies appear
    to have edge when they do not.

    Method: stationary bootstrap (Politis & Romano 1994) approximated
    by block bootstrap with mean block size proportional to the
    autocorrelation structure of the returns.  For i.i.d. returns,
    block_size=1 and this reduces to standard bootstrap.

    Parameters
    ----------
    n_bootstrap : number of bootstrap resamples (2000 is sufficient
                  for 95% CI; use 5000 for 99% CI).
    ci_level    : confidence level (default 0.95 for 95% CI).

    Returns
    -------
    (lower, upper) : bootstrap percentile CI bounds.
    """
    ret = np.asarray(portfolio_returns, dtype=np.float64).ravel()
    ret = ret[np.isfinite(ret)]
    if len(ret) < 10:
        return (float('-inf'), float('inf'))

    ann_factor = _ANN.get(frequency, 252.0)
    rf_per     = risk_free_rate / ann_factor

    # Estimate mean block size from lag-1 autocorrelation
    # (Politis & Romano heuristic: block_size ≈ 1 / (1 - |ρ_1|))
    if len(ret) > 5:
        rho1 = float(np.corrcoef(ret[:-1], ret[1:])[0, 1])
        rho1 = np.clip(rho1, -0.99, 0.99)
        block_size = max(1, int(round(1.0 / (1.0 - abs(rho1)))))
        block_size = min(block_size, len(ret) // 4)   # cap at 25% of series
    else:
        block_size = 1

    rng        = np.random.default_rng(seed)
    sharpes_bs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Block bootstrap: sample starting indices, then stitch blocks
        n_blocks   = int(np.ceil(len(ret) / block_size))
        starts     = rng.integers(0, len(ret), size=n_blocks)
        boot_idx   = np.concatenate(
            [np.arange(s, min(s + block_size, len(ret))) for s in starts]
        )[:len(ret)]
        boot_ret   = ret[boot_idx]
        exc        = boot_ret - rf_per
        std_e      = exc.std(ddof=1)
        sharpes_bs[i] = (float(np.sqrt(ann_factor) * exc.mean() / std_e)
                         if std_e > 1e-10 else 0.0)

    alpha = 1.0 - ci_level
    lower = float(np.percentile(sharpes_bs, 100 * alpha / 2))
    upper = float(np.percentile(sharpes_bs, 100 * (1 - alpha / 2)))
    return lower, upper


def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_observations: int,
    n_trials: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    frequency: str = "daily",
) -> Tuple[float, float]:
    """
    Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2016).

    When a strategy is selected from N_trials candidates the expected maximum
    Sharpe under the null hypothesis of zero edge is NOT zero:
      E[max SR*] ≈ (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))
    where γ is the Euler–Mascheroni constant and the result is in annualised
    SR units (= Z-score units for daily data with T=252).

    The DSR is P(SR_obs > SR_benchmark | null), adjusted for:
      1. Number of trials (multiple-testing penalty)
      2. Non-normality of returns (skewness, excess kurtosis)
      3. Estimation uncertainty from finite sample size

    Parameters
    ----------
    sharpe_observed : annualised Sharpe ratio from trading_metrics().
    n_observations  : number of return bars in the test period (T).
    n_trials        : effective number of independent trials (model versions,
                      HPO iterations, walk-forward folds combined).
    skewness        : return skewness (0 = normal).
    excess_kurtosis : excess kurtosis = scipy kurtosis(fisher=True),
                      i.e. Pearson kurtosis - 3.  0 = normal distribution.
    frequency       : 'daily' | 'weekly' | 'monthly' for annualisation factor.

    Returns
    -------
    (dsr, sr_benchmark) :
        dsr          — P(genuine edge) in [0,1].  DSR > 0.95 is meaningful.
        sr_benchmark — annualised SR threshold that must be exceeded.
    """
    from scipy.stats import norm

    ann_factor = _ANN.get(frequency, 252.0)
    T = max(n_observations, 2)

    # Per-period SR — used only in the variance formula (BLP Eq. 5)
    sr_per = sharpe_observed / np.sqrt(ann_factor)

    # ------------------------------------------------------------------ #
    # Expected max SR benchmark (BLP 2016, Eq. 8)                         #
    # Result is in annualised SR units (Z-score scale).                   #
    # Minimum n_trials=2 to avoid norm.ppf(0) = -inf.                     #
    # ------------------------------------------------------------------ #
    gamma_em = 0.5772156649
    n_t = max(n_trials, 2)
    sr_star_ann = (
        (1.0 - gamma_em) * norm.ppf(1.0 - 1.0 / n_t)
        + gamma_em       * norm.ppf(1.0 - 1.0 / (n_t * np.e))
    )

    # ------------------------------------------------------------------ #
    # Std of annualised SR estimator under non-normality (BLP Eq. 5)      #
    # Var[SR_per] = (1 - skew·SR_per + (excess_kurt/4)·SR_per²) / (T-1)  #
    # Var[SR_ann] = Var[SR_per] · ann_factor  (SR_ann = SR_per·√ann)      #
    # ------------------------------------------------------------------ #
    sr_var_per = (
        1.0
        - skewness * sr_per
        + (excess_kurtosis / 4.0) * sr_per ** 2
    ) / max(T - 1, 1)

    sr_std_ann = np.sqrt(max(sr_var_per * ann_factor, 1e-12))

    # DSR: probability observed SR exceeds the null benchmark
    dsr = float(norm.cdf((sharpe_observed - sr_star_ann) / sr_std_ann))

    return dsr, float(sr_star_ann)

def compute_statistical_summary(
    portfolio_returns: np.ndarray,
    sharpe: float,
    n_trials: int = 9,
    risk_free_rate: float = 0.0,
    frequency: str = "daily",
    n_bootstrap: int = 2000,
) -> Dict:
    """
    Full statistical validity summary for a strategy.

    Combines bootstrap CI and DSR into a single dict that should be
    reported alongside every set of backtest metrics.

    Parameters
    ----------
    n_trials : effective number of model versions / configurations
               evaluated against this test set.  For this system,
               V1–V9 = 9 versions minimum; with HPO = 30–50 effective.

    Returns
    -------
    Dict with:
        sharpe_ci_lower, sharpe_ci_upper : 95% bootstrap CI
        sharpe_ci_width                  : CI width (smaller = more reliable)
        deflated_sharpe_ratio            : DSR (0–1 probability of genuine edge)
        sr_benchmark                     : null-hypothesis SR threshold
        is_significant                   : DSR > 0.95
        n_obs                            : test observations
        n_trials_assumed                 : n_trials used in DSR
        verdict                          : human-readable summary
    """
    ret = np.asarray(portfolio_returns, dtype=np.float64)
    ret = ret[np.isfinite(ret)]
    n_obs = len(ret)

    # Return skewness and kurtosis for DSR non-normality adjustment
    if n_obs >= 4:
        from scipy.stats import skew as sp_skew, kurtosis as sp_kurt
        skewness = float(sp_skew(ret))
        excess_kurtosis = float(sp_kurt(ret, fisher=True))   # FIX: excess kurtosis (0=normal)
    else:
        skewness, excess_kurtosis = 0.0, 0.0   # FIX: excess kurtosis default

    ci_lower, ci_upper = sharpe_bootstrap_ci(
        ret, risk_free_rate=risk_free_rate,
        frequency=frequency, n_bootstrap=n_bootstrap,
    )

    dsr, sr_benchmark = deflated_sharpe_ratio(
        sharpe_observed=sharpe,
        n_observations=n_obs,
        n_trials=n_trials,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,   # FIX: now passing excess kurtosis
        frequency=frequency,
    )

    is_significant = dsr > 0.95

    if dsr < 0.50:
        verdict = "FAIL — Sharpe within null distribution of random strategies"
    elif dsr < 0.80:
        verdict = "WEAK — Some edge possible but not statistically convincing"
    elif dsr < 0.95:
        verdict = "MODERATE — Edge likely but DSR < 0.95 threshold"
    else:
        verdict = "SIGNIFICANT — DSR > 0.95, edge likely genuine"

    return dict(
        sharpe_ci_lower    = round(ci_lower, 3),
        sharpe_ci_upper    = round(ci_upper, 3),
        sharpe_ci_width    = round(ci_upper - ci_lower, 3),
        deflated_sharpe_ratio = round(dsr, 4),
        sr_benchmark       = round(sr_benchmark, 3),
        is_significant     = is_significant,
        n_obs              = n_obs,
        n_trials_assumed   = n_trials,
        skewness           = round(skewness, 3),
        excess_kurtosis    = round(excess_kurtosis, 3),
        verdict            = verdict,
    )


def generate_report(model_name: str,
                    clf_metrics: Dict,
                    trd_metrics: Dict,
                    n_trials: int = 9,
                    ) -> Dict:
    """
    Assemble a complete, structured performance report for one model.

    V8: Automatically includes statistical validity summary
    (bootstrap Sharpe CI + Deflated Sharpe Ratio) so every report
    is self-contained and statistically auditable.

    Parameters
    ----------
    n_trials : effective number of model versions evaluated against
               the test set.  Default 9 matches V1–V9 history.
               Set higher if HPO iterations were run.
    """
    trading_scalars = {
        k: (round(float(v), 6) if isinstance(v, (int, float, np.floating))
            and np.isfinite(v)
            else (str(v) if not isinstance(v, (np.ndarray, pd.DataFrame,
                                                EquityCurve, DrawdownStats,
                                                BenchmarkMetrics))
                  else None))
        for k, v in trd_metrics.items()
        if k in _REPORT_SCALARS
    }

    clf_scalars = {
        k: (round(float(v), 6) if isinstance(v, (int, float, np.floating))
            else None)
        for k, v in clf_metrics.items()
        if k not in ('cm', 'report')
    }

    # V8 NEW: statistical significance summary (H2 fix)
    daily_ret = trd_metrics.get('daily_strat', np.array([]))
    sharpe    = trd_metrics.get('sharpe', 0.0)
    stats     = compute_statistical_summary(
        portfolio_returns=daily_ret,
        sharpe=sharpe,
        n_trials=n_trials,
    )

    return dict(
        model          = model_name,
        classification = clf_scalars,
        trading        = trading_scalars,
        statistics     = stats,            # V8 NEW: DSR + bootstrap CI
        summary_line   = (
            f"Sharpe={trd_metrics.get('sharpe', 0):+.2f}  "
            f"Sortino={trd_metrics.get('sortino', 0):+.2f}  "
            f"MaxDD={trd_metrics.get('max_dd', 0)*100:.1f}%  "
            f"AnnRet={trd_metrics.get('ann_ret', 0)*100:+.1f}%  "
            f"Alpha={trd_metrics.get('alpha', 0)*100:+.1f}%  "
            f"IC={trd_metrics.get('ic', 0):+.4f}  "
            f"Trades={trd_metrics.get('n_trades', 0)}  "
            f"Stopped={trd_metrics.get('n_stopped', 0)}  "    # V8 stop-loss count
            f"DSR={stats['deflated_sharpe_ratio']:.3f}  "     # V8 DSR
            f"CI=[{stats['sharpe_ci_lower']:+.2f},{stats['sharpe_ci_upper']:+.2f}]"
        ),
    )


def compare_models(reports: Dict[str, Dict],
                   sort_by: str = 'sharpe') -> pd.DataFrame:
    """
    Produce a ranked comparison table from a collection of model reports.

    Parameters
    ----------
    reports : {model_name: generate_report() output}.
    sort_by : metric column to sort by (descending).

    Returns
    -------
    pd.DataFrame with one row per model, columns = scalar metrics,
    sorted by sort_by descending. Ready for display or CSV export.

    Example
    -------
        table = compare_models(
            {name: generate_report(name, clf, trd)
             for name, (clf, trd) in results.items()}
        )
        print(table[['sharpe', 'sortino', 'max_dd', 'ann_ret', 'alpha']])
    """
    rows = []
    for name, report in reports.items():
        row = {'model': name}
        row.update(report.get('classification', {}))
        row.update(report.get('trading', {}))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index('model')

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    return df


# ════════════════════════════════════════════════════════════════
#  PRINT  (backward-compatible)
# ════════════════════════════════════════════════════════════════

def print_metrics(name: str, clf: Dict, trd: Dict) -> None:
    """
    Log a human-readable summary of classification + trading metrics.

    V8: includes new metrics (omega, VaR, alpha, information ratio)
    and handles the EquityCurve object in trd['cum'] gracefully.
    """
    pf     = trd.get('profit_factor', 0)
    pf_str = f"{pf:.2f}" if np.isfinite(pf) else "inf"

    omega = trd.get('omega', float('nan'))
    omega_str = f"{omega:.2f}" if np.isfinite(omega) else "inf"

    logger.info(
        f"\n  ── {name} ──\n"
        f"  Acc={clf['accuracy']*100:.2f}%  AUC={clf['auc']:.4f}  "
        f"F1up={clf['f1_up']:.4f}  F1dn={clf['f1_down']:.4f}  "
        f"Brier={clf['brier']:.4f}\n"
        f"  Sharpe={trd['sharpe']:+.2f}  Sortino={trd['sortino']:+.2f}  "
        f"Calmar={trd['calmar']:+.2f}  Omega={omega_str}  "
        f"IC={trd['ic']:+.4f}\n"
        f"  MaxDD={trd['max_dd']*100:.1f}%  "
        f"MaxDDDur={trd.get('max_dd_duration', 0)}d  "
        f"VaR95={trd.get('var_95', 0)*100:.2f}%  "
        f"CVaR95={trd.get('cvar_95', 0)*100:.2f}%\n"
        f"  AnnRet={trd['ann_ret']*100:+.1f}%  "
        f"Alpha={trd.get('alpha', 0)*100:+.1f}%  "
        f"Beta={trd.get('beta', 1):.2f}  "
        f"IR={trd.get('information_ratio', 0):+.2f}\n"
        f"  WinRate={trd['win_rate']*100:.1f}%  PF={pf_str}  "
        f"Trades={trd['n_trades']}  "
        f"Turnover={trd.get('turnover', 0)*100:.1f}%/d  "
        f"AvgHold={trd.get('avg_hold', 0):.1f}bars  "
        f"Confident={trd['pct_confident']*100:.0f}%"
    )
    logger.info(clf['report'])


# ════════════════════════════════════════════════════════════════
#  COST STRESS TEST
# ════════════════════════════════════════════════════════════════

def cost_stress_test(
    preds: np.ndarray,
    y_prob: np.ndarray,
    ohlcv: pd.DataFrame,
    cost_scenarios_bps: list = None,
    risk_free: float = BACKTEST.risk_free_annual,
    adv_shares: float = BACKTEST.adv_shares,
    trade_size_shares: float = BACKTEST.trade_size_shares,
    impact_coeff: float = BACKTEST.impact_coeff,
) -> pd.DataFrame:
    """
    Stress-test strategy performance across a range of total one-way
    cost assumptions (commission + slippage combined, before market impact).

    Each scenario runs the full backtest with that flat cost level plus
    the configured market impact on top.

    Cost allocation per scenario
    ----------------------------
    total_bps = commission_bps + slippage_bps  (impact is additive on top)

    We split the scenario bps 33/67 commission/slippage, which reflects
    typical retail broker economics. Market impact is applied uniformly
    across all scenarios using the configured adv_shares / trade_size_shares.

    Parameters
    ----------
    cost_scenarios_bps : list of total one-way cost levels to test.
                         Default: [1, 5, 10, 20, 50] bps.

    Returns
    -------
    pd.DataFrame with one row per scenario, columns:
        total_cost_bps, commission_bps, slippage_bps, impact_bps,
        sharpe, ann_ret_pct, max_dd_pct, win_rate_pct,
        n_trades, turnover_pct_day,
        sharpe_vs_baseline (delta vs 1 bps scenario),
        cost_breakeven_bps (estimated cost level where Sharpe = 0)
    """
    if cost_scenarios_bps is None:
        cost_scenarios_bps = [1, 5, 10, 20, 50]

    # Compute impact bps once (same for all scenarios)
    if adv_shares > 0 and trade_size_shares > 0:
        participation_rate = trade_size_shares / adv_shares
        impact_bps = impact_coeff * np.sqrt(participation_rate) * 10_000
    else:
        impact_bps = 0.0

    rows = []
    for scenario_bps in cost_scenarios_bps:
        # Split: 1/3 commission, 2/3 slippage (conventional split)
        comm   = round(scenario_bps / 3.0, 4)
        slip   = round(scenario_bps - comm, 4)

        try:
            m = trading_metrics(
                preds=preds,
                y_prob=y_prob,
                ohlcv=ohlcv,
                commission_bps=comm,
                slippage_bps=slip,
                impact_coeff=impact_coeff,
                adv_shares=adv_shares,
                trade_size_shares=trade_size_shares,
                risk_free=risk_free,
            )
            rows.append(dict(
                total_cost_bps   = scenario_bps,
                commission_bps   = round(comm, 2),
                slippage_bps     = round(slip, 2),
                impact_bps       = round(impact_bps, 2),
                effective_bps    = round(scenario_bps + impact_bps, 2),
                sharpe           = round(m['sharpe'], 3),
                ann_ret_pct      = round(m['ann_ret'] * 100, 2),
                max_dd_pct       = round(m['max_dd'] * 100, 2),
                win_rate_pct     = round(m['win_rate'] * 100, 1),
                n_trades         = m['n_trades'],
                turnover_pct_day = round(m.get('turnover', 0) * 100, 2),
            ))
        except Exception as e:
            rows.append(dict(total_cost_bps=scenario_bps, error=str(e)))

    df = pd.DataFrame(rows)

    if 'sharpe' in df.columns and not df['sharpe'].isna().all():
        baseline_sharpe = df.loc[df['total_cost_bps'] == df['total_cost_bps'].min(),
                                 'sharpe'].iloc[0]
        df['sharpe_vs_cheapest'] = (df['sharpe'] - baseline_sharpe).round(3)

        # Linear interpolation to estimate breakeven cost
        try:
            pos = df[df['sharpe'] >= 0]
            neg = df[df['sharpe'] < 0]
            if len(pos) > 0 and len(neg) > 0:
                c1, s1 = pos.iloc[-1]['total_cost_bps'], pos.iloc[-1]['sharpe']
                c2, s2 = neg.iloc[0]['total_cost_bps'],  neg.iloc[0]['sharpe']
                breakeven = c1 + (0 - s1) * (c2 - c1) / (s2 - s1)
                df['breakeven_bps'] = round(breakeven, 1)
            else:
                df['breakeven_bps'] = np.nan
        except Exception:
            df['breakeven_bps'] = np.nan

    return df
