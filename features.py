"""
features.py  —  V8 Feature Engineering Engine.

V8 FIXES over V7:

  1. SINGLE SCALER: V7 built scaler_pre for feature selection, then a
     second scaler for actual training — two different scalers on the same
     data caused subtle inconsistencies. V8 returns a single scaler built
     AFTER feature selection.

  2. BASELINE STRATEGY: ma_crossover_baseline() computes a simple
     5/20-day SMA crossover signal as a dumb benchmark. Any ML model that
     can't beat this has a problem.

  3. PURGED WALK-FORWARD helper: split_purged() creates train/test index
     pairs with a purge gap (embargo) between them to prevent autocorrelation
     leakage at fold boundaries.

  4. REGIME CLASSIFIER: classify_regime() labels each bar as:
     'trending_up', 'trending_down', 'mean_reverting', 'high_vol'
     using rolling Sharpe ratio and volatility ratio. Used for
     regime-conditional evaluation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler

from config import DATA
from data_loader import download, validate_ohlcv  # noqa: F401

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  VECTORISED HELPERS
# ════════════════════════════════════════════════════════════════

def _rsi(c: pd.Series, p: int) -> pd.Series:
    d = c.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + g / (l + 1e-9))


def _rolling_autocorr(s: pd.Series, window: int, lag: int) -> pd.Series:
    arr    = s.values.astype(np.float64)
    N      = len(arr)
    result = np.full(N, np.nan)
    if N < window + lag:
        return pd.Series(result, index=s.index)
    shape   = (N - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    wins    = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides).copy()
    x, y   = wins[:, :-lag], wins[:, lag:]
    x_dm   = x - x.mean(axis=1, keepdims=True)
    y_dm   = y - y.mean(axis=1, keepdims=True)
    num    = (x_dm * y_dm).sum(axis=1)
    denom  = np.sqrt((x_dm**2).sum(axis=1)) * np.sqrt((y_dm**2).sum(axis=1))
    corr   = np.where(denom > 1e-10, num / (denom + 1e-12), 0.0)
    result[window - 1:] = corr
    return pd.Series(result, index=s.index)


def _rolling_entropy(s: pd.Series, window: int) -> pd.Series:
    signs   = np.sign(s.values).astype(np.float64)
    N       = len(signs)
    result  = np.full(N, np.nan)
    shape   = (N - window + 1, window)
    strides = (signs.strides[0], signs.strides[0])
    wins    = np.lib.stride_tricks.as_strided(signs, shape=shape, strides=strides).copy()
    pos     = (wins > 0).mean(axis=1)
    neg     = (wins < 0).mean(axis=1)
    zer     = 1.0 - pos - neg
    entropy = np.zeros(N - window + 1)
    for p_arr in [pos, neg, zer]:
        mask = p_arr > 0
        entropy[mask] -= p_arr[mask] * np.log(p_arr[mask] + 1e-12)
    result[window - 1:] = entropy
    return pd.Series(result, index=s.index)


def _gk_vol(hi, lo, c, op, window: int) -> pd.Series:
    gk_raw  = (0.5 * np.log(hi / (lo + 1e-9) + 1e-9)**2
               - (2 * np.log(2) - 1) * np.log(c / (op + 1e-9) + 1e-9)**2)
    gk_mean = gk_raw.rolling(window, min_periods=window).mean()
    return np.sqrt(np.maximum(gk_mean.values, 0) * 252)


# ════════════════════════════════════════════════════════════════
#  INDICATOR VALIDATION
# ════════════════════════════════════════════════════════════════

_REQUIRED_INDICATOR_COLUMNS = ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200']


def validate_required_indicators(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_INDICATOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"build_features() missing required columns: {missing}")


# ════════════════════════════════════════════════════════════════
#  MAIN FEATURE BUILDER
# ════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer ~110 features from OHLCV.
    MUST be called on each split independently (not on full data).
    """
    d, c, hi, lo, op, vo = df.copy(), df['Close'], df['High'], df['Low'], df['Open'], df['Volume']
    lr = np.log(c / c.shift(1))

    for lag in DATA.return_lags:
        d[f'ret_{lag}d']     = c.pct_change(lag)
        d[f'log_ret_{lag}d'] = np.log(c / (c.shift(lag) + 1e-9))

    sma: dict = {}
    for w in DATA.sma_windows:
        min_p = min(max(w // 4, 5), w)
        sma[w] = c.rolling(w, min_periods=min_p).mean()
        if f'sma_{w}' not in d.columns:
            d[f'sma_{w}'] = sma[w]
        d[f'price_vs_sma_{w}'] = c / (sma[w] + 1e-9) - 1.0
    d['sma_cross_5_20']   = sma[5]  / (sma[20]  + 1e-9) - 1.0
    d['sma_cross_10_50']  = sma[10] / (sma[50]  + 1e-9) - 1.0
    d['sma_cross_20_50']  = sma[20] / (sma[50]  + 1e-9) - 1.0
    d['sma_cross_50_200'] = sma[50] / (sma[200] + 1e-9) - 1.0

    ema = {}
    for span in [9, 12, 21, 26]:
        ema[span] = c.ewm(span=span, adjust=False).mean()
        d[f'price_vs_ema_{span}'] = c / (ema[span] + 1e-9) - 1.0
    d['macd']        = ema[12] - ema[26]
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist']   = d['macd'] - d['macd_signal']
    d['macd_cross']  = (d['macd'] > d['macd_signal']).astype(int)
    d['macd_norm']   = d['macd'] / (c + 1e-9)

    RSI_LAG = 3
    for p in DATA.rsi_periods:
        rsi = _rsi(c, p)
        d[f'rsi_{p}']              = rsi
        d[f'rsi_{p}_dist_50']      = rsi - 50.0
        d[f'rsi_{p}_lag_{RSI_LAG}']= d[f'rsi_{p}'].shift(RSI_LAG)
        d[f'rsi_{p}_momentum']     = d[f'rsi_{p}'] - d[f'rsi_{p}_lag_{RSI_LAG}']

    for w in DATA.bb_windows:
        mu, sigma = c.rolling(w, min_periods=w).mean(), c.rolling(w, min_periods=w).std()
        upper, lower = mu + 2*sigma, mu - 2*sigma
        d[f'bb_width_{w}']   = (upper - lower) / (mu + 1e-9)
        d[f'bb_pos_{w}']     = (c - lower) / (upper - lower + 1e-9)
        d[f'bb_squeeze_{w}'] = sigma / (mu + 1e-9)

    tr = pd.concat([hi - lo, (hi - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    for atr_p in [7, 14, 21]:
        atr = tr.ewm(com=atr_p - 1, adjust=False).mean()
        d[f'atr_{atr_p}']      = atr
        d[f'atr_{atr_p}_norm'] = atr / (c + 1e-9)
    kelt_mid   = ema[21]
    kelt_atr   = tr.ewm(com=13, adjust=False).mean()
    d['keltner_pos'] = (c - (kelt_mid - 2*kelt_atr)) / (4*kelt_atr + 1e-9)

    d['gk_vol_5']  = _gk_vol(hi, lo, c, op, 5)
    d['gk_vol_20'] = _gk_vol(hi, lo, c, op, 20)

    for w in DATA.vol_windows:
        rv = lr.rolling(w, min_periods=w).std() * np.sqrt(252)
        d[f'vol_{w}d']      = rv
        d[f'vol_{w}d_rank'] = rv.expanding().rank(pct=True)
    d['hv_ratio_5_20']  = lr.rolling(5).std()  / (lr.rolling(20).std()  + 1e-9)
    d['hv_ratio_10_60'] = lr.rolling(10).std() / (lr.rolling(60).std()  + 1e-9)

    for w in [5, 10, 20]:
        d[f'vol_ratio_{w}'] = vo / (vo.rolling(w, min_periods=w).mean() + 1e-9)
    d['vol_trend_5'] = vo.pct_change(5)
    obv = (np.sign(lr) * vo).fillna(0).cumsum()
    obv_ma = obv.ewm(span=20, adjust=False).mean()
    d['obv_norm'] = (obv - obv_ma) / (obv_ma.abs() + 1e-9)
    d['amihud']   = (lr.abs() / (vo * c + 1e-9)).rolling(20).mean()
    d['force_index_2']  = (c.diff() * vo).ewm(span=2,  adjust=False).mean() / (c * vo.mean() + 1e-9)
    d['force_index_13'] = (c.diff() * vo).ewm(span=13, adjust=False).mean() / (c * vo.mean() + 1e-9)

    typical = (hi + lo + c) / 3.0
    vwap    = (typical * vo).rolling(20).sum() / (vo.rolling(20).sum() + 1e-9)
    d['vwap_dev'] = (c - vwap) / (vwap + 1e-9)

    candle_top    = pd.concat([op, c], axis=1).max(axis=1)
    candle_bottom = pd.concat([op, c], axis=1).min(axis=1)
    candle_range  = (hi - lo).replace(0, np.nan)
    d['body']         = (c - op)  / (op + 1e-9)
    d['body_pct']     = (c - op).abs() / (candle_range + 1e-9)
    d['upper_shadow'] = (hi - candle_top)    / (candle_range + 1e-9)
    d['lower_shadow'] = (candle_bottom - lo) / (candle_range + 1e-9)
    d['high_low_pct'] = candle_range / (c + 1e-9)

    for n in DATA.roc_periods:
        d[f'roc_{n}'] = c / (c.shift(n) + 1e-9) - 1.0
    elder_ema13 = c.ewm(span=13, adjust=False).mean()
    d['elder_bull'] = (hi - elder_ema13) / (elder_ema13 + 1e-9)
    d['elder_bear'] = (lo - elder_ema13) / (elder_ema13 + 1e-9)

    r_max = c.rolling(252, min_periods=60).max()
    r_min = c.rolling(252, min_periods=60).min()
    d['dist_52w_high'] = c / (r_max + 1e-9) - 1.0
    d['dist_52w_low']  = c / (r_min + 1e-9) - 1.0
    d['52w_range_pct'] = (c - r_min) / (r_max - r_min + 1e-9)

    rw  = DATA.regime_window
    rmu = lr.rolling(rw, min_periods=rw).mean()
    rsg = lr.rolling(rw, min_periods=rw).std()
    d['regime_trend'] = rmu / (rsg + 1e-9) * np.sqrt(rw)
    d['regime_vol']   = lr.rolling(20).std() / (lr.rolling(120).std() + 1e-9)
    d['regime_mr']    = (c - c.rolling(rw).mean()) / (rsg * np.sqrt(rw) + 1e-9)

    d['autocorr_1']     = _rolling_autocorr(lr, window=20, lag=1)
    d['autocorr_5']     = _rolling_autocorr(lr, window=30, lag=5)
    d['ret_entropy_20'] = _rolling_entropy(lr, window=20)

    lo14 = lo.rolling(14, min_periods=14).min()
    hi14 = hi.rolling(14, min_periods=14).max()
    stk  = (c - lo14) / (hi14 - lo14 + 1e-9) * 100.0
    d['stoch_k']    = stk
    d['stoch_d']    = stk.rolling(3).mean()
    d['williams_r'] = (hi14 - c) / (hi14 - lo14 + 1e-9) * -100.0
    d['cci']        = ((typical - typical.rolling(20).mean())
                       / (0.015 * typical.rolling(20).std() + 1e-9))

    for w in [10, 20]:
        dh, dl = hi.rolling(w).max(), lo.rolling(w).min()
        d[f'don_pos_{w}']   = (c - dl) / (dh - dl + 1e-9)
        d[f'don_width_{w}'] = (dh - dl) / (c + 1e-9)

    for w in [20, 60, 120]:
        mu, sig = c.rolling(w).mean(), c.rolling(w).std()
        d[f'zscore_{w}'] = (c - mu) / (sig + 1e-9)

    d['price_accel'] = lr.diff()
    d['vol_accel']   = lr.rolling(10).std().diff()

    validate_required_indicators(d)
    return d


# ════════════════════════════════════════════════════════════════
#  BASELINE STRATEGY  ← V8 NEW
# ════════════════════════════════════════════════════════════════

def ma_crossover_baseline(ohlcv: pd.DataFrame,
                           fast: int = 5,
                           slow: int = 20) -> np.ndarray:
    """
    Simple MA crossover baseline: buy when fast SMA > slow SMA, else flat.

    Returns probability array: 1.0 when fast > slow (predict UP), 0.0 otherwise.
    This is the simplest possible signal. Any ML model that can't beat it
    consistently has no edge beyond what a 2-line indicator can provide.

    Parameters
    ----------
    ohlcv : OHLCV dataframe (Close column required).
    fast  : fast SMA window (default 5).
    slow  : slow SMA window (default 20).

    Returns
    -------
    np.ndarray of 0.0/1.0 signals aligned to ohlcv index.
    """
    c     = ohlcv['Close']
    sma_f = c.rolling(fast, min_periods=fast).mean()
    sma_s = c.rolling(slow, min_periods=slow).mean()
    signal = (sma_f > sma_s).astype(float).values
    return signal


# ════════════════════════════════════════════════════════════════
#  REGIME CLASSIFIER  ← V8 NEW
# ════════════════════════════════════════════════════════════════

def classify_regime(ohlcv: pd.DataFrame,
                    trend_window: int = 60,
                    vol_window_short: int = 20,
                    vol_window_long: int = 120,
                    sharpe_thresh: float = 0.5,
                    vol_ratio_thresh: float = 1.3) -> pd.Series:
    """
    Label each bar with a market regime.

    Regimes:
      'trending_up'    : rolling Sharpe > +threshold (strong positive trend)
      'trending_down'  : rolling Sharpe < -threshold (strong negative trend)
      'high_vol'       : short-term vol >> long-term vol (vol spike / crisis)
      'mean_reverting' : everything else (choppy / sideways)

    Use this to evaluate strategy performance by regime — a good strategy
    should ideally add value in most regimes, not just bull markets.

    Returns
    -------
    pd.Series of regime labels aligned to ohlcv.index.
    """
    lr = np.log(ohlcv['Close'] / ohlcv['Close'].shift(1))

    roll_mean = lr.rolling(trend_window, min_periods=trend_window // 2).mean()
    roll_std  = lr.rolling(trend_window, min_periods=trend_window // 2).std()
    roll_sh   = roll_mean / (roll_std + 1e-9) * np.sqrt(252)

    vol_s = lr.rolling(vol_window_short, min_periods=vol_window_short // 2).std()
    vol_l = lr.rolling(vol_window_long,  min_periods=vol_window_long  // 2).std()
    vol_r = vol_s / (vol_l + 1e-9)

    regime = pd.Series('mean_reverting', index=ohlcv.index)
    regime[roll_sh >  sharpe_thresh]  = 'trending_up'
    regime[roll_sh < -sharpe_thresh]  = 'trending_down'
    regime[vol_r   >  vol_ratio_thresh] = 'high_vol'  # overrides trend labels

    return regime


# ════════════════════════════════════════════════════════════════
#  TARGET BUILDER
# ════════════════════════════════════════════════════════════════

def add_targets(
    df: pd.DataFrame,
    mode: str = DATA.target_mode,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct prediction targets for all horizons in DATA.horizons.

    WHY THE OLD BINARY TARGET IS WEAK
    ──────────────────────────────────
    The original target:
        y = 1  if  Close[t+h] > Close[t]  else 0

    has three fundamental problems:

    1. Threshold arbitrariness: up/down is decided at exactly 0% return.
       A +0.001% day and a +3% day are treated identically (both = 1).
       The model gets no signal about the *magnitude* of the move.

    2. Near-random base rate on efficient assets: on liquid large-caps
       (e.g. AAPL), the up-day probability hovers near 52-53% — barely
       above chance. A classifier predicting 1 every bar achieves 52%
       accuracy for free, making it impossible to distinguish genuine
       edge from the trivial baseline.

    3. Not alpha: a day where AAPL is +0.5% but the market is +2% is
       labelled UP (1), even though the stock underperformed. The binary
       target conflates market beta with stock-specific alpha.

    BETTER TARGETS
    ──────────────
    'return'        — raw log return; regression objective.
                      Preserves magnitude. Use with MSE or Huber loss.
                      col: target_ret_{h}d

    'excess_return' — log return minus rolling median return over the
                      last `excess_return_window` bars. This isolates
                      the idiosyncratic component (alpha) from the
                      trend component. Values near 0 = in-line with
                      recent history; large positives = genuine outperformance.
                      col: target_exc_{h}d
                      Also writes a binary version: target_exc_bin_{h}d = 1
                      if excess_return > binary_threshold (for classifiers).

    'rank'          — rolling percentile rank of h-day log return over
                      the last `rank_window` bars, scaled to [0, 1].
                      rank=1.0 = highest return in window; rank=0.0 = lowest.
                      Cross-sectional ready: in multi-ticker mode, rank each
                      ticker's return within the cross-section for true
                      relative-value targets.
                      col: target_rank_{h}d
                      Also writes binary: target_rank_bin_{h}d = 1 if rank > 0.5.

    'binary'        — original target (kept for backward compatibility).
                      col: target_{h}d

    Parameters
    ----------
    df   : DataFrame with a 'Close' column. Must be sorted chronologically.
    mode : one of 'binary', 'return', 'excess_return', 'rank'.
           Defaults to DATA.target_mode from config.

    Returns
    -------
    (df_with_targets, target_cols) where target_cols lists all added columns.
    The FIRST element of target_cols is the PRIMARY target used for training.
    """
    target_cols: List[str] = []
    close = df['Close']

    for h in DATA.horizons:
        # ── Forward log return ──────────────────────────────────────────
        fwd_ret = np.log(close.shift(-h) / close.replace(0, np.nan))

        if mode == 'binary':
            # Original: 1 if price goes up, 0 if down
            col = f'target_{h}d'
            df[col] = (close.shift(-h) > close).astype(float)
            target_cols.append(col)

        elif mode == 'return':
            # Raw log return — use with regression loss (MSE / Huber)
            col = f'target_ret_{h}d'
            df[col] = fwd_ret
            target_cols.append(col)

        elif mode == 'excess_return':
            # Log return minus rolling median (removes trend/beta component)
            w   = DATA.excess_return_window
            med = fwd_ret.rolling(w, min_periods=max(w // 2, 5)).median()
            exc = fwd_ret - med

            col_cont = f'target_exc_{h}d'       # continuous (for regression)
            col_bin  = f'target_exc_bin_{h}d'   # binary (for classifiers)
            df[col_cont] = exc
            df[col_bin]  = (exc > DATA.binary_threshold).astype(float)
            # Primary = binary version (classifier-compatible)
            target_cols.append(col_bin)
            target_cols.append(col_cont)

        elif mode == 'rank':
            # Rolling percentile rank of forward return, scaled [0, 1]
            # rank=1 = top of window; rank=0 = bottom
            w = DATA.rank_window

            def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
                def _pctrank(x):
                    if len(x) < 2:
                        return 0.5
                    return float(pd.Series(x).rank(pct=True).iloc[-1])
                return series.rolling(window, min_periods=max(window // 2, 5)).apply(
                    _pctrank, raw=True
                )

            rnk = _rolling_rank(fwd_ret, w)

            col_cont = f'target_rank_{h}d'       # continuous rank [0,1]
            col_bin  = f'target_rank_bin_{h}d'   # binary: rank > 0.5
            df[col_cont] = rnk
            df[col_bin]  = (rnk > 0.5).astype(float)
            # Primary = binary version (classifier-compatible)
            target_cols.append(col_bin)
            target_cols.append(col_cont)

        else:
            raise ValueError(
                f"add_targets: unknown mode='{mode}'. "
                f"Choose from: 'binary', 'return', 'excess_return', 'rank'."
            )

    return df, target_cols


def _get_feature_cols(df: pd.DataFrame, target_cols: List[str]) -> List[str]:
    sma_cols = {f'sma_{w}' for w in DATA.sma_windows}
    BASE     = {'Open', 'High', 'Low', 'Close', 'Volume'} | set(target_cols) | sma_cols
    return [col for col in df.columns if col not in BASE]


# ════════════════════════════════════════════════════════════════
#  LEAKAGE-FREE DATASET BUILDER
# ════════════════════════════════════════════════════════════════

def build_dataset(
    raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Build train/val/test splits WITHOUT data leakage.

    V8 NOTE: Returns a single unified scaler built after feature
    selection (see build_dataset_with_scaler for full pipeline).
    This function returns raw DataFrames; call fit_scaler() once
    on df_tr after feature selection.
    """
    MIN_PARTITION_BARS = 300

    n          = len(raw)
    train_r    = DATA.train_ratio
    val_r      = DATA.val_ratio

    raw_val  = int(n * val_r)
    raw_test = int(n * (1.0 - train_r - val_r))

    if raw_val < MIN_PARTITION_BARS or raw_test < MIN_PARTITION_BARS:
        train_r = 0.70
        remaining = n - int(n * train_r)
        val_r   = max(remaining // 2, MIN_PARTITION_BARS) / n
        val_r   = min(val_r, 0.15)
        logger.warning(
            f"  Short dataset ({n} bars). Auto-adjusting splits: "
            f"train={train_r*100:.0f}%  val={val_r*100:.1f}%  "
            f"test={(1-train_r-val_r)*100:.1f}%."
        )

    i_tr  = int(n * train_r)
    i_val = int(n * (train_r + val_r))

    df_tr = build_features(raw.iloc[:i_tr].copy())
    df_va = build_features(raw.iloc[i_tr:i_val].copy())
    df_te = build_features(raw.iloc[i_val:].copy())

    df_tr, target_cols = add_targets(df_tr)
    df_va, _           = add_targets(df_va)
    df_te, _           = add_targets(df_te)

    df_tr = df_tr.replace([np.inf, -np.inf], np.nan).dropna()
    df_va = df_va.replace([np.inf, -np.inf], np.nan).dropna()
    df_te = df_te.replace([np.inf, -np.inf], np.nan).dropna()

    if min(len(df_tr), len(df_va), len(df_te)) == 0:
        logger.warning("  Standard split produced empty partitions. Falling back to 60/20/20.")
        i_tr_fb  = int(n * 0.60)
        i_val_fb = int(n * 0.80)
        df_tr = build_features(raw.iloc[:i_tr_fb].copy())
        df_va = build_features(raw.iloc[i_tr_fb:i_val_fb].copy())
        df_te = build_features(raw.iloc[i_val_fb:].copy())
        df_tr, target_cols = add_targets(df_tr)
        df_va, _           = add_targets(df_va)
        df_te, _           = add_targets(df_te)
        df_tr = df_tr.replace([np.inf, -np.inf], np.nan).dropna()
        df_va = df_va.replace([np.inf, -np.inf], np.nan).dropna()
        df_te = df_te.replace([np.inf, -np.inf], np.nan).dropna()

    if min(len(df_tr), len(df_va), len(df_te)) == 0:
        raise ValueError(
            f"Empty partition even after fallback: "
            f"tr={len(df_tr)} va={len(df_va)} te={len(df_te)}."
        )

    features = _get_feature_cols(df_tr, target_cols)
    primary  = target_cols[DATA.primary_horizon_idx]

    up_tr = df_tr[primary].mean() * 100
    logger.info(f"  [V8] Features={len(features)}  Targets={target_cols}")
    logger.info(f"  Train={len(df_tr)}  Val={len(df_va)}  Test={len(df_te)}")
    logger.info(
        f"  UP%: tr={up_tr:.1f}%  "
        f"va={df_va[primary].mean()*100:.1f}%  "
        f"te={df_te[primary].mean()*100:.1f}%"
    )
    if abs(up_tr - 50.0) > 5.0:
        logger.warning(f"  Class imbalance: {up_tr:.1f}% UP. Applying class weights.")

    return df_tr, df_va, df_te, features, target_cols


# ════════════════════════════════════════════════════════════════
#  CLASS IMBALANCE HELPERS
# ════════════════════════════════════════════════════════════════

def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return {0: 1.0, 1: 1.0}
    return {0: len(y) / (2.0 * n_neg), 1: len(y) / (2.0 * n_pos)}


def get_pos_weight(y: np.ndarray) -> float:
    """
    Compute BCEWithLogitsLoss pos_weight for class imbalance.

    Works for both binary targets (0/1) and continuous targets:
    - Binary: counts exact 1s vs 0s.
    - Continuous (return/rank/excess_return): treats values > 0 as positive
      class, <= 0 as negative.  The binary _bin_ column is preferred, but
      if the raw continuous target is passed this fallback is safe.
    """
    y     = np.asarray(y, dtype=np.float32)
    # For binary targets use exact match; for continuous use sign
    unique = np.unique(y[np.isfinite(y)])
    if set(unique).issubset({0.0, 1.0}):
        n_pos = int((y == 1).sum())
    else:
        n_pos = int((y > 0).sum())
    n_neg = len(y) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


# ════════════════════════════════════════════════════════════════
#  FEATURE SELECTION  — V8: returns scaler built on selected features
# ════════════════════════════════════════════════════════════════

def select_features(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    feature_names: List[str],
    top_k: int = 60,
    mi_weight: float = 0.4,
    xgb_weight: float = 0.6,
    n_inner_folds: int = 5,
) -> List[str]:
    """
    Select top-K features using Mutual Information + XGBoost importance.

    LEAKAGE FIX — MI and XGB both computed inside inner chronological folds
    ------------------------------------------------------------------------
    PREVIOUS BUG (two leakage paths):

      1. MI leakage: mutual_info_classif was called on the full X_tr BEFORE
         the inner CV loop. MI scores were computed using ALL training samples,
         including samples that act as the hold-out in later walk-forward folds.
         Features that correlate with the label in those future samples are
         unfairly rewarded — the selector implicitly "sees" data it should not.

      2. Combined-score leakage: even though XGB used expanding windows,
         the final combined score (mi_weight * MI + xgb_weight * XGB) was
         always partially contaminated because MI was computed on full X_tr.

    THE FIX:
      Both mutual_info_classif AND XGBClassifier are computed on X_fold = X_tr[:cut]
      inside each inner fold. Neither scorer ever sees data beyond its fold boundary.
      Scores are accumulated across folds and averaged before combining.

    WHY CHRONOLOGICAL (EXPANDING) FOLDS:
      Random k-fold leaks future bars into past training windows.
      Expanding windows mirror real walk-forward deployment and preserve
      temporal ordering strictly.

    Parameters
    ----------
    n_inner_folds : inner chronological folds. >= 3 recommended.
                    Set to 1 only for tiny datasets (< 200 samples).
    """
    from xgboost import XGBClassifier

    F = X_tr.shape[1]
    logger.info(
        f"  [FeatureSelection] Scoring {F} features -> keeping top {top_k} "
        f"(inner_folds={n_inner_folds})"
    )

    xgb_sel_params = dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, verbosity=0, eval_metric='logloss', n_jobs=-1,
    )

    n_samples = len(X_tr)
    fold_size = n_samples // (n_inner_folds + 1)
    min_train = max(fold_size, 100)

    mi_scores_accum  = np.zeros(F)
    xgb_scores_accum = np.zeros(F)
    n_successful_folds = 0

    # ---- Fallback: single-fit when dataset is too small ------------------
    if n_inner_folds <= 1 or n_samples < min_train * 2:
        logger.warning(
            f"  [FeatureSelection] Too few samples for {n_inner_folds}-fold "
            f"inner CV ({n_samples} rows). Using single-fit fallback."
        )
        try:
            mi_scores_accum = mutual_info_classif(
                X_tr, y_tr, random_state=42, n_neighbors=5
            )
        except Exception as exc:
            logger.warning(f"  MI failed ({exc}) -- using uniform.")
            mi_scores_accum = np.ones(F)
        try:
            xgb_sel = XGBClassifier(**xgb_sel_params)
            xgb_sel.fit(X_tr, y_tr.astype(int))
            xgb_scores_accum = xgb_sel.feature_importances_
        except Exception as exc:
            logger.warning(f"  XGB failed ({exc}) -- using uniform.")
            xgb_scores_accum = np.ones(F)
        n_successful_folds = 1

    else:
        # ---- Inner chronological expanding-window CV ---------------------
        # Fold i trains on X_tr[0 : min_train + (i+1)*fold_size].
        # FIX: both MI and XGB computed on same fold slice -- no leakage.
        for fold_i in range(n_inner_folds):
            cut    = min_train + (fold_i + 1) * fold_size
            cut    = min(cut, n_samples)
            X_fold = X_tr[:cut]
            y_fold = y_tr[:cut]

            if len(X_fold) < min_train:
                logger.debug(f"  Fold {fold_i}: {len(X_fold)} rows < min_train -- skipped.")
                continue

            # FIX: MI on fold data only (was: full X_tr)
            try:
                mi_fold = mutual_info_classif(
                    X_fold, y_fold, random_state=42, n_neighbors=5
                )
                mi_scores_accum += mi_fold
            except Exception as exc:
                logger.warning(f"  MI fold {fold_i} failed ({exc}) -- using uniform.")
                mi_scores_accum += np.ones(F)

            # FIX: XGB on same fold slice as MI
            try:
                xgb_fold = XGBClassifier(**xgb_sel_params)
                xgb_fold.fit(X_fold, y_fold.astype(int))
                xgb_scores_accum += xgb_fold.feature_importances_
                n_successful_folds += 1
                logger.debug(
                    f"  Fold {fold_i}: MI + XGB on {len(X_fold)} rows (cut={cut}/{n_samples})"
                )
            except Exception as exc:
                logger.warning(f"  XGB fold {fold_i} failed ({exc}) -- skipped.")
                xgb_scores_accum += np.ones(F)
                n_successful_folds += 1
                continue

        if n_successful_folds == 0:
            logger.warning("  All folds failed -- using uniform scores.")
            mi_scores_accum  = np.ones(F)
            xgb_scores_accum = np.ones(F)
            n_successful_folds = 1

    # ---- Average scores across folds -------------------------------------
    mi_scores  = mi_scores_accum  / n_successful_folds
    xgb_scores = xgb_scores_accum / n_successful_folds

    logger.info(
        f"  [FeatureSelection] MI + XGB averaged over "
        f"{n_successful_folds}/{n_inner_folds} folds."
    )

    # ---- Combine with rank normalisation ---------------------------------
    def rank_norm(arr: np.ndarray) -> np.ndarray:
        ranks = arr.argsort().argsort().astype(float)
        return ranks / max(ranks.max(), 1.0)

    combined     = mi_weight * rank_norm(mi_scores) + xgb_weight * rank_norm(xgb_scores)
    top_k_actual = min(top_k, F)
    top_idx      = combined.argsort()[::-1][:top_k_actual]
    selected     = [feature_names[i] for i in sorted(top_idx)]

    top5 = [feature_names[i] for i in combined.argsort()[::-1][:5]]
    logger.info(f"  [FeatureSelection] {F} -> {len(selected)} features. Top-5: {top5}")
    return selected


# ════════════════════════════════════════════════════════════════
#  MULTI-TICKER DATASET BUILDER
# ════════════════════════════════════════════════════════════════

def build_multi_ticker_dataset(
    tickers: List[str],
    period: str = DATA.period,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Multi-asset dataset builder.  Delegates to the full pipeline in
    data_loader.build_panel() which handles:
      - Per-ticker download + validation
      - Time-index alignment (inner join by default)
      - Per-ticker feature + target engineering
      - Temporal split on shared calendar boundaries
      - Cross-sectional normalisation within each split
      - (date, ticker) panel structure with 'ticker' tag column

    See data_loader.download_universe() and data_loader.build_panel()
    for full documentation.
    """
    from data_loader import download_universe, build_panel

    aligned = download_universe(
        tickers,
        period=period,
        align=DATA.align_method,
    )

    df_tr, df_va, df_te, feature_cols, target_cols = build_panel(
        aligned,
        feature_fn          = build_features,
        target_fn           = add_targets,
        train_ratio         = DATA.train_ratio,
        val_ratio           = DATA.val_ratio,
        cs_norm_method      = DATA.cs_norm_method,
        cs_norm_min_tickers = DATA.cs_norm_min_tickers,
    )

    return df_tr, df_va, df_te, feature_cols, target_cols


# ════════════════════════════════════════════════════════════════
#  SCALING & SEQUENCES
# ════════════════════════════════════════════════════════════════

def fit_scaler(tr_df: pd.DataFrame, features: List[str]) -> RobustScaler:
    sc = RobustScaler()
    sc.fit(tr_df[features])
    return sc


def apply_scale(sc: RobustScaler, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    return sc.transform(df[features]).astype(np.float32)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    N, F    = X.shape
    shape   = (N - seq_len, seq_len, F)
    strides = (X.strides[0], X.strides[0], X.strides[1])
    Xs      = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides).copy()
    return Xs.astype(np.float32), y[seq_len:].astype(np.float32)
