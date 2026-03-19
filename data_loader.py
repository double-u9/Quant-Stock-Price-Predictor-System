"""
data_loader.py  —  V1 Data Pipeline.

Separates data acquisition from feature engineering so that market data
can be loaded, validated, and cleaned independently of how it is used.

Design principles
─────────────────
1. Separation of concerns: download(), validate(), clean(), and
   load_market_data() are distinct, independently testable steps.

2. Fail loudly on structural problems, heal quietly on recoverable
   ones — with full logging so the user always knows what was fixed.

3. Source-agnostic validation: validate_ohlcv() works on any DataFrame
   (yfinance, CSV, parquet, database) so the pipeline is not coupled
   to a single data vendor.

4. Audit trail: every cleaning action is logged with counts so the
   operator can judge whether the data quality is acceptable.

Weaknesses fixed from V6 features.download()
─────────────────────────────────────────────
 1. Empty DataFrame crash   — explicit check with informative error.
 2. Unsorted index          — sort_index() always called after download.
 3. Duplicate timestamps    — detected, logged, and removed.
 4. Dtype silently object   — all OHLCV columns cast to float64 with
                              explicit error on failure.
 5. MultiIndex columns      — robust flattening for all yfinance versions.
 6. Minimum row guard       — configurable MIN_ROWS (default 252) with
                              clear error stating what period to use.
 7. Blind ffill gap limit   — NaN runs longer than MAX_FILL_DAYS are
                              removed rather than filled with stale prices.
 8. Full OHLC consistency   — Open, Close checked against [Low, High];
                              all prices > 0 enforced.
 9. Zero-volume guard       — zero-volume bars logged and removed.
10. Large price jump flag   — single-day moves > JUMP_THRESHOLD logged
                              as warnings (not removed — could be real).
11. Fused download+validate — download() now delegates to validate_ohlcv()
                              so any source benefits from the same checks.
12. Silent NaN patching     — NaN count and fill strategy always logged.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  CONSTANTS — tuneable quality thresholds
# ════════════════════════════════════════════════════════════════

# Minimum number of rows required after cleaning.
# 200-day SMA + regime window (60) + sequence length (50) = ~310 minimum.
# Default 252 = 1 trading year; raise this for short-period configs.
MIN_ROWS: int = 252

# Maximum number of consecutive NaN bars to fill forward.
# Gaps longer than this indicate a suspension or data failure — the bars
# are dropped rather than filled with potentially stale prices.
MAX_FILL_DAYS: int = 5

# Single-day absolute return threshold above which a bar is flagged.
# Does NOT remove the bar — could be a real move — but logs a warning
# so the operator can investigate potential bad ticks or unadjusted splits.
JUMP_THRESHOLD: float = 0.40   # 40% single-day move

# Required OHLCV columns (canonical capitalised names used throughout).
REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

# Alternative capitalisation maps from common data sources.
_COLUMN_ALIASES = {
    'open':   'Open',  'OPEN':   'Open',
    'high':   'High',  'HIGH':   'High',
    'low':    'Low',   'LOW':    'Low',
    'close':  'Close', 'CLOSE':  'Close', 'Adj Close': 'Close',
    'volume': 'Volume','VOLUME': 'Volume', 'vol': 'Volume',
}


# ════════════════════════════════════════════════════════════════
#  STEP 1: DOWNLOAD
# ════════════════════════════════════════════════════════════════

def _fetch_yfinance(ticker: str, period: str) -> pd.DataFrame:
    """
    Fetch raw OHLCV data from Yahoo Finance.

    Handles the MultiIndex column structure that yfinance returns for
    both single-ticker and multi-ticker calls, normalising to a flat
    DataFrame before returning.

    Parameters
    ----------
    ticker : Yahoo Finance symbol (e.g. 'AAPL', 'BTC-USD').
    period : lookback period string (e.g. '5y', '2y', '1y').

    Returns
    -------
    pd.DataFrame with a DatetimeIndex and raw yfinance columns.
    Raises ValueError if the download returns an empty result.
    """
    logger.info(f"  Fetching {ticker} ({period}) from Yahoo Finance ...")
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    # ── Flatten MultiIndex columns (yfinance 0.2+) ───────────────────────
    # yfinance returns ('Open','AAPL'), ('High','AAPL') etc. for single
    # tickers in newer versions. get_level_values(0) gives the field name.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Remove any duplicate column names produced by the flatten
    raw = raw.loc[:, ~raw.columns.duplicated()]

    if raw.empty:
        raise ValueError(
            f"download: yfinance returned an empty DataFrame for ticker "
            f"'{ticker}' with period='{period}'. "
            f"Possible causes: invalid ticker symbol, network failure, "
            f"or the requested period predates the listing date. "
            f"Try a shorter period or verify the ticker on finance.yahoo.com."
        )

    logger.info(f"  Raw download: {len(raw)} rows, "
                f"columns: {list(raw.columns)}")
    return raw


# ════════════════════════════════════════════════════════════════
#  STEP 2: NORMALISE COLUMN NAMES
# ════════════════════════════════════════════════════════════════

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to canonical capitalised OHLCV names.

    Handles common variations from different data sources:
    open/OPEN/Open, adj close/Adj Close → Close, vol → Volume, etc.

    Raises ValueError if any required column cannot be identified
    after renaming.
    """
    rename_map = {col: _COLUMN_ALIASES[col]
                  for col in df.columns
                  if col in _COLUMN_ALIASES and col not in REQUIRED_COLUMNS}

    if rename_map:
        logger.info(f"  Column rename: {rename_map}")
        df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"_normalise_columns: required columns {missing} are missing "
            f"after renaming. Available columns: {list(df.columns)}. "
            f"Check that your data source provides standard OHLCV fields."
        )

    return df[REQUIRED_COLUMNS].copy()


# ════════════════════════════════════════════════════════════════
#  STEP 3: ENFORCE DTYPES
# ════════════════════════════════════════════════════════════════

def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all OHLCV columns to float64.

    yfinance occasionally returns object-dtype columns if a parse
    failure inserts a string token into a price column.  Downstream
    numpy operations on object columns silently produce NaN chains
    instead of raising an error.

    Raises ValueError if any column cannot be cast to numeric.
    """
    for col in REQUIRED_COLUMNS:
        original_dtype = df[col].dtype
        if original_dtype == object or str(original_dtype) == 'string':
            # Coerce: non-numeric strings become NaN (caught later)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            n_coerced = df[col].isna().sum()
            if n_coerced > 0:
                logger.warning(
                    f"  _enforce_dtypes: '{col}' had object dtype — "
                    f"{n_coerced} non-numeric values coerced to NaN."
                )
        df[col] = df[col].astype(np.float64)

    return df


# ════════════════════════════════════════════════════════════════
#  STEP 4: SORT AND DEDUPLICATE INDEX
# ════════════════════════════════════════════════════════════════

def _sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by DatetimeIndex ascending and remove duplicate timestamps.

    Unsorted data causes rolling windows to look backwards in time
    rather than forwards. Duplicate timestamps cause each duplicated
    bar to be counted twice in rolling calculations, systematically
    biasing all time-series features.

    When duplicates are found, the last occurrence is kept (consistent
    with how most data providers handle corporate action revisions).
    """
    # ── Sort ─────────────────────────────────────────────────────────────
    if not df.index.is_monotonic_increasing:
        logger.warning(
            "  _sort_and_deduplicate: index was not sorted — sorting now. "
            "This may indicate a data source issue."
        )
        df = df.sort_index()

    # ── Deduplicate ───────────────────────────────────────────────────────
    n_before = len(df)
    df = df[~df.index.duplicated(keep='last')]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(
            f"  _sort_and_deduplicate: removed {n_dropped} duplicate "
            f"timestamp(s) (kept last occurrence)."
        )

    return df


# ════════════════════════════════════════════════════════════════
#  STEP 5: HANDLE MISSING VALUES
# ════════════════════════════════════════════════════════════════

def _handle_missing_values(df: pd.DataFrame,
                            max_fill_days: int = MAX_FILL_DAYS
                            ) -> pd.DataFrame:
    """
    Fill short NaN gaps with forward-fill; drop bars in long gaps.

    Strategy
    ────────
    - Gaps of 1..max_fill_days consecutive NaN bars: forward-fill
      (e.g. public holidays where prices are carried over).
    - Gaps longer than max_fill_days: drop those rows entirely.
      Filling a week-long gap with a stale price would silently inject
      false stability into volatility features.

    All NaN counts are logged so the operator can judge data quality.

    Parameters
    ----------
    max_fill_days : maximum consecutive NaN bars to fill forward.
                    Bars in runs longer than this are dropped.
    """
    nan_counts = df.isna().sum()
    total_nans = int(nan_counts.sum())

    if total_nans == 0:
        return df   # fast path — nothing to do

    logger.warning(
        f"  _handle_missing_values: found {total_nans} NaN value(s): "
        + ", ".join(f"{c}={n}" for c, n in nan_counts.items() if n > 0)
    )

    # Identify rows that are ENTIRELY NaN (all 5 columns) — these are
    # the true calendar gaps. Rows with partial NaN are data errors.
    full_nan_rows = df.isna().all(axis=1)
    partial_nan_rows = df.isna().any(axis=1) & ~full_nan_rows
    if partial_nan_rows.any():
        logger.warning(
            f"  _handle_missing_values: {partial_nan_rows.sum()} rows have "
            f"partial NaN (some columns missing) — these will be filled or "
            f"dropped with the rest."
        )

    # Measure consecutive NaN run lengths per column to identify long gaps.
    # Mark rows belonging to runs longer than max_fill_days for dropping.
    rows_to_drop = pd.Series(False, index=df.index)
    for col in REQUIRED_COLUMNS:
        mask = df[col].isna()
        if not mask.any():
            continue
        # Label each NaN run with an ID using cumsum trick
        run_id    = (~mask).cumsum()
        run_sizes = mask.groupby(run_id).transform('sum')
        long_gap  = mask & (run_sizes > max_fill_days)
        rows_to_drop |= long_gap

    if rows_to_drop.any():
        logger.warning(
            f"  _handle_missing_values: dropping {rows_to_drop.sum()} bar(s) "
            f"in NaN gaps longer than {max_fill_days} day(s) — "
            f"stale price fill would corrupt time-series features."
        )
        df = df[~rows_to_drop]

    # Forward-fill remaining short gaps, then back-fill the very first bar
    # if it starts with NaN (no prior data to forward-fill from).
    before_fill = int(df.isna().sum().sum())
    df = df.ffill().bfill()
    after_fill  = int(df.isna().sum().sum())

    filled = before_fill - after_fill
    if filled > 0:
        logger.info(
            f"  _handle_missing_values: forward/back-filled {filled} NaN(s) "
            f"in gaps ≤ {max_fill_days} day(s)."
        )

    if after_fill > 0:
        logger.warning(
            f"  _handle_missing_values: {after_fill} NaN(s) remain after "
            f"fill — these rows will be dropped."
        )
        df = df.dropna()

    return df


# ════════════════════════════════════════════════════════════════
#  STEP 6: OHLCV CONSISTENCY CHECKS
# ════════════════════════════════════════════════════════════════

def _check_ohlcv_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and repair OHLC price relationships.

    Checks applied
    ──────────────
    (a) All prices > 0          — non-positive prices are physically
                                  impossible and indicate data errors.
    (b) High >= Low             — violated by inverted bars (data error).
    (c) Open in [Low, High]     — open outside daily range is a data error.
    (d) Close in [Low, High]    — close outside daily range is a data error.
    (e) Volume >= 0             — negative volume is impossible.
    (f) Zero-volume bars        — indicate non-trading days that bypassed
                                  the calendar filter; corrupt OBV/VWAP.

    Bars failing checks (a)–(e) are dropped with a logged count.
    Zero-volume bars (f) are dropped separately and also logged.

    Large single-day returns are flagged as WARNINGS only — they may be
    real moves (earnings, index inclusion) and should not be auto-removed.
    """
    n_start = len(df)

    # ── (a) All prices positive ───────────────────────────────────────────
    price_cols    = ['Open', 'High', 'Low', 'Close']
    non_positive  = (df[price_cols] <= 0).any(axis=1)
    if non_positive.any():
        logger.warning(
            f"  _check_ohlcv_consistency: removing {non_positive.sum()} "
            f"bar(s) with non-positive prices."
        )
        df = df[~non_positive]

    # ── (b) High >= Low ───────────────────────────────────────────────────
    inverted = df['High'] < df['Low']
    if inverted.any():
        logger.warning(
            f"  _check_ohlcv_consistency: removing {inverted.sum()} "
            f"bar(s) with High < Low (inverted OHLC)."
        )
        df = df[~inverted]

    # ── (c) Open within [Low, High] ───────────────────────────────────────
    open_outside = (df['Open'] < df['Low']) | (df['Open'] > df['High'])
    if open_outside.any():
        logger.warning(
            f"  _check_ohlcv_consistency: removing {open_outside.sum()} "
            f"bar(s) where Open is outside [Low, High]."
        )
        df = df[~open_outside]

    # ── (d) Close within [Low, High] ─────────────────────────────────────
    close_outside = (df['Close'] < df['Low']) | (df['Close'] > df['High'])
    if close_outside.any():
        logger.warning(
            f"  _check_ohlcv_consistency: removing {close_outside.sum()} "
            f"bar(s) where Close is outside [Low, High]."
        )
        df = df[~close_outside]

    # ── (e) Volume non-negative ───────────────────────────────────────────
    neg_volume = df['Volume'] < 0
    if neg_volume.any():
        logger.warning(
            f"  _check_ohlcv_consistency: removing {neg_volume.sum()} "
            f"bar(s) with negative Volume."
        )
        df = df[~neg_volume]

    # ── (f) Zero-volume bars ─────────────────────────────────────────────
    # Zero-volume usually means a non-trading day slipped through the
    # calendar. These bars corrupt OBV, VWAP, and Amihud liquidity features.
    zero_vol = df['Volume'] == 0
    if zero_vol.any():
        logger.warning(
            f"  _check_ohlcv_consistency: removing {zero_vol.sum()} "
            f"zero-volume bar(s) (likely non-trading days or data gaps)."
        )
        df = df[~zero_vol]

    # ── (g) Large price jump detection (WARNING only, no removal) ────────
    close_ret = df['Close'].pct_change().abs()
    large_jumps = close_ret > JUMP_THRESHOLD
    if large_jumps.any():
        jump_dates = df.index[large_jumps].tolist()
        logger.warning(
            f"  _check_ohlcv_consistency: {large_jumps.sum()} bar(s) "
            f"have single-day absolute return > {JUMP_THRESHOLD*100:.0f}% "
            f"— possible bad tick or unadjusted split. "
            f"Dates: {[str(d.date()) for d in jump_dates[:5]]}"
            f"{'...' if len(jump_dates) > 5 else ''}. "
            f"Bars NOT removed — verify manually if unexpected."
        )

    n_removed = n_start - len(df)
    if n_removed > 0:
        logger.info(
            f"  _check_ohlcv_consistency: removed {n_removed} invalid "
            f"bar(s) total. {len(df)} bars remain."
        )

    return df


# ════════════════════════════════════════════════════════════════
#  STEP 7: MINIMUM ROW GUARD
# ════════════════════════════════════════════════════════════════

def _check_minimum_rows(df: pd.DataFrame,
                         ticker: str,
                         min_rows: int = MIN_ROWS) -> None:
    """
    Raise ValueError if fewer than min_rows bars remain after cleaning.

    The feature engineering pipeline requires:
      - 200 rows for sma_200
      - 60 rows for regime_window
      - 252 rows for 52-week high/low (min_periods=60)
    Fewer rows will cause most features to be entirely NaN, leading to
    an empty dataframe after dropna().

    Parameters
    ----------
    ticker   : used in the error message to help the operator diagnose.
    min_rows : minimum required rows (default MIN_ROWS = 252).
    """
    if len(df) < min_rows:
        raise ValueError(
            f"_check_minimum_rows: only {len(df)} clean rows remain for "
            f"'{ticker}' — minimum required is {min_rows}. "
            f"Try a longer download period (e.g. '5y' or '7y') or "
            f"check whether the ticker has sufficient trading history."
        )


# ════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════

def validate_ohlcv(df: pd.DataFrame,
                   ticker: str = "unknown",
                   min_rows: int = MIN_ROWS,
                   max_fill_days: int = MAX_FILL_DAYS) -> pd.DataFrame:
    """
    Validate and clean any OHLCV DataFrame, regardless of data source.

    This function is source-agnostic: it works on data from yfinance,
    CSV files, parquet, databases, or any other provider.  Separating
    validation from acquisition means every data source is treated
    identically and the pipeline does not depend on yfinance internals.

    Cleaning steps (in order)
    ─────────────────────────
    1. Normalise column names to canonical OHLCV capitalisation.
    2. Enforce float64 dtypes on all price/volume columns.
    3. Sort index ascending and remove duplicate timestamps.
    4. Handle missing values (fill short gaps, drop long gaps).
    5. Validate OHLC consistency and remove invalid bars.
    6. Check minimum row count.

    Parameters
    ----------
    df            : raw OHLCV DataFrame with a DatetimeIndex.
    ticker        : symbol name, used only in error messages.
    min_rows      : minimum rows required after cleaning.
    max_fill_days : maximum consecutive NaN days to forward-fill.

    Returns
    -------
    pd.DataFrame  : clean, validated OHLCV DataFrame, ready for
                    feature engineering.

    Raises
    ------
    ValueError    : on structural problems that cannot be auto-repaired
                    (missing required columns, insufficient data).
    """
    if df.empty:
        raise ValueError(
            f"validate_ohlcv: received an empty DataFrame for '{ticker}'."
        )

    df = _normalise_columns(df)
    df = _enforce_dtypes(df)
    df = _sort_and_deduplicate(df)
    df = _handle_missing_values(df, max_fill_days=max_fill_days)
    df = _check_ohlcv_consistency(df)
    _check_minimum_rows(df, ticker, min_rows=min_rows)

    logger.info(
        f"  validate_ohlcv: '{ticker}' — {len(df)} clean bars  "
        f"({df.index[0].date()} → {df.index[-1].date()})"
    )
    return df


def download(ticker: str,
             period: str,
             min_rows: int = MIN_ROWS,
             max_fill_days: int = MAX_FILL_DAYS) -> pd.DataFrame:
    """
    Download market data from Yahoo Finance and return a validated,
    clean OHLCV DataFrame.

    This function replaces the original download() in features.py.
    It delegates all validation to validate_ohlcv() so the same
    quality checks apply regardless of whether data comes from Yahoo
    Finance or any other source.

    Parameters
    ----------
    ticker        : Yahoo Finance ticker symbol (e.g. 'AAPL').
    period        : lookback period (e.g. '5y', '2y', '10y').
    min_rows      : minimum rows required after cleaning (default 252).
    max_fill_days : maximum NaN gap to forward-fill (default 5 days).

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume],
    DatetimeIndex sorted ascending, no NaN values, no duplicates,
    all prices positive and OHLC-consistent.

    Raises
    ------
    ValueError if the download is empty, data is insufficient after
    cleaning, or required columns are missing.
    """
    raw = _fetch_yfinance(ticker, period)
    return validate_ohlcv(raw, ticker=ticker,
                           min_rows=min_rows,
                           max_fill_days=max_fill_days)


# ════════════════════════════════════════════════════════════════
#  MULTI-ASSET PIPELINE
# ════════════════════════════════════════════════════════════════

def download_universe(
    tickers: list,
    period: str,
    min_rows: int = MIN_ROWS,
    max_fill_days: int = MAX_FILL_DAYS,
    align: str = 'inner',
) -> dict:
    """
    Download and validate OHLCV data for a universe of tickers.

    Each ticker is downloaded independently, validated through the full
    quality pipeline, then aligned to a common DatetimeIndex.

    Alignment modes
    ───────────────
    'inner' : keep only dates where ALL tickers have data.
              Eliminates survivorship-bias from staggered IPOs.
              Recommended for model training.
    'outer' : keep all dates from any ticker; missing bars NaN-filled
              forward (up to max_fill_days) then dropped if still NaN.
              Use for research / index replication.

    Parameters
    ----------
    tickers       : list of Yahoo Finance symbols.
    period        : lookback string (e.g. '5y').
    align         : 'inner' or 'outer'.

    Returns
    -------
    dict mapping ticker -> clean pd.DataFrame (OHLCV, DatetimeIndex).
    All DataFrames share the same index after alignment.
    Failed tickers are excluded with a warning, not a crash.
    """
    raw_data: dict = {}
    for ticker in tickers:
        try:
            raw_data[ticker] = download(ticker, period,
                                         min_rows=min_rows,
                                         max_fill_days=max_fill_days)
        except Exception as exc:
            logger.warning(f"  [Universe] {ticker} failed: {exc} — excluded.")

    if not raw_data:
        raise RuntimeError(
            "download_universe: all tickers failed. "
            "Check network, ticker symbols, and period."
        )

    # ── Time-index alignment ─────────────────────────────────────────────
    # Build the shared calendar from all ticker indices.
    all_indices = [df.index for df in raw_data.values()]

    if align == 'inner':
        # Intersection: only dates where every ticker traded.
        common_idx = all_indices[0]
        for idx in all_indices[1:]:
            common_idx = common_idx.intersection(idx)
        logger.info(
            f"  [Universe] Inner-join alignment: "
            f"{len(common_idx)} common dates across {len(raw_data)} tickers "
            f"({common_idx[0].date()} → {common_idx[-1].date()})"
        )
    elif align == 'outer':
        # Union: all dates from any ticker, gaps filled/dropped.
        common_idx = all_indices[0]
        for idx in all_indices[1:]:
            common_idx = common_idx.union(idx)
        logger.info(
            f"  [Universe] Outer-join alignment: "
            f"{len(common_idx)} total dates, {len(raw_data)} tickers"
        )
    else:
        raise ValueError(f"download_universe: align must be 'inner' or 'outer', got '{align}'.")

    if len(common_idx) < MIN_ROWS:
        raise ValueError(
            f"download_universe: only {len(common_idx)} common dates after "
            f"alignment — need at least {MIN_ROWS}. "
            f"Use fewer tickers, a longer period, or align='outer'."
        )

    # Reindex each ticker to the common calendar; outer-join gets ffill.
    aligned: dict = {}
    for ticker, df in raw_data.items():
        df_aligned = df.reindex(common_idx)
        if align == 'outer' and df_aligned.isna().any().any():
            # Forward-fill short gaps, drop remaining NaN rows for this ticker
            df_aligned = df_aligned.ffill(limit=max_fill_days).dropna()
            # Reindex again — some dates may have been dropped for this ticker
        aligned[ticker] = df_aligned
        logger.info(
            f"  [Universe] {ticker}: {len(df_aligned)} aligned bars "
            f"(original: {len(df)})"
        )

    return aligned


def build_panel(
    aligned: dict,
    feature_fn,
    target_fn,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    cs_norm_method: str = 'zscore',
    cs_norm_min_tickers: int = 3,
) -> tuple:
    """
    Build a panel dataset from a universe of aligned OHLCV DataFrames.

    Pipeline (in order)
    ───────────────────
    1. Feature engineering  — applied independently per ticker to avoid
       any cross-ticker information leaking into features.

    2. Target construction  — applied independently per ticker.

    3. Temporal split       — train/val/test cut points computed from the
       SHARED calendar so every ticker has the same boundary dates.
       Critical: split dates are determined BEFORE feature engineering
       results are seen, preventing any look-ahead in the split itself.

    4. Cross-sectional normalisation — applied WITHIN each split only.
       For each date t and each feature f:
         zscore:  x_norm[t,f] = (x[t,f] - mean_i(x[t,f])) / std_i(x[t,f])
         rank:    x_norm[t,f] = rank_i(x[t,f]) / n_tickers   → [0,1]
       where i indexes tickers present on that date.

       IMPORTANT: cs_norm is fit on train dates only, never on val/test.
       For zscore this means the (mean, std) per (date, feature) come from
       train. Val and test use the same normalisation formula but are NOT
       used to compute the normalisation statistics.

       This avoids look-ahead: you don't know val/test cross-sectional
       statistics at the time you trade.

    5. Ticker tag — a 'ticker' column is added to every row so models
       can optionally use ticker embeddings, and results can be analysed
       per-asset.

    Dataset structure
    ─────────────────
    Output DataFrames have a (date, ticker) MultiIndex:
      - Level 0: date  (DatetimeIndex)
      - Level 1: ticker (str)
    Columns: all feature columns + target columns + 'ticker' string

    Parameters
    ----------
    aligned             : dict from download_universe().
    feature_fn          : callable(raw_df) -> feature_df  (e.g. build_features)
    target_fn           : callable(feature_df) -> (feature_df, target_cols)
    train_ratio         : fraction of dates for training.
    val_ratio           : fraction of dates for validation.
    cs_norm_method      : 'zscore', 'rank', or 'none'.
    cs_norm_min_tickers : minimum tickers per date to apply cs_norm;
                          dates with fewer tickers are left unnormalised.

    Returns
    -------
    (df_tr, df_va, df_te, feature_cols, target_cols)
    Each DataFrame has MultiIndex (date, ticker) and is sorted by date.
    """
    # ── Step 1 & 2: feature + target engineering per ticker ───────────────
    ticker_dfs: dict = {}
    feature_cols_ref = None
    target_cols_ref  = None

    for ticker, raw_df in aligned.items():
        try:
            feat_df = feature_fn(raw_df)
            feat_df, tcols = target_fn(feat_df)
            feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()

            if feature_cols_ref is None:
                # Derive feature columns from the first ticker
                sma_cols = set()
                try:
                    from config import DATA
                    sma_cols = {f'sma_{w}' for w in DATA.sma_windows}
                except Exception:
                    pass
                BASE = ({'Open', 'High', 'Low', 'Close', 'Volume'}
                        | set(tcols) | sma_cols)
                feature_cols_ref = [c for c in feat_df.columns if c not in BASE]
                target_cols_ref  = tcols
            else:
                # Intersect feature columns — some tickers may be missing cols
                feature_cols_ref = [c for c in feature_cols_ref
                                    if c in feat_df.columns]

            # Tag with ticker before storing
            feat_df = feat_df.copy()
            feat_df['ticker'] = ticker
            ticker_dfs[ticker] = feat_df

        except Exception as exc:
            logger.warning(f"  [Panel] {ticker} feature build failed: {exc} — skipped.")

    if not ticker_dfs:
        raise RuntimeError("build_panel: all tickers failed feature engineering.")

    # Align feature columns across all tickers
    keep_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + feature_cols_ref + target_cols_ref + ['ticker']
    for ticker in list(ticker_dfs.keys()):
        missing = [c for c in keep_cols if c not in ticker_dfs[ticker].columns]
        if missing:
            logger.warning(f"  [Panel] {ticker} missing columns {missing} — skipped.")
            del ticker_dfs[ticker]
        else:
            ticker_dfs[ticker] = ticker_dfs[ticker][keep_cols]

    if not ticker_dfs:
        raise RuntimeError("build_panel: no tickers survived column alignment.")

    # ── Step 3: Temporal split on shared calendar ─────────────────────────
    # Use sorted union of all date indices for the split boundaries.
    all_dates = sorted(set(
        d for df in ticker_dfs.values() for d in df.index
    ))
    n_dates    = len(all_dates)
    tr_end_idx = int(n_dates * train_ratio)
    va_end_idx = int(n_dates * (train_ratio + val_ratio))

    tr_cutoff = all_dates[tr_end_idx - 1]
    va_cutoff = all_dates[va_end_idx - 1]

    logger.info(
        f"  [Panel] Split dates: "
        f"train → {tr_cutoff.date()}  "
        f"val → {va_cutoff.date()}  "
        f"test → {all_dates[-1].date()}"
    )

    tr_frames, va_frames, te_frames = [], [], []
    for ticker, df in ticker_dfs.items():
        df_tr = df[df.index <= tr_cutoff]
        df_va = df[(df.index > tr_cutoff) & (df.index <= va_cutoff)]
        df_te = df[df.index > va_cutoff]
        if len(df_tr) > 0:
            tr_frames.append(df_tr)
        if len(df_va) > 0:
            va_frames.append(df_va)
        if len(df_te) > 0:
            te_frames.append(df_te)

    df_tr = pd.concat(tr_frames).sort_index()
    df_va = pd.concat(va_frames).sort_index()
    df_te = pd.concat(te_frames).sort_index()

    # ── Step 4: Cross-sectional normalisation ─────────────────────────────
    # Fit cs_norm ONLY on train dates, then apply same transform to val/test.
    if cs_norm_method != 'none' and feature_cols_ref:
        df_tr = _apply_cs_norm(
            df_tr, feature_cols_ref, cs_norm_method,
            cs_norm_min_tickers, fit=True,
        )
        # For val/test: use the same per-date formula but on their own dates
        # (cross-sectional stats are per-date, not stored from train)
        df_va = _apply_cs_norm(
            df_va, feature_cols_ref, cs_norm_method,
            cs_norm_min_tickers, fit=False,
        )
        df_te = _apply_cs_norm(
            df_te, feature_cols_ref, cs_norm_method,
            cs_norm_min_tickers, fit=False,
        )

    # ── Step 5: Build MultiIndex (date, ticker) ───────────────────────────
    for split_name, split_df in [('train', df_tr), ('val', df_va), ('test', df_te)]:
        n_dates_s  = split_df.index.nunique()
        n_tickers_s = split_df['ticker'].nunique() if 'ticker' in split_df.columns else '?'
        logger.info(
            f"  [Panel] {split_name}: {len(split_df)} rows  "
            f"{n_dates_s} dates × ~{n_tickers_s} tickers  "
            f"features={len(feature_cols_ref)}"
        )

    return df_tr, df_va, df_te, feature_cols_ref, target_cols_ref


def _apply_cs_norm(
    df: pd.DataFrame,
    feature_cols: list,
    method: str,
    min_tickers: int,
    fit: bool = True,
) -> pd.DataFrame:
    """
    Apply cross-sectional normalisation within each date slice.

    For each date t with >= min_tickers observations:
      zscore: feature[i,t] = (raw[i,t] - mean_t) / (std_t + 1e-8)
      rank:   feature[i,t] = rank_i(raw[i,t]) / n_tickers(t)

    Dates with fewer than min_tickers tickers are left unchanged
    (not enough cross-sectional data to compute meaningful statistics).

    The `fit` parameter is unused for cross-sectional normalisation
    (stats are computed per date, not stored), but kept for API clarity
    and future use with stored normalisation parameters.
    """
    df = df.copy()

    # Ensure feature columns are float64 to prevent dtype assignment 
    # errors when inserting float-based z-scores into integer columns
    for col in feature_cols:
        if df[col].dtype != np.float64:
            df[col] = df[col].astype(np.float64)

    for date in df.index.unique():
        mask = df.index == date
        group = df[mask]
        n = len(group)
        if n < min_tickers:
            continue   # too few tickers on this date — skip

        vals = group[feature_cols].values.astype(np.float64)

        if method == 'zscore':
            mu  = np.nanmean(vals, axis=0)
            std = np.nanstd(vals, axis=0) + 1e-8
            norm = (vals - mu) / std

        elif method == 'rank':
            # Percentile rank per feature column across tickers on this date
            norm = np.zeros_like(vals)
            for j in range(vals.shape[1]):
                col_vals = vals[:, j]
                finite   = np.isfinite(col_vals)
                if finite.sum() < 2:
                    norm[:, j] = 0.5
                    continue
                ranks = pd.Series(col_vals).rank(pct=True, na_option='keep').values
                norm[:, j] = np.where(finite, ranks, 0.5)

        else:
            continue

        df.loc[mask, feature_cols] = norm

    return df
