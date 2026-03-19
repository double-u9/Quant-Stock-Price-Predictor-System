"""
logger.py  —  V1 Structured Logging & Monitoring Framework.

Replaces the ad-hoc logging.basicConfig() in main.py with a
centralised, feature-rich logging system.

Problems fixed from V6/V7 main.py
───────────────────────────────────
 1. FORMAT: format='%(message)s' had no timestamp or severity.
    Fixed: rich format includes timestamp, level, and module name.

 2. FORMAT: No machine-readable output for metric parsing.
    Fixed: optional JSON-lines mode; every metric event is a keyed dict.

 3. FORMAT: No run ID — logs from parallel runs were indistinguishable.
    Fixed: every message carries the run_id in structured mode.

 4. SETUP: basicConfig() fired at import time in main.py.
    Fixed: setup_logging() is an explicit function called once at startup.

 5. SETUP: No log rotation / size cap.
    Fixed: RotatingFileHandler with configurable max_bytes and backups.

 6. SETUP: No dedicated ERROR sink.
    Fixed: separate error log file captures WARNING+ events only.

 7. SETUP: Duplicate handler risk from third-party libs.
    Fixed: all existing handlers are cleared before new ones are attached.

 8. COVERAGE: Unhandled exceptions had no run context.
    Fixed: sys.excepthook replaced with log_exception(), which records
    the full traceback alongside the run_id and current step.

 9. COVERAGE: No step timing.
    Fixed: StepTimer context manager logs elapsed time on exit.

10. COVERAGE: No system/environment snapshot at startup.
    Fixed: log_environment() captures Python, platform, GPU, key packages.

11. COVERAGE: No config snapshot in logs.
    Fixed: log_config() serialises any dataclass or dict to the log.

12. COVERAGE: Training metrics logged as freeform strings.
    Fixed: log_epoch() emits a structured JSON event for each epoch.

13. MONITORING: Evaluation metrics not individually keyed.
    Fixed: log_metrics() emits each metric as a separate key=value line,
    optionally also as a JSON event.

14. MONITORING: No warning/error counter in the run summary.
    Fixed: CountingHandler tallies warnings and errors; summary printed
    at shutdown via log_run_summary().

15. MONITORING: No safe_call() for graceful failure handling.
    Fixed: safe_call() wraps any callable, logs exceptions with full
    context, and returns a default value so the pipeline continues.
"""
from __future__ import annotations

import contextlib
import json
import logging
import logging.handlers
import os
import platform
import sys
import time
import traceback
import uuid
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


# ════════════════════════════════════════════════════════════════
#  CONSTANTS & FORMATS
# ════════════════════════════════════════════════════════════════

# Human-readable format for console and plain-text file output.
_PLAIN_FMT = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s"
_DATE_FMT  = "%Y-%m-%d %H:%M:%S"

# JSON-lines format prefix — the rest of the line is the JSON payload.
_JSON_PREFIX = "JSON_EVENT: "

# Default rotating log limits
_DEFAULT_MAX_BYTES  = 10 * 1024 * 1024   # 10 MB per log file
_DEFAULT_BACKUP_CNT = 5                   # keep 5 rotated files


# ════════════════════════════════════════════════════════════════
#  WARNING / ERROR COUNTER
# ════════════════════════════════════════════════════════════════

class CountingHandler(logging.Handler):
    """
    Tallies warnings and errors during a run without emitting any output.

    Attached to the root logger alongside the normal handlers so that
    log_run_summary() can report the total count at the end of each run.
    This makes silent data-quality warnings visible even when the console
    has scrolled past them.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.counts: Dict[str, int] = {
            "WARNING":  0,
            "ERROR":    0,
            "CRITICAL": 0,
        }

    def emit(self, record: logging.LogRecord) -> None:
        level = record.levelname
        if level in self.counts:
            self.counts[level] += 1

    def total_problems(self) -> int:
        return sum(self.counts.values())

    def reset(self) -> None:
        for k in self.counts:
            self.counts[k] = 0


# Module-level counter — populated by setup_logging(), read by summary.
_counter: Optional[CountingHandler] = None

# Current run ID — set by setup_logging(), included in structured events.
_run_id: str = "unknown"

# Current pipeline step — updated by StepTimer, included in exceptions.
_current_step: str = "init"


# ════════════════════════════════════════════════════════════════
#  SETUP
# ════════════════════════════════════════════════════════════════

def setup_logging(
    log_dir: str,
    run_id: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int    = logging.DEBUG,
    max_bytes: int     = _DEFAULT_MAX_BYTES,
    backup_count: int  = _DEFAULT_BACKUP_CNT,
    json_events: bool  = False,
) -> logging.Logger:
    """
    Configure the root logger for the entire pipeline.

    Call this ONCE at the start of main() — not at import time.
    All other modules use logging.getLogger(__name__) as usual; they
    inherit the handlers configured here automatically.

    Handlers attached
    ─────────────────
    1. StreamHandler (stdout)  — console output, INFO+ by default.
    2. RotatingFileHandler     — full log with rotation, DEBUG+ by default.
    3. RotatingFileHandler     — error-only log (WARNING+), for alerting.
    4. CountingHandler         — silent tally of warnings/errors.

    Parameters
    ----------
    log_dir       : directory where log files are written.
    run_id        : experiment ID string (auto-generated UUID4 if None).
    console_level : minimum level shown on console (default INFO).
    file_level    : minimum level written to full log file (default DEBUG).
    max_bytes     : rotate the log file after this many bytes.
    backup_count  : number of rotated backups to keep.
    json_events   : if True, structured metric events are emitted as
                    JSON-lines alongside plain-text messages.

    Returns
    -------
    logging.Logger  the configured root logger.
    """
    global _run_id, _counter

    os.makedirs(log_dir, exist_ok=True)

    _run_id = run_id or str(uuid.uuid4())[:8]

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"run_{ts}.log")
    errfile = os.path.join(log_dir, f"run_{ts}_errors.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # root captures everything; handlers filter

    # ── Clear any existing handlers to prevent duplicates ────────────────
    # Third-party libraries (yfinance, sklearn, optuna) attach handlers to
    # the root logger. Without this, every message is emitted multiple times.
    root.handlers.clear()

    formatter = logging.Formatter(_PLAIN_FMT, datefmt=_DATE_FMT)

    # ── Handler 1: Console (stdout) ───────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # ── Handler 2: Full rotating log file ────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        logfile,
        maxBytes    = max_bytes,
        backupCount = backup_count,
        encoding    = "utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # ── Handler 3: Error-only log (WARNING+) ─────────────────────────────
    # Separate file makes it easy to grep for problems without wading
    # through INFO messages.
    err_handler = logging.handlers.RotatingFileHandler(
        errfile,
        maxBytes    = max_bytes,
        backupCount = backup_count,
        encoding    = "utf-8",
    )
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(formatter)
    root.addHandler(err_handler)

    # ── Handler 4: Silent warning/error counter ───────────────────────────
    _counter = CountingHandler()
    root.addHandler(_counter)

    # ── Replace sys.excepthook for unhandled exception capture ───────────
    def _excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logging.getLogger("pipeline.exception").critical(
            f"[run={_run_id}] [step={_current_step}] "
            f"UNHANDLED EXCEPTION:\n{tb_str}"
        )

    sys.excepthook = _excepthook

    logger = logging.getLogger("pipeline")
    logger.info(
        f"[run={_run_id}] Logging initialised  "
        f"logfile={logfile}  errfile={errfile}  "
        f"json_events={json_events}"
    )

    # Store json_events flag for use by log_epoch / log_metrics
    setup_logging._json_events = json_events
    setup_logging._run_id      = _run_id

    return logger


# ════════════════════════════════════════════════════════════════
#  ENVIRONMENT SNAPSHOT
# ════════════════════════════════════════════════════════════════

def log_environment(logger: logging.Logger) -> None:
    """
    Log system and library versions at run startup.

    Captures Python version, OS, CPU/GPU, and key package versions
    so that environment-specific bugs can be reproduced exactly.
    A run that worked with sklearn 1.3 but fails on 1.4 is immediately
    diagnosable when the version is in the log.
    """
    import importlib

    python_ver  = sys.version.replace("\n", " ")
    platform_str = platform.platform()

    # ── GPU / CUDA detection ─────────────────────────────────────────────
    gpu_info = "CPU only"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = (
                f"CUDA {torch.version.cuda}  "
                f"GPU={torch.cuda.get_device_name(0)}  "
                f"VRAM={torch.cuda.get_device_properties(0).total_memory // 1024**2}MB"
            )
    except ImportError:
        pass

    # ── Memory ───────────────────────────────────────────────────────────
    ram_info = "unknown"
    try:
        import psutil
        ram_gb   = psutil.virtual_memory().total / 1024 ** 3
        ram_info = f"{ram_gb:.1f} GB total"
    except ImportError:
        pass

    # ── Key package versions ──────────────────────────────────────────────
    packages = ["numpy", "pandas", "sklearn", "torch", "xgboost",
                "scipy", "optuna", "yfinance"]
    versions = {}
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            versions[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            versions[pkg] = "not installed"

    logger.info(
        f"[run={_run_id}] ENVIRONMENT\n"
        f"  Python  : {python_ver}\n"
        f"  Platform: {platform_str}\n"
        f"  GPU     : {gpu_info}\n"
        f"  RAM     : {ram_info}\n"
        f"  Packages: "
        + "  ".join(f"{k}={v}" for k, v in versions.items())
    )


# ════════════════════════════════════════════════════════════════
#  CONFIGURATION SNAPSHOT
# ════════════════════════════════════════════════════════════════

def log_config(logger: logging.Logger,
               *configs: Any,
               label: str = "CONFIG") -> None:
    """
    Serialise one or more config objects (dataclasses or dicts) to the log.

    Without this, the log contains model performance but not the settings
    that produced it. With this, a run is fully self-describing: you can
    reconstruct the exact experiment from the log alone.

    Parameters
    ----------
    *configs : any number of dataclass instances or plain dicts.
    label    : prefix for the log line (default "CONFIG").
    """
    merged: Dict[str, Any] = {}
    for cfg in configs:
        if is_dataclass(cfg) and not isinstance(cfg, type):
            merged.update(asdict(cfg))
        elif isinstance(cfg, dict):
            merged.update(cfg)
        else:
            # Best-effort: try __dict__
            try:
                merged.update(vars(cfg))
            except TypeError:
                merged[str(type(cfg).__name__)] = str(cfg)

    # Format as aligned key=value pairs
    lines = "\n".join(
        f"    {k:<30} = {v}" for k, v in merged.items()
    )
    logger.info(f"[run={_run_id}] {label}\n{lines}")


# ════════════════════════════════════════════════════════════════
#  STEP TIMER
# ════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def StepTimer(step_name: str, logger: logging.Logger):
    """
    Context manager that logs the start and elapsed time of a pipeline step.

    Usage
    -----
        with StepTimer("STEP 1 / 8 -- Data", logger):
            raw = download(...)

    Elapsed time is logged on exit so the operator can immediately see
    which step is the bottleneck without diffing timestamps manually.
    Also updates _current_step so that any exception during the step
    is attributed to the correct stage in the error log.
    """
    global _current_step
    _current_step = step_name
    bar = "=" * 60
    logger.info(f"\n{bar}\n  {step_name}\n{bar}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info(
            f"  [run={_run_id}] {step_name} completed in {elapsed:.1f}s"
        )


# ════════════════════════════════════════════════════════════════
#  STRUCTURED METRIC EVENTS
# ════════════════════════════════════════════════════════════════

def _emit_json(logger: logging.Logger,
               event_type: str,
               payload: Dict[str, Any]) -> None:
    """
    Emit a JSON-lines event if json_events mode is enabled.

    JSON events are written at DEBUG level so they appear only in the
    full log file, not on the console. They can be parsed by log
    analysis tools (e.g. `grep JSON_EVENT run.log | jq .`) without
    affecting human readability.
    """
    if not getattr(setup_logging, "_json_events", False):
        return
    record = {
        "ts":       datetime.now(timezone.utc).isoformat(),
        "run_id":   _run_id,
        "step":     _current_step,
        "event":    event_type,
        **payload,
    }
    logger.debug(f"{_JSON_PREFIX}{json.dumps(record)}")


def log_epoch(logger: logging.Logger,
              model_name: str,
              epoch: int,
              total_epochs: int,
              train_loss: float,
              val_loss: float,
              lr: float,
              grad_norm: float,
              is_best: bool = False,
              extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Log one training epoch with both a human-readable line and a
    structured JSON event.

    The structured event makes it possible to reconstruct training
    curves from the log file without regex parsing — just grep for
    JSON_EVENT lines with event='epoch'.

    Parameters
    ----------
    model_name   : e.g. 'lstm', 'gru'.
    epoch        : current epoch (1-based for display).
    total_epochs : maximum configured epochs.
    train_loss   : training loss for this epoch.
    val_loss     : validation loss for this epoch.
    lr           : current learning rate.
    grad_norm    : mean gradient norm for this epoch.
    is_best      : whether this epoch set a new validation best.
    extra        : any additional key-value pairs to include.
    """
    best_tag = " ★ best" if is_best else ""
    logger.info(
        f"  Epoch {epoch:3d}/{total_epochs}  "
        f"[{model_name}]  "
        f"train={train_loss:.4f}  val={val_loss:.4f}  "
        f"lr={lr:.2e}  grad={grad_norm:.3f}"
        f"{best_tag}"
    )
    payload = dict(
        model=model_name, epoch=epoch, total_epochs=total_epochs,
        train_loss=round(train_loss, 6), val_loss=round(val_loss, 6),
        lr=lr, grad_norm=round(grad_norm, 4), is_best=is_best,
    )
    if extra:
        payload.update(extra)
    _emit_json(logger, "epoch", payload)


def log_metrics(logger: logging.Logger,
                model_name: str,
                clf: Dict[str, Any],
                trd: Dict[str, Any],
                split: str = "test") -> None:
    """
    Log evaluation metrics as individually-keyed lines and structured events.

    Emitting each metric on a separate line (or as a JSON key) makes it
    possible to grep for a specific metric across all models in one pass:
        grep 'sharpe' run.log
    rather than manually scanning multi-line print_metrics() blocks.

    Parameters
    ----------
    model_name : model identifier.
    clf        : dict from classification_metrics().
    trd        : dict from trading_metrics().
    split      : 'val' or 'test' — recorded in the JSON event.
    """
    # ── Human-readable block ─────────────────────────────────────────────
    pf        = trd.get('profit_factor', 0)
    pf_str    = f"{pf:.2f}" if isinstance(pf, float) and pf != float('inf') else "inf"
    omega     = trd.get('omega', float('nan'))
    omega_str = f"{omega:.2f}" if isinstance(omega, float) and omega == omega else "nan"

    logger.info(
        f"\n  ── {model_name} [{split}] ──\n"
        f"  acc={clf['accuracy']*100:.2f}%  "
        f"auc={clf['auc']:.4f}  "
        f"f1_up={clf['f1_up']:.4f}  "
        f"f1_dn={clf['f1_down']:.4f}  "
        f"brier={clf['brier']:.4f}\n"
        f"  sharpe={trd['sharpe']:+.2f}  "
        f"sortino={trd['sortino']:+.2f}  "
        f"calmar={trd['calmar']:+.2f}  "
        f"omega={omega_str}  "
        f"ic={trd['ic']:+.4f}\n"
        f"  max_dd={trd['max_dd']*100:.1f}%  "
        f"dd_dur={trd.get('max_dd_duration', 0)}d  "
        f"var95={trd.get('var_95', 0)*100:.2f}%  "
        f"cvar95={trd.get('cvar_95', 0)*100:.2f}%\n"
        f"  ann_ret={trd['ann_ret']*100:+.1f}%  "
        f"alpha={trd.get('alpha', 0)*100:+.1f}%  "
        f"beta={trd.get('beta', 1):.2f}  "
        f"ir={trd.get('information_ratio', 0):+.2f}\n"
        f"  win_rate={trd['win_rate']*100:.1f}%  "
        f"pf={pf_str}  "
        f"n_trades={trd['n_trades']}  "
        f"turnover={trd.get('turnover', 0)*100:.1f}%/d  "
        f"avg_hold={trd.get('avg_hold', 0):.1f}bars  "
        f"confident={trd['pct_confident']*100:.0f}%"
    )
    logger.info(clf.get('report', ''))

    # ── Structured JSON event ────────────────────────────────────────────
    scalar_trd = {
        k: v for k, v in trd.items()
        if isinstance(v, (int, float)) and not callable(v)
    }
    _emit_json(logger, "metrics", dict(
        model=model_name,
        split=split,
        accuracy=round(clf['accuracy'], 6),
        auc=round(clf['auc'], 6),
        f1_up=round(clf['f1_up'], 6),
        f1_down=round(clf['f1_down'], 6),
        brier=round(clf['brier'], 6),
        **{k: round(v, 6) if isinstance(v, float) else v
           for k, v in scalar_trd.items()},
    ))


# ════════════════════════════════════════════════════════════════
#  EXCEPTION HANDLING
# ════════════════════════════════════════════════════════════════

def log_exception(logger: logging.Logger,
                  exc: Exception,
                  context: str = "",
                  reraise: bool = False) -> None:
    """
    Log an exception with run context, then optionally re-raise.

    Unlike a bare except+print, this captures:
      - the run ID for correlation across distributed logs
      - the current pipeline step where the error occurred
      - the full traceback formatted for the log file

    Parameters
    ----------
    exc     : the caught exception.
    context : additional description (e.g. 'during LSTM training').
    reraise : if True, re-raise the exception after logging.
    """
    tb_str = traceback.format_exc()
    logger.error(
        f"[run={_run_id}] [step={_current_step}] "
        f"EXCEPTION{f' ({context})' if context else ''}: "
        f"{type(exc).__name__}: {exc}\n{tb_str}"
    )
    _emit_json(logger, "exception", dict(
        context=context,
        exc_type=type(exc).__name__,
        exc_msg=str(exc),
    ))
    if reraise:
        raise exc


def safe_call(func: Callable[..., T],
              *args: Any,
              logger: logging.Logger,
              context: str = "",
              default: Any = None,
              reraise: bool = False,
              **kwargs: Any) -> T:
    """
    Call a function, logging any exception without crashing the pipeline.

    This is the key mechanism for resilient pipeline execution. If a
    visualisation chart fails or a single model crashes during evaluation,
    the failure is logged with full context and the pipeline continues
    with the remaining models/charts rather than aborting entirely.

    Parameters
    ----------
    func    : callable to invoke.
    *args   : positional arguments for func.
    logger  : logger to use for error reporting.
    context : description for error messages (e.g. 'chart_calibration').
    default : value to return if func raises (default None).
    reraise : if True, re-raise after logging (defeats the purpose, but
              useful for debugging).
    **kwargs: keyword arguments for func.

    Returns
    -------
    The return value of func, or default if func raised.

    Example
    -------
        result = safe_call(
            viz.chart_calibration, y_true, all_probs,
            logger=logger, context="chart_calibration",
            default=None,
        )
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        log_exception(logger, exc, context=context or func.__name__,
                      reraise=reraise)
        return default


# ════════════════════════════════════════════════════════════════
#  RUN SUMMARY
# ════════════════════════════════════════════════════════════════

def log_run_summary(logger: logging.Logger,
                    run_start_time: float,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a structured summary at the end of a pipeline run.

    Includes:
    - total elapsed time
    - warning and error counts (so a silent flood of warnings is visible)
    - any additional key-value pairs from the caller (e.g. best model name)

    Call this as the very last line of main() so the summary is always
    written even if earlier steps encountered errors.

    Parameters
    ----------
    run_start_time : value of time.perf_counter() at the start of main().
    extra          : additional key-value pairs to include.
    """
    elapsed = time.perf_counter() - run_start_time
    mins, secs = divmod(int(elapsed), 60)

    counts = _counter.counts if _counter else {}
    n_warn = counts.get("WARNING", 0)
    n_err  = counts.get("ERROR", 0)
    n_crit = counts.get("CRITICAL", 0)

    health = "OK" if (n_err + n_crit) == 0 else "DEGRADED"
    if n_crit > 0:
        health = "FAILED"

    summary_lines = [
        f"\n{'=' * 60}",
        f"  RUN SUMMARY  [run={_run_id}]",
        f"{'=' * 60}",
        f"  Status   : {health}",
        f"  Elapsed  : {mins}m {secs}s",
        f"  Warnings : {n_warn}",
        f"  Errors   : {n_err}",
        f"  Critical : {n_crit}",
    ]
    if extra:
        for k, v in extra.items():
            summary_lines.append(f"  {k:<10}: {v}")
    summary_lines.append("=" * 60)

    log_fn = logger.error if health != "OK" else logger.info
    log_fn("\n".join(summary_lines))

    _emit_json(logger, "run_summary", dict(
        health=health, elapsed_s=round(elapsed, 1),
        n_warnings=n_warn, n_errors=n_err, n_critical=n_crit,
        **(extra or {}),
    ))

    if _counter:
        _counter.reset()
