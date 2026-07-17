"""Reusable BiLSTM rainfall workflow used by the notebook and command line.

The module owns preprocessing, sequence construction, scaling, model training,
evaluation, and artifact writing so the notebook can stay focused on analysis.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import logging
import os
import platform
import random
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib_cache")))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 0. Imports, constants, and global settings
# =============================================================================

DEFAULT_DATA_PATH = Path(__file__).with_name("sample_weather_data.csv")
DEFAULT_DATE_COL = "DATA TIMESTAMP"
DEFAULT_TARGET_COL = "RAINFALL 24H MM"
DEFAULT_EPOCHS = 100
DEFAULT_LR_DROP_FACTOR = 0.1
DEFAULT_OPTIMIZER = "adam"
DEFAULT_LOSS_FUNCTION = "mse"
DEFAULT_BILSTM_LAYERS = 3
SUPPORTED_OPTIMIZERS = {"adam", "rmsprop", "sgd"}
SUPPORTED_LOSS_FUNCTIONS = {"mse", "mae", "huber"}
NORMALIZED_MAAPE_PERCENT_SCALE = 100.0 / (np.pi / 2.0)
METRIC_DISPLAY_DECIMALS = 4
NORMALIZED_DISPLAY_DECIMALS = 4
SOURCE_DECIMAL_PLACES = {
    "TEMPERATURE AVG C": 3,
    "RAINFALL 24H MM": 1,
    "SUNSHINE 24H H": 1,
    "REL HUMIDITY AVG PC": 2,
    "WIND SPEED 24H MEAN MS": 1,
}
DEFAULT_KEEP_RUNS = 1
DEFAULT_CPU_THREADS = -1


# =============================================================================
# 1. Runtime bootstrap and warning handling
# =============================================================================

def configure_warning_filters() -> None:
    """Hide known non-critical runtime noise while keeping real errors visible."""
    warnings.filterwarnings(
        "ignore",
        message=".*oneDNN custom operations are on.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*__array__ implementation doesn't accept a copy keyword.*",
        category=DeprecationWarning,
        module=r"keras\.src\.backend\.tensorflow\.core",
    )


configure_warning_filters()
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


@contextlib.contextmanager
def suppress_native_stderr(enabled: bool = True):
    if not enabled:
        yield
        return

    try:
        stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                yield
            finally:
                os.dup2(saved_stderr_fd, stderr_fd)
                os.close(saved_stderr_fd)
    except (AttributeError, OSError):
        yield


# =============================================================================
# 2. CLI configuration and validation
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Build the command-line interface for the complete BiLSTM workflow."""
    parser = argparse.ArgumentParser(
        description=(
            "Rainfall prediction with BiLSTM, 7-day lag, Normalized MAAPE (%), "
            "and units/batch size/learning-rate drop grid search."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--date-col", default=DEFAULT_DATE_COL)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--lag", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument(
        "--units",
        nargs="+",
        type=int,
        default=[32, 64, 128],
        help="Number of LSTM units inside each stacked BiLSTM layer.",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument(
        "--bilstm-layers",
        type=int,
        default=DEFAULT_BILSTM_LAYERS,
        help=(
            "Number of stacked Bidirectional LSTM layers. This is configurable "
            "but not part of the grid search."
        ),
    )
    parser.add_argument(
        "--lr-drop-periods",
        nargs="+",
        type=int,
        default=[10, 20, 25],
        help=(
            "Learning-rate drop period. Example: 10 means the learning rate "
            "is reduced every 10 epochs."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=(
            "Training epochs for every grid-search model. This is configurable "
            "but not part of the grid."
        ),
    )
    parser.add_argument(
        "--lr-drop-factor",
        type=float,
        default=DEFAULT_LR_DROP_FACTOR,
        help=(
            "Learning-rate multiplier applied at each configured drop period. "
            "This is configurable but not part of the grid."
        ),
    )
    parser.add_argument(
        "--initial-learning-rate",
        type=float,
        default=None,
        help=(
            "Initial optimizer learning rate. Leave unset to use the selected "
            "Keras optimizer default."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=sorted(SUPPORTED_OPTIMIZERS),
        default=DEFAULT_OPTIMIZER,
        help="Keras optimizer used during training.",
    )
    parser.add_argument(
        "--loss-function",
        choices=sorted(SUPPORTED_LOSS_FUNCTIONS),
        default=DEFAULT_LOSS_FUNCTION,
        help="Keras loss function used during training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help=(
            "TensorFlow CPU thread count. Use -1 to use every logical CPU "
            "thread, or a positive integer to set an explicit limit."
        ),
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--keep-runs",
        type=int,
        default=DEFAULT_KEEP_RUNS,
        help="Number of latest completed output run folders to keep.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show the actual vs predicted plot for the best model after training.",
    )
    parser.add_argument(
        "--zero-codes",
        "--missing-codes",
        dest="zero_codes",
        nargs="+",
        type=float,
        default=[8888.0, 9999.0],
        help=(
            "Special missing/unmeasured codes. In X variables these codes are "
            "treated as missing values and linearly interpolated; in the "
            "rainfall target they are replaced with 0."
        ),
    )
    parser.add_argument(
        "--no-target-history",
        dest="include_target_history",
        action="store_false",
        help=(
            "Use only X variables except rainfall in the previous 7 days. "
            "By default, historical rainfall is also included as a lag feature."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Check preprocessing and data shapes without running TensorFlow.",
    )
    parser.set_defaults(include_target_history=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.lag < 1:
        raise ValueError("--lag must be at least 1.")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if any(n < 1 for n in args.units):
        raise ValueError("--units must contain positive values.")
    if any(b < 1 for b in args.batch_sizes):
        raise ValueError("--batch-sizes must contain positive values.")
    if getattr(args, "bilstm_layers", DEFAULT_BILSTM_LAYERS) < 1:
        raise ValueError("--bilstm-layers must be at least 1.")
    if any(period < 1 for period in args.lr_drop_periods):
        raise ValueError("--lr-drop-periods must contain positive values.")
    if getattr(args, "epochs", DEFAULT_EPOCHS) < 1:
        raise ValueError("--epochs must be at least 1.")
    if not 0 < getattr(args, "lr_drop_factor", DEFAULT_LR_DROP_FACTOR) <= 1:
        raise ValueError("--lr-drop-factor must be greater than 0 and less than or equal to 1.")
    initial_lr = getattr(args, "initial_learning_rate", None)
    if initial_lr is not None and initial_lr <= 0:
        raise ValueError("--initial-learning-rate must be positive when set.")
    if getattr(args, "optimizer", DEFAULT_OPTIMIZER) not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"--optimizer must be one of: {sorted(SUPPORTED_OPTIMIZERS)}.")
    if getattr(args, "loss_function", DEFAULT_LOSS_FUNCTION) not in SUPPORTED_LOSS_FUNCTIONS:
        raise ValueError(f"--loss-function must be one of: {sorted(SUPPORTED_LOSS_FUNCTIONS)}.")
    if getattr(args, "cpu_threads", DEFAULT_CPU_THREADS) == 0 or getattr(
        args, "cpu_threads", DEFAULT_CPU_THREADS
    ) < -1:
        raise ValueError("--cpu-threads must be -1 or a positive integer.")
    if args.keep_runs < 1:
        raise ValueError("--keep-runs must be at least 1.")


# =============================================================================
# 3. Reproducibility and CPU runtime helpers
# =============================================================================

def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def query_cpu_info() -> dict[str, str]:
    processor = platform.processor().strip()
    machine = platform.machine().strip()
    system = platform.system().strip()

    generic_processor = (
        not processor
        or "family" in processor.lower()
        or processor.upper() in {"AMD64", "X86_64"}
    )
    if generic_processor and system.lower() == "windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                registry_processor, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            processor = str(registry_processor).strip()
        except (ImportError, FileNotFoundError, OSError):
            processor = ""

    generic_processor = (
        not processor
        or "family" in processor.lower()
        or processor.upper() in {"AMD64", "X86_64"}
    )
    if generic_processor and system.lower() == "windows":
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            processor = completed.stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            processor = ""

    return {
        "processor": processor or "unknown",
        "machine": machine or "unknown",
        "system": system or "unknown",
    }


def format_tensorflow_cpu_report(cpu_runtime: dict[str, object]) -> str:
    cpu_info = cpu_runtime.get("cpu_info", {})
    lines = [
        "TensorFlow CPU runtime:",
        f"- processor: {cpu_info.get('processor', 'unknown')}",
        f"- machine: {cpu_info.get('machine', 'unknown')}",
        f"- system: {cpu_info.get('system', 'unknown')}",
        f"- logical_threads_used: {cpu_runtime.get('cpu_threads', 'auto')}",
        f"- inter_op_threads: {cpu_runtime.get('inter_op_threads', 'auto')}",
        f"- oneDNN_enabled: {cpu_runtime.get('onednn_enabled', True)}",
    ]
    return "\n".join(lines)


def configure_tensorflow_cpu(
    seed: int,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    print_report: bool = True,
) -> tuple[object, dict[str, object]]:
    """Configure deterministic TensorFlow execution on CPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    resolved_cpu_threads = (
        max(1, os.cpu_count() or 1)
        if int(cpu_threads) == -1
        else max(1, int(cpu_threads))
    )
    inter_op_threads = max(1, min(4, resolved_cpu_threads))
    # Configure TensorFlow thread pools before the runtime is imported.
    os.environ["OMP_NUM_THREADS"] = str(resolved_cpu_threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(resolved_cpu_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(inter_op_threads)

    try:
        with suppress_native_stderr():
            import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        tf.config.threading.set_intra_op_parallelism_threads(resolved_cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    except RuntimeError:
        pass

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        print(
            "TensorFlow was already initialized. Restart the notebook kernel "
            "before running all cells to apply the CPU-only configuration cleanly."
        )

    set_reproducible_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    cpu_runtime = {
        "runtime": "cpu",
        "cpu_info": query_cpu_info(),
        "cpu_threads": resolved_cpu_threads,
        "inter_op_threads": inter_op_threads,
        "onednn_enabled": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "1") == "1",
    }
    if print_report:
        print(format_tensorflow_cpu_report(cpu_runtime))
    return tf, cpu_runtime


# =============================================================================
# 4. Data loading, audit, and preprocessing
# =============================================================================

def load_and_prepare_dataframe(
    data_path: Path,
    date_col: str,
    target_col: str,
    zero_codes: list[float],
    include_target_history: bool,
) -> tuple[pd.DataFrame, dict[str, object], list[str], list[str]]:
    """Load weather data and apply the rainfall-specific cleaning rules.

    Input-feature special codes are interpolated as missing observations;
    rainfall special codes and missing values are treated as zero rainfall.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file was not found: {data_path}")

    df = pd.read_csv(data_path)
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' was not found in the CSV.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the CSV.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        bad_rows = int(df[date_col].isna().sum())
        raise ValueError(f"There are {bad_rows} rows with invalid dates.")

    df = df.sort_values(date_col).drop_duplicates(subset=[date_col])
    df = df.set_index(date_col)
    full_daily_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="D",
        name=date_col,
    )
    missing_daily_rows_inserted = int(len(full_daily_index.difference(df.index)))
    df = df.reindex(full_daily_index)

    numeric_cols = list(df.columns)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    null_counts_before_fill = {
        col: int(df[col].isna().sum()) for col in numeric_cols
    }
    zero_code_counts = {
        str(code): int((df[numeric_cols] == code).sum().sum())
        for code in zero_codes
    }

    exogenous_cols = [col for col in numeric_cols if col != target_col]
    x_zero_code_counts = {
        str(code): int((df[exogenous_cols] == code).sum().sum())
        for code in zero_codes
    }
    y_zero_code_counts = {
        str(code): int((df[target_col] == code).sum())
        for code in zero_codes
    }
    if zero_codes:
        df[exogenous_cols] = df[exogenous_cols].replace(zero_codes, np.nan)
        df[target_col] = df[target_col].replace(zero_codes, 0.0)

    df_prepared = df[numeric_cols].copy()
    df_prepared[exogenous_cols] = df_prepared[exogenous_cols].interpolate(
        method="linear",
        limit_direction="both",
    )
    remaining_x_missing_after_interpolation = {
        col: int(df_prepared[col].isna().sum()) for col in exogenous_cols
    }
    unresolved_x_missing = {
        col: count
        for col, count in remaining_x_missing_after_interpolation.items()
        if count > 0
    }
    if unresolved_x_missing:
        raise ValueError(
            "Some X columns still contain missing values after linear interpolation. "
            "They were not filled with 0 because X special codes must be interpolated. "
            f"Unresolved missing counts: {unresolved_x_missing}"
        )
    df_prepared[target_col] = df_prepared[target_col].fillna(0.0)

    for col, decimal_places in SOURCE_DECIMAL_PLACES.items():
        if col in df_prepared.columns:
            df_prepared[col] = df_prepared[col].round(decimal_places)

    if df_prepared[target_col].isna().all():
        raise ValueError("The target column is still fully empty after preprocessing.")

    model_order_cols = exogenous_cols + [target_col]
    df_prepared = df_prepared[model_order_cols]

    if include_target_history:
        feature_cols = exogenous_cols + [target_col]
    else:
        feature_cols = exogenous_cols

    preprocessing_stats = {
        "missing_daily_rows_inserted": missing_daily_rows_inserted,
        "zero_code_counts_before_preprocessing": zero_code_counts,
        "x_zero_codes_treated_as_missing": x_zero_code_counts,
        "y_zero_codes_replaced_with_zero": y_zero_code_counts,
        "empty_values_before_fill": null_counts_before_fill,
        "x_empty_fill_method": "special codes and missing values use linear interpolation",
        "y_empty_fill_method": "special codes and missing values fill with 0",
        "remaining_x_empty_after_interpolation": remaining_x_missing_after_interpolation,
        "decimal_rounding_after_fill": SOURCE_DECIMAL_PLACES,
        "remaining_empty_values_after_fill": {
            col: int(df_prepared[col].isna().sum()) for col in numeric_cols
        },
    }

    return df_prepared, preprocessing_stats, feature_cols, exogenous_cols


# =============================================================================
# 5. Feature construction and model-data preparation
# =============================================================================

def make_lagged_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    feature_values = df[feature_cols].to_numpy(dtype=np.float32)
    target_values = df[target_col].to_numpy(dtype=np.float32)
    dates = df.index.to_numpy()

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    y_dates: list[np.datetime64] = []
    skipped_nan_input = 0

    for target_idx in range(lag, len(df)):
        x_seq = feature_values[target_idx - lag : target_idx]
        y_value = target_values[target_idx]

        if np.isnan(x_seq).any() or np.isnan(y_value):
            skipped_nan_input += 1
            continue

        x_rows.append(x_seq)
        y_rows.append(float(y_value))
        y_dates.append(dates[target_idx])

    if not x_rows:
        raise ValueError("No valid sequence was found after preprocessing.")

    stats = {
        "skipped_nan_input": skipped_nan_input,
    }
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32).reshape(-1, 1),
        np.asarray(y_dates),
        stats,
    )


def build_lagged_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lag: int,
) -> pd.DataFrame:
    feature_values = df[feature_cols].to_numpy(dtype=np.float32)
    target_values = df[target_col].to_numpy(dtype=np.float32)
    dates = df.index.to_numpy()

    rows: list[dict[str, object]] = []
    for target_idx in range(lag, len(df)):
        x_seq = feature_values[target_idx - lag : target_idx]
        y_value = target_values[target_idx]

        if np.isnan(x_seq).any() or np.isnan(y_value):
            continue

        row: dict[str, object] = {}
        for lag_offset in range(lag, 0, -1):
            input_idx = target_idx - lag_offset
            row[f"t_minus_{lag_offset}_date"] = pd.to_datetime(dates[input_idx])
            for feature_index, feature_col in enumerate(feature_cols):
                row[f"t_minus_{lag_offset}_{feature_col}"] = float(
                    feature_values[input_idx, feature_index]
                )

        row["target_date"] = pd.to_datetime(dates[target_idx])
        row["rainfall_target_mm"] = float(y_value)
        rows.append(row)

    if not rows:
        raise ValueError("No valid lagged rows were found after preprocessing.")
    return pd.DataFrame(rows)


def chronological_split(
    x_data: np.ndarray,
    y_data: np.ndarray,
    dates: np.ndarray,
    train_ratio: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_samples = len(x_data)
    train_end = int(n_samples * train_ratio)

    if train_end < 1 or train_end >= n_samples:
        raise ValueError(
            "There is not enough data for the train/test split. Change --train-ratio."
        )

    return {
        "train": (x_data[:train_end], y_data[:train_end], dates[:train_end]),
        "test": (x_data[train_end:], y_data[train_end:], dates[train_end:]),
    }


def fit_minmax_2d(values: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(values, dtype=np.float32)
    data_min = np.nanmin(arr, axis=0)
    data_max = np.nanmax(arr, axis=0)
    return {"min": data_min.tolist(), "max": data_max.tolist()}


def transform_minmax(values: np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    data_min = np.asarray(scaler["min"], dtype=np.float32)
    data_max = np.asarray(scaler["max"], dtype=np.float32)
    denom = np.where((data_max - data_min) == 0, 1.0, data_max - data_min)
    return (arr - data_min) / denom


def fit_target_scale(y_train: np.ndarray) -> float:
    max_abs = float(np.nanmax(np.abs(y_train)))
    return max_abs if max_abs > 0 else 1.0


def scale_split_data(
    split_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, object]]:
    x_train, y_train, _ = split_data["train"]
    n_features = x_train.shape[-1]

    x_scaler = fit_minmax_2d(x_train.reshape(-1, n_features))
    y_scale = fit_target_scale(y_train)

    scaled: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split_name, (x_part, y_part, date_part) in split_data.items():
        x_shape = x_part.shape
        x_scaled = transform_minmax(x_part.reshape(-1, n_features), x_scaler).reshape(
            x_shape
        )
        y_scaled = y_part / y_scale
        scaled[split_name] = (x_scaled.astype(np.float32), y_scaled.astype(np.float32), date_part)

    scalers = {"x_scaler": x_scaler, "y_scale": y_scale}
    return scaled, scalers


def inverse_target_scale(y_scaled: np.ndarray, y_scale: float) -> np.ndarray:
    return np.asarray(y_scaled, dtype=np.float32) * float(y_scale)


# =============================================================================
# 6. Metrics and result formatting
# =============================================================================

def maape_angle_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    return float(np.mean(np.arctan2(np.abs(y_true - y_pred), np.abs(y_true))))


def normalized_maape_percent_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return maape_angle_np(y_true, y_pred) * NORMALIZED_MAAPE_PERCENT_SCALE


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    error = y_true - y_pred
    return {
        "maape": maape_angle_np(y_true, y_pred),
        "normalized_maape_percent": normalized_maape_percent_np(y_true, y_pred),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def format_decimal(value: float, decimals: int = METRIC_DISPLAY_DECIMALS) -> str:
    """Format a displayed number with at most ``decimals`` decimal places."""
    rounded_value = round(float(value), decimals)
    if rounded_value == 0:
        rounded_value = 0.0
    return f"{rounded_value:.{decimals}f}".rstrip("0").rstrip(".")


def round_metric_columns(
    df: pd.DataFrame,
    decimals: int = METRIC_DISPLAY_DECIMALS,
) -> pd.DataFrame:
    rounded = df.copy()
    metric_cols = [
        col
        for col in rounded.columns
        if col.endswith("_maape_percent")
        or col.endswith("_maape")
        or col.endswith("_mae")
        or col.endswith("_rmse")
    ]
    if metric_cols:
        rounded[metric_cols] = rounded[metric_cols].round(decimals)
    return rounded


def build_ranked_results(rows: list[dict[str, object]]) -> pd.DataFrame:
    results_df = pd.DataFrame(rows).sort_values("test_normalized_maape_percent", ascending=True)
    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, "rank", np.arange(1, len(results_df) + 1))
    results_df.insert(1, "is_best", results_df["rank"].eq(1))
    return round_metric_columns(results_df)


def cleanup_old_output_runs(
    output_root: Path,
    current_output_dir: Path,
    keep_runs: int = DEFAULT_KEEP_RUNS,
) -> list[Path]:
    output_root = output_root.resolve()
    current_output_dir = current_output_dir.resolve()
    if not output_root.exists():
        return []

    run_dirs = [
        path
        for path in output_root.iterdir()
        if path.is_dir()
        and (
            path.name.startswith("bilstm_run_")
            or path.name.startswith("notebook_bilstm_run_")
        )
    ]
    completed_run_dirs = [
        path
        for path in run_dirs
        if (path / "hyperparameter_results.csv").exists()
        or (path / "best_test_predictions.csv").exists()
    ]

    completed_run_dirs = sorted(
        completed_run_dirs,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    keep = {current_output_dir}
    for path in completed_run_dirs:
        if len(keep) >= keep_runs:
            break
        keep.add(path.resolve())

    removed: list[Path] = []
    for path in run_dirs:
        resolved = path.resolve()
        if output_root not in resolved.parents:
            continue
        if resolved in keep:
            continue
        for child in sorted(resolved.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        resolved.rmdir()
        removed.append(path)
    return removed


# =============================================================================
# 7. Model building, validation, and search
# =============================================================================

def build_keras_optimizer(
    tf,
    optimizer: str = DEFAULT_OPTIMIZER,
    initial_learning_rate: float | None = None,
):
    optimizer_name = str(optimizer).lower()
    kwargs = {}
    if initial_learning_rate is not None:
        kwargs["learning_rate"] = float(initial_learning_rate)

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(**kwargs)
    if optimizer_name == "rmsprop":
        return tf.keras.optimizers.RMSprop(**kwargs)
    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(**kwargs)
    raise ValueError(f"Unsupported optimizer: {optimizer}")


def build_keras_loss(tf, loss_function: str = DEFAULT_LOSS_FUNCTION):
    loss_name = str(loss_function).lower()
    if loss_name == "mse":
        return "mse"
    if loss_name == "mae":
        return "mae"
    if loss_name == "huber":
        return tf.keras.losses.Huber()
    raise ValueError(f"Unsupported loss_function: {loss_function}")


def training_setting_summary(args: argparse.Namespace) -> dict[str, object]:
    initial_learning_rate = getattr(args, "initial_learning_rate", None)
    return {
        "bilstm_layers": int(getattr(args, "bilstm_layers", DEFAULT_BILSTM_LAYERS)),
        "epochs": int(getattr(args, "epochs", DEFAULT_EPOCHS)),
        "lr_drop_factor": float(getattr(args, "lr_drop_factor", DEFAULT_LR_DROP_FACTOR)),
        "optimizer": str(getattr(args, "optimizer", DEFAULT_OPTIMIZER)),
        "loss_function": str(getattr(args, "loss_function", DEFAULT_LOSS_FUNCTION)),
        "initial_learning_rate": (
            "keras_default" if initial_learning_rate is None else float(initial_learning_rate)
        ),
    }


def build_bilstm_model(
    tf,
    input_shape: tuple[int, int],
    units: int,
    bilstm_layers: int = DEFAULT_BILSTM_LAYERS,
    optimizer: str = DEFAULT_OPTIMIZER,
    loss_function: str = DEFAULT_LOSS_FUNCTION,
    initial_learning_rate: float | None = None,
):
    def normalized_maape_percent_tf(y_true, y_pred):
        maape = tf.reduce_mean(
            tf.math.atan2(tf.abs(y_true - y_pred), tf.abs(y_true))
        )
        return maape * tf.constant(NORMALIZED_MAAPE_PERCENT_SCALE, dtype=maape.dtype)

    normalized_maape_percent_tf.__name__ = "normalized_maape_percent"

    layers = [tf.keras.layers.Input(shape=input_shape)]
    for layer_index in range(int(bilstm_layers)):
        return_sequences = layer_index < int(bilstm_layers) - 1
        layers.append(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=return_sequences)
            )
        )
    layers.append(tf.keras.layers.Dense(1))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=build_keras_optimizer(tf, optimizer, initial_learning_rate),
        loss=build_keras_loss(tf, loss_function),
        metrics=[normalized_maape_percent_tf],
    )
    return model


# =============================================================================
# 8. Output, plotting, and artifact management
# =============================================================================

def save_prediction_plot(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
    show_plot: bool = False,
) -> None:
    plt.figure(figsize=(15, 6))
    plt.plot(pd.to_datetime(dates), y_true.reshape(-1), label="Actual", linewidth=2.0)
    plt.plot(pd.to_datetime(dates), y_pred.reshape(-1), label="Predicted", linewidth=2.0)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("24-hour rainfall (mm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    if show_plot:
        plt.show()
    plt.close()


def run_grid_search(
    args: argparse.Namespace,
    scaled_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    original_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    scalers: dict[str, object],
    output_dir: Path,
    tf: object | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate every configured BiLSTM combination and retain the best run."""
    if tf is None:
        tf, _ = configure_tensorflow_cpu(
            args.seed,
            getattr(args, "cpu_threads", DEFAULT_CPU_THREADS),
        )

    x_train, y_train, _ = scaled_data["train"]
    x_test, _, test_dates = scaled_data["test"]
    _, y_train_original, _ = original_data["train"]
    _, y_test_original, _ = original_data["test"]
    y_scale = float(scalers["y_scale"])

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    best: dict[str, object] = {"test_normalized_maape_percent": float("inf")}
    training_settings = training_setting_summary(args)
    epochs = int(training_settings["epochs"])
    lr_drop_factor = float(training_settings["lr_drop_factor"])

    total_combinations = (
        len(args.units) * len(args.batch_sizes) * len(args.lr_drop_periods)
    )
    combination_index = 0

    # Every combination starts from a fresh seeded model and writes one result row.
    for units, batch_size, lr_drop_period in itertools.product(
        args.units, args.batch_sizes, args.lr_drop_periods
    ):
        combination_index += 1
        tf.keras.backend.clear_session()
        model = build_bilstm_model(
            tf=tf,
            input_shape=x_train.shape[1:],
            units=units,
            bilstm_layers=int(training_settings["bilstm_layers"]),
            optimizer=str(training_settings["optimizer"]),
            loss_function=str(training_settings["loss_function"]),
            initial_learning_rate=getattr(args, "initial_learning_rate", None),
        )

        def step_decay(epoch: int, current_lr: float) -> float:
            if epoch > 0 and epoch % lr_drop_period == 0:
                return float(current_lr) * lr_drop_factor
            return float(current_lr)

        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(
                step_decay,
                verbose=1 if args.verbose else 0,
            ),
        ]

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=args.verbose,
            shuffle=False,
        )

        y_train_pred = inverse_target_scale(model.predict(x_train, verbose=0), y_scale)
        y_test_pred = inverse_target_scale(model.predict(x_test, verbose=0), y_scale)
        y_train_pred = np.clip(y_train_pred, 0, None)
        y_test_pred = np.clip(y_test_pred, 0, None)

        train_scores = regression_scores(y_train_original, y_train_pred)
        test_scores = regression_scores(y_test_original, y_test_pred)
        best_epoch = int(np.argmin(history.history["normalized_maape_percent"]) + 1)

        row = {
            "units": int(units),
            "batch_size": int(batch_size),
            "lr_drop_period": int(lr_drop_period),
            "lr_drop_factor": lr_drop_factor,
            "bilstm_layers": int(training_settings["bilstm_layers"]),
            "epochs": epochs,
            "optimizer": training_settings["optimizer"],
            "loss_function": training_settings["loss_function"],
            "initial_learning_rate": training_settings["initial_learning_rate"],
            "best_epoch": best_epoch,
            "train_normalized_maape_percent": train_scores["normalized_maape_percent"],
            "train_maape": train_scores["maape"],
            "train_mae": train_scores["mae"],
            "train_rmse": train_scores["rmse"],
            "test_normalized_maape_percent": test_scores["normalized_maape_percent"],
            "test_maape": test_scores["maape"],
            "test_mae": test_scores["mae"],
            "test_rmse": test_scores["rmse"],
        }
        rows.append(row)

        results_df = build_ranked_results(rows)
        results_df.to_csv(output_dir / "hyperparameter_results.csv", index=False)

        print(
            f"[{combination_index}/{total_combinations}] "
            f"layers={training_settings['bilstm_layers']}, units={units}, "
            f"batch={batch_size}, lr_drop_period={lr_drop_period} -> "
            f"train={format_decimal(train_scores['normalized_maape_percent'])}%, "
            f"test={format_decimal(test_scores['normalized_maape_percent'])}%"
        )

        if test_scores["normalized_maape_percent"] < float(best["test_normalized_maape_percent"]):
            best = {
                "train_normalized_maape_percent": train_scores["normalized_maape_percent"],
                "train_maape": train_scores["maape"],
                "train_mae": train_scores["mae"],
                "train_rmse": train_scores["rmse"],
                "test_normalized_maape_percent": test_scores["normalized_maape_percent"],
                "test_maape": test_scores["maape"],
                "test_mae": test_scores["mae"],
                "test_rmse": test_scores["rmse"],
                "units": units,
                "batch_size": batch_size,
                "lr_drop_period": lr_drop_period,
                "lr_drop_factor": lr_drop_factor,
                "bilstm_layers": int(training_settings["bilstm_layers"]),
                "epochs": epochs,
                "optimizer": training_settings["optimizer"],
                "loss_function": training_settings["loss_function"],
                "initial_learning_rate": training_settings["initial_learning_rate"],
                "best_epoch": best_epoch,
                "model": model,
                "y_test_pred": y_test_pred,
                "test_dates": test_dates,
                "history": history.history,
            }
            # Saving a full zipped .keras archive can stall on some Windows/
            # TensorFlow combinations. Architecture plus weights is portable
            # for inference and avoids that serialization bottleneck.
            (output_dir / "best_bilstm_model.json").write_text(
                json.dumps(model.get_config(), indent=2),
                encoding="utf-8",
            )
            model.save_weights(output_dir / "best_bilstm_model.weights.h5")

            pd.DataFrame(
                {
                    "date": pd.to_datetime(test_dates),
                    "actual_rainfall_mm": y_test_original.reshape(-1),
                    "predicted_rainfall_mm": y_test_pred.reshape(-1),
                }
            ).to_csv(output_dir / "best_test_predictions.csv", index=False)

            save_prediction_plot(
                test_dates,
                y_test_original,
                y_test_pred,
                output_dir / "best_test_prediction_plot.png",
                "Rainfall Prediction",
            )

    if args.show_plot and "y_test_pred" in best:
        save_prediction_plot(
            test_dates,
            y_test_original,
            best["y_test_pred"],
            output_dir / "best_test_prediction_plot.png",
            "Rainfall Prediction",
            show_plot=True,
        )

    best.pop("model", None)
    return build_ranked_results(rows), best


# =============================================================================
# 9. Main workflow
# =============================================================================

def main() -> None:
    """Run the command-line workflow from raw CSV data to saved artifacts."""
    args = parse_args()
    validate_args(args)
    set_reproducible_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = args.output_dir
    output_dir = output_root / f"bilstm_run_{timestamp}"

    df, preprocessing_stats, feature_cols, exogenous_cols = load_and_prepare_dataframe(
        data_path=args.data,
        date_col=args.date_col,
        target_col=args.target_col,
        zero_codes=args.zero_codes,
        include_target_history=args.include_target_history,
    )
    x_data, y_data, dates, sequence_stats = make_lagged_sequences(
        df=df,
        feature_cols=feature_cols,
        target_col=args.target_col,
        lag=args.lag,
    )
    lagged_df = build_lagged_dataframe(
        df=df,
        feature_cols=feature_cols,
        target_col=args.target_col,
        lag=args.lag,
    )
    split_data = chronological_split(
        x_data=x_data,
        y_data=y_data,
        dates=dates,
        train_ratio=args.train_ratio,
    )
    scaled_data, scalers = scale_split_data(split_data)

    metadata = {
        "data_path": str(args.data),
        "date_col": args.date_col,
        "target_col": args.target_col,
        "lag": args.lag,
        "feature_cols": feature_cols,
        "exogenous_cols": exogenous_cols,
        "include_target_history": args.include_target_history,
        "zero_codes": args.zero_codes,
        "training_settings": training_setting_summary(args),
        "preprocessing_stats": preprocessing_stats,
        "sequence_stats": sequence_stats,
        "split_sizes": {
            name: int(len(values[0])) for name, values in split_data.items()
        },
        "date_ranges": {
            name: {
                "start": str(pd.to_datetime(values[2][0]).date()),
                "end": str(pd.to_datetime(values[2][-1]).date()),
            }
            for name, values in split_data.items()
        },
        "scalers": scalers,
        "cpu_runtime": {
            "runtime": "not_configured_prepare_only",
            "cpu_threads": getattr(args, "cpu_threads", DEFAULT_CPU_THREADS),
            "onednn_enabled": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "1") == "1",
        },
    }
    print("\nData summary")
    print(f"Daily rows after sorting: {len(df)}")
    print(f"Input features: {feature_cols}")
    print(f"X shape: {x_data.shape} | y shape: {y_data.shape}")
    print(
        f"Codes {args.zero_codes} in X variables were treated as missing values "
        "and filled by linear interpolation."
    )
    print(
        f"Codes {args.zero_codes} and missing values in the rainfall target "
        "were filled with 0."
    )
    print(f"Sequences skipped because they still contained NaN: {sequence_stats['skipped_nan_input']}")
    for split_name, (x_part, _, date_part) in split_data.items():
        print(
            f"{split_name}: {len(x_part)} samples "
            f"({pd.to_datetime(date_part[0]).date()} to {pd.to_datetime(date_part[-1]).date()})"
        )

    if args.prepare_only:
        print("\n--prepare-only is active; TensorFlow training was not run.")
        print("No new output folder is created in prepare-only mode.")
        return

    tf, cpu_runtime = configure_tensorflow_cpu(
        args.seed,
        getattr(args, "cpu_threads", DEFAULT_CPU_THREADS),
    )
    metadata["cpu_runtime"] = cpu_runtime

    output_dir.mkdir(parents=True, exist_ok=True)
    lagged_df.to_csv(output_dir / "lagged_dataset.csv", index=False)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Output folder: {output_dir}")

    results_df, best = run_grid_search(
        args=args,
        scaled_data=scaled_data,
        original_data=split_data,
        scalers=scalers,
        output_dir=output_dir,
        tf=tf,
    )

    print("\nChronological-holdout metrics for all hyperparameter combinations")
    combination_columns = [
        "rank",
        "bilstm_layers",
        "units",
        "batch_size",
        "lr_drop_period",
        "test_normalized_maape_percent",
        "test_maape",
        "test_mae",
        "test_rmse",
    ]
    print(results_df[combination_columns].to_string(index=False, float_format=format_decimal))

    print("\nFinal result summary")
    print(
        "Best hyperparameter combination: "
        f"layers={int(best['bilstm_layers'])}, "
        f"units={int(best['units'])}, "
        f"batch_size={int(best['batch_size'])}, "
        f"lr_drop_period={int(best['lr_drop_period'])}"
    )
    print(
        "Best model Normalized MAAPE (%): "
        f"{format_decimal(best['test_normalized_maape_percent'])}"
    )
    print(f"Best model MAAPE: {format_decimal(best['test_maape'])}")
    print(f"Best model RMSE: {format_decimal(best['test_rmse'])}")
    print(f"Best model MAE: {format_decimal(best['test_mae'])}")

    removed_dirs = cleanup_old_output_runs(output_root, output_dir, args.keep_runs)
    if removed_dirs:
        print("\nOld output folders removed:")
        for path in removed_dirs:
            print(f"- {path}")

if __name__ == "__main__":
    main()
