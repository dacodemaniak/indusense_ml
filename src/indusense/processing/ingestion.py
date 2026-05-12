"""Preparation helpers for Bronze -> Silver -> Gold candidate datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


ARTIFACT_ROOT = Path("artifacts/data-ingestion")


@dataclass(slots=True)
class ImputationContext:
    """Context for a single imputation benchmark."""

    dataset_name: str
    target_column: str
    group_column: str
    time_column: str
    plausible_min: float | None = None
    plausible_max: float | None = None
    mask_fraction: float = 0.1
    random_state: int = 42


def create_artifact_run_dir(root: Path | str = ARTIFACT_ROOT) -> Path:
    """Create a timestamped directory for ingestion artifacts."""
    base_dir = Path(root)
    run_dir = base_dir / datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Artifact directory created at {}", run_dir)
    return run_dir


def normalize_temperature_units(
    frame: pd.DataFrame,
    value_column: str = "temperature",
    group_column: str = "machine_id_std",
    time_column: str = "event_ts",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize temperature readings to Celsius when Fahrenheit windows are detected."""
    working = frame.copy().sort_values([group_column, time_column]).reset_index(drop=True)
    working["temperature_original"] = working[value_column]
    working["temperature_normalized"] = working[value_column]
    working["detected_unit"] = "C"
    working["normalization_rule"] = "identity"

    normalized_windows: list[dict[str, object]] = []

    for machine_id, machine_frame in working.groupby(group_column, dropna=False):
        if pd.isna(machine_id):
            continue

        valid = machine_frame[value_column].dropna()
        baseline_candidates = valid[valid.between(10, 80)]
        baseline = baseline_candidates.median() if not baseline_candidates.empty else valid.median()
        if pd.isna(baseline):
            continue

        raw_values = machine_frame[value_column]
        converted_values = (raw_values - 32.0) * 5.0 / 9.0
        suspected = (
            raw_values.notna()
            & raw_values.between(80, 180)
            & converted_values.between(10, 80)
            & ((converted_values - baseline).abs() + 3 < (raw_values - baseline).abs())
        )

        if not suspected.any():
            continue

        machine_index = machine_frame.index.to_series()
        segment_id = (machine_frame[time_column].diff().gt(pd.Timedelta(hours=2)) | ~suspected).cumsum()

        for _, segment in machine_frame.loc[suspected].groupby(segment_id[suspected]):
            if len(segment) < 3:
                continue
            idx = segment.index
            working.loc[idx, "temperature_normalized"] = converted_values.loc[idx]
            working.loc[idx, "detected_unit"] = "F"
            working.loc[idx, "normalization_rule"] = "fahrenheit_window_to_celsius"
            normalized_windows.append(
                {
                    "machine_id_std": machine_id,
                    "window_start": segment[time_column].min(),
                    "window_end": segment[time_column].max(),
                    "points_converted": len(segment),
                    "raw_mean": float(raw_values.loc[idx].mean()),
                    "converted_mean_celsius": float(converted_values.loc[idx].mean()),
                    "baseline_celsius": float(baseline),
                }
            )

    logger.info(
        "Temperature normalization converted {} windows and {} points",
        len(normalized_windows),
        int((working["detected_unit"] == "F").sum()),
    )
    return working, pd.DataFrame(normalized_windows)


def deduplicate_sensor_records(
    frame: pd.DataFrame,
    dataset_name: str,
    value_column: str,
    key_columns: tuple[str, str] = ("machine_id_std", "event_ts"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate sensor records on their business key."""
    working = frame.copy().sort_values(list(key_columns)).reset_index(drop=True)
    clean_rows: list[pd.Series] = []
    decisions: list[dict[str, object]] = []

    for key_values, group in working.groupby(list(key_columns), dropna=False, sort=False):
        group = group.copy()
        duplicate_count = len(group)
        group_values = group[value_column].dropna()

        if duplicate_count == 1:
            row = group.iloc[0].copy()
            row["is_duplicate"] = False
            row["dedupe_resolution"] = "unique"
            clean_rows.append(row)
            continue

        if group.nunique(dropna=False).max() == 1:
            row = group.iloc[0].copy()
            row["is_duplicate"] = True
            row["dedupe_resolution"] = "strict_duplicate_drop"
            clean_rows.append(row)
            decisions.append(
                {
                    "dataset_name": dataset_name,
                    "key": str(key_values),
                    "records_in_group": duplicate_count,
                    "resolution": "strict_duplicate_drop",
                }
            )
            continue

        row = group.iloc[0].copy()
        if group_values.empty:
            resolved_value = np.nan
            resolution = "all_missing_keep_first"
        elif group_values.nunique() == 1:
            resolved_value = group_values.iloc[0]
            resolution = "keep_non_missing_value"
        else:
            resolved_value = float(group_values.median())
            resolution = "median_conflict_resolution"

        row[value_column] = resolved_value
        row["is_duplicate"] = True
        row["dedupe_resolution"] = resolution
        clean_rows.append(row)
        decisions.append(
            {
                "dataset_name": dataset_name,
                "key": str(key_values),
                "records_in_group": duplicate_count,
                "resolution": resolution,
                "resolved_value": resolved_value,
            }
        )

    logger.info(
        "Deduplicated {} with {} duplicate groups resolved",
        dataset_name,
        len(decisions),
    )
    return pd.DataFrame(clean_rows), pd.DataFrame(decisions)


def deduplicate_incidents(
    frame: pd.DataFrame,
    key_column: str = "incident_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate incidents by keeping the most complete row for each incident id."""
    working = frame.copy().sort_values(key_column).reset_index(drop=True)
    clean_rows: list[pd.Series] = []
    decisions: list[dict[str, object]] = []

    for incident_id, group in working.groupby(key_column, dropna=False, sort=False):
        if len(group) == 1:
            row = group.iloc[0].copy()
            row["is_duplicate"] = False
            row["dedupe_resolution"] = "unique"
            clean_rows.append(row)
            continue

        best_index = group.notna().sum(axis=1).idxmax()
        row = group.loc[best_index].copy()
        row["is_duplicate"] = True
        row["dedupe_resolution"] = "keep_most_complete_row"
        clean_rows.append(row)
        decisions.append(
            {
                "dataset_name": "releves_incidents",
                "key": incident_id,
                "records_in_group": len(group),
                "resolution": "keep_most_complete_row",
            }
        )

    logger.info("Deduplicated incidents with {} duplicate groups resolved", len(decisions))
    return pd.DataFrame(clean_rows), pd.DataFrame(decisions)


def _build_imputation_features(
    frame: pd.DataFrame,
    target_column: str,
    group_column: str,
    time_column: str,
    auxiliary_numeric_columns: list[str] | None = None,
) -> pd.DataFrame:
    engineered = pd.DataFrame(index=frame.index)
    engineered[target_column] = frame[target_column]
    engineered["machine_code_numeric"] = frame[group_column].astype("category").cat.codes.replace(-1, np.nan)
    engineered["hour"] = frame[time_column].dt.hour
    engineered["dayofweek"] = frame[time_column].dt.dayofweek
    engineered["month"] = frame[time_column].dt.month
    engineered["time_ordinal_hours"] = (
        frame[time_column].astype("int64", copy=False) // 3_600_000_000_000
    )

    for column in auxiliary_numeric_columns or []:
        if column in frame.columns and column != target_column:
            engineered[column] = pd.to_numeric(frame[column], errors="coerce")

    return engineered


def _simple_group_impute(
    frame: pd.DataFrame,
    target_column: str,
    group_column: str,
    strategy: str,
) -> pd.Series:
    stats = frame.groupby(group_column)[target_column].transform(strategy)
    if strategy == "median":
        fallback = frame[target_column].median()
    elif strategy == "mean":
        fallback = frame[target_column].mean()
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return frame[target_column].fillna(stats).fillna(fallback)


def _iterative_impute(
    frame: pd.DataFrame,
    target_column: str,
    group_column: str,
    time_column: str,
    auxiliary_numeric_columns: list[str] | None,
    random_state: int,
) -> pd.Series:
    features = _build_imputation_features(
        frame=frame,
        target_column=target_column,
        group_column=group_column,
        time_column=time_column,
        auxiliary_numeric_columns=auxiliary_numeric_columns,
    )
    imputer = IterativeImputer(
        random_state=random_state,
        initial_strategy="median",
        max_iter=15,
        sample_posterior=False,
    )
    imputed = imputer.fit_transform(features)
    return pd.Series(imputed[:, 0], index=frame.index, name=f"{target_column}_iterative")


def _distribution_drift(
    original: pd.Series,
    imputed: pd.Series,
) -> float:
    original_stats = np.array(
        [
            original.mean(),
            original.median(),
            original.std(ddof=0),
        ],
        dtype=float,
    )
    imputed_stats = np.array(
        [
            imputed.mean(),
            imputed.median(),
            imputed.std(ddof=0),
        ],
        dtype=float,
    )
    return float(np.nanmean(np.abs(original_stats - imputed_stats)))


def evaluate_imputation_strategies(
    frame: pd.DataFrame,
    context: ImputationContext,
    auxiliary_numeric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark mean, median and iterative imputers on masked observed data."""
    working = frame.copy().reset_index(drop=True)
    observed_mask = working[context.target_column].notna()
    observed_index = working.index[observed_mask]

    if len(observed_index) < 30:
        raise ValueError(f"Not enough observed rows to evaluate {context.dataset_name}.{context.target_column}")

    rng = np.random.default_rng(context.random_state)
    mask_size = max(10, int(len(observed_index) * context.mask_fraction))
    mask_size = min(mask_size, len(observed_index))
    masked_index = rng.choice(observed_index.to_numpy(), size=mask_size, replace=False)

    benchmark_frame = working.copy()
    ground_truth = benchmark_frame.loc[masked_index, context.target_column].astype(float)
    benchmark_frame.loc[masked_index, context.target_column] = np.nan

    candidates = {
        "mean": _simple_group_impute(
            benchmark_frame,
            target_column=context.target_column,
            group_column=context.group_column,
            strategy="mean",
        ),
        "median": _simple_group_impute(
            benchmark_frame,
            target_column=context.target_column,
            group_column=context.group_column,
            strategy="median",
        ),
        "iterative": _iterative_impute(
            benchmark_frame,
            target_column=context.target_column,
            group_column=context.group_column,
            time_column=context.time_column,
            auxiliary_numeric_columns=auxiliary_numeric_columns,
            random_state=context.random_state,
        ),
    }

    metrics_rows: list[dict[str, object]] = []
    imputed_columns = pd.DataFrame(index=working.index)

    for method_name, candidate in candidates.items():
        prediction = candidate.loc[masked_index].astype(float)
        full_candidate = working[context.target_column].fillna(candidate).astype(float)

        if context.plausible_min is not None and context.plausible_max is not None:
            out_of_range_rate = float(
                (~full_candidate.between(context.plausible_min, context.plausible_max)).mean()
            )
        else:
            out_of_range_rate = 0.0

        metrics_rows.append(
            {
                "dataset_name": context.dataset_name,
                "target_column": context.target_column,
                "method": method_name,
                "masked_rows": len(masked_index),
                "mae": float(mean_absolute_error(ground_truth, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(ground_truth, prediction))),
                "medae": float(median_absolute_error(ground_truth, prediction)),
                "bias": float((prediction - ground_truth).mean()),
                "out_of_range_rate": out_of_range_rate,
                "distribution_drift": _distribution_drift(
                    original=working.loc[observed_mask, context.target_column].astype(float),
                    imputed=full_candidate,
                ),
            }
        )
        imputed_columns[f"{context.target_column}_{method_name}"] = full_candidate

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["out_of_range_rate", "rmse", "mae", "distribution_drift", "medae"]
    )
    return metrics.reset_index(drop=True), imputed_columns


def summarize_imputation_decisions(metrics: pd.DataFrame) -> pd.DataFrame:
    """Convert method metrics into a one-line decision per dataset and column."""
    decisions: list[dict[str, object]] = []
    for (dataset_name, target_column), group in metrics.groupby(["dataset_name", "target_column"], sort=False):
        best = group.sort_values(
            ["out_of_range_rate", "rmse", "mae", "distribution_drift", "medae"]
        ).iloc[0]
        decisions.append(
            {
                "dataset_name": dataset_name,
                "target_column": target_column,
                "recommended_method": best["method"],
                "mae": best["mae"],
                "rmse": best["rmse"],
                "medae": best["medae"],
                "bias": best["bias"],
                "out_of_range_rate": best["out_of_range_rate"],
                "distribution_drift": best["distribution_drift"],
                "decision_reason": (
                    f"Best trade-off on out-of-range rate, RMSE and MAE for {dataset_name}.{target_column}"
                ),
            }
        )
    return pd.DataFrame(decisions)


def build_sensor_silver_candidate(
    frame: pd.DataFrame,
    value_column: str,
    sensor_type: str,
    unit: str,
    chosen_method: str,
    imputed_columns: pd.DataFrame,
) -> pd.DataFrame:
    """Create a Silver candidate frame from normalized, deduplicated sensor data."""
    value_series = frame[value_column].copy()
    if chosen_method != "none" and f"{value_column}_{chosen_method}" in imputed_columns.columns:
        value_series = value_series.fillna(imputed_columns[f"{value_column}_{chosen_method}"])

    silver = frame.copy()
    silver["sensor_type"] = sensor_type
    silver["unit"] = unit
    silver["sensor_value"] = value_series
    silver["is_missing"] = frame[value_column].isna()
    silver["was_imputed"] = frame[value_column].isna() & silver["sensor_value"].notna()
    return silver


def build_incident_silver_candidate(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a Silver incident candidate with conservative imputation rules."""
    silver = frame.copy()
    silver["operator_name_clean"] = silver["operator_name"].fillna("UNKNOWN")
    silver["operator_badge_clean"] = silver["operator_badge"].fillna("UNKNOWN")
    silver["comment_clean"] = silver["comment"]
    silver["severity_imputation_policy"] = np.where(
        silver["severity"].isna(),
        "reject_or_manual_rule",
        "keep_original",
    )
    return silver


def build_gold_dataset_candidate(
    temperature_silver: pd.DataFrame,
    pressure_silver: pd.DataFrame,
    incident_silver: pd.DataFrame,
    time_column: str = "event_ts",
    machine_column: str = "machine_id_std",
) -> pd.DataFrame:
    """Build a multi-horizon hourly Gold candidate with rolling window features for ML training.

    Rolling windows computed per machine (6h, 12h, 24h) for temperature and pressure cover
    short-term spikes, medium-term drift and daily patterns without cross-machine leakage.
    Labels are computed at four horizons (6h, 12h, 24h, 48h) to support model selection.
    """
    temp_hourly = (
        temperature_silver.dropna(subset=[time_column, "sensor_value"])
        .assign(window_start=lambda df: df[time_column].dt.floor("h"))
        .groupby([machine_column, "window_start"], as_index=False)
        .agg(
            temp_mean_1h=("sensor_value", "mean"),
            temp_max_1h=("sensor_value", "max"),
            temp_missing_count=("is_missing", "sum"),
        )
    )
    pressure_hourly = (
        pressure_silver.dropna(subset=[time_column, "sensor_value"])
        .assign(window_start=lambda df: df[time_column].dt.floor("h"))
        .groupby([machine_column, "window_start"], as_index=False)
        .agg(
            pressure_mean_1h=("sensor_value", "mean"),
            pressure_max_1h=("sensor_value", "max"),
            pressure_missing_count=("is_missing", "sum"),
        )
    )
    gold = (
        temp_hourly.merge(
            pressure_hourly,
            on=[machine_column, "window_start"],
            how="outer",
        )
        .sort_values([machine_column, "window_start"])
        .reset_index(drop=True)
    )
    gold["window_end"] = gold["window_start"] + pd.Timedelta(hours=1)

    # Multi-horizon rolling windows for temperature and pressure (6h / 12h / 24h)
    for hours in [6, 12, 24]:
        gold[f"temp_mean_{hours}h"] = gold.groupby(machine_column)["temp_mean_1h"].transform(
            lambda s, w=hours: s.rolling(w, min_periods=1).mean()
        )
        gold[f"temp_max_{hours}h"] = gold.groupby(machine_column)["temp_max_1h"].transform(
            lambda s, w=hours: s.rolling(w, min_periods=1).max()
        )
        gold[f"temp_std_{hours}h"] = gold.groupby(machine_column)["temp_mean_1h"].transform(
            lambda s, w=hours: s.rolling(w, min_periods=2).std()
        )
        gold[f"pressure_mean_{hours}h"] = gold.groupby(machine_column)["pressure_mean_1h"].transform(
            lambda s, w=hours: s.rolling(w, min_periods=1).mean()
        )
        gold[f"pressure_max_{hours}h"] = gold.groupby(machine_column)["pressure_max_1h"].transform(
            lambda s, w=hours: s.rolling(w, min_periods=1).max()
        )
        gold[f"pressure_std_{hours}h"] = gold.groupby(machine_column)["pressure_mean_1h"].transform(
            lambda s, w=hours: s.rolling(w, min_periods=2).std()
        )

    # Trend features: temperature and pressure change over the past 6 hours
    gold["temp_trend_6h"] = gold.groupby(machine_column)["temp_mean_1h"].transform(
        lambda s: s - s.shift(6)
    )
    gold["pressure_trend_6h"] = gold.groupby(machine_column)["pressure_mean_1h"].transform(
        lambda s: s - s.shift(6)
    )

    # Rolling z-score: standardised anomaly score relative to the 24h baseline
    temp_std_24h_safe = gold["temp_std_24h"].replace(0.0, np.nan)
    gold["temp_zscore_24h"] = (
        (gold["temp_mean_1h"] - gold["temp_mean_24h"]) / temp_std_24h_safe
    ).clip(-10.0, 10.0)

    # Incident features
    incident_events = incident_silver.copy()
    incident_events["window_start"] = incident_events["event_ts"].dt.floor("h")
    incident_features = (
        incident_events.groupby([machine_column, "window_start"], as_index=False)
        .agg(
            incident_count_1h=("incident_id", "count"),
            incident_max_severity_1h=("severity", "max"),
        )
    )
    gold = gold.merge(incident_features, on=[machine_column, "window_start"], how="left")
    gold["incident_count_1h"] = gold["incident_count_1h"].fillna(0)

    gold["incident_count_prev_24h"] = gold.groupby(machine_column)["incident_count_1h"].transform(
        lambda s: s.rolling(24, min_periods=1).sum()
    )
    gold["incident_max_severity_prev_24h"] = gold.groupby(machine_column)["incident_max_severity_1h"].transform(
        lambda s: s.rolling(24, min_periods=1).max()
    )
    # 7-day lookback captures chronic maintenance patterns beyond the daily window
    gold["incident_count_prev_7d"] = gold.groupby(machine_column)["incident_count_1h"].transform(
        lambda s: s.rolling(168, min_periods=1).sum()
    )

    # Hours since last incident per machine (timestamp-aware; NaN if no prior incident)
    machine_frames: list[pd.DataFrame] = []
    for _, grp in gold.groupby(machine_column, sort=False):
        grp = grp.copy()
        last_incident_ts = grp["window_start"].where(grp["incident_count_1h"] > 0).ffill()
        grp["hours_since_last_incident"] = (
            (grp["window_start"] - last_incident_ts).dt.total_seconds() / 3600
        )
        machine_frames.append(grp)
    gold = (
        pd.concat(machine_frames)
        .sort_values([machine_column, "window_start"])
        .reset_index(drop=True)
    )

    # Multi-horizon lookahead labels — future window only, no leakage into features
    for hours in [6, 12, 24, 48]:
        future_count = gold.groupby(machine_column)["incident_count_1h"].transform(
            lambda s, w=hours: s[::-1].rolling(w, min_periods=1).sum()[::-1]
        )
        gold[f"future_incident_count_{hours}h"] = future_count
        gold[f"label_failure_next_{hours}h"] = future_count > 0

    gold["split_set"] = np.select(
        [
            gold["window_start"] < gold["window_start"].quantile(0.7),
            gold["window_start"] < gold["window_start"].quantile(0.85),
        ],
        ["train", "validation"],
        default="test",
    )
    logger.info("Gold dataset candidate built with {} rows", len(gold))
    return gold
