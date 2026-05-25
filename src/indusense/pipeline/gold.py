"""Gold layer: Silver tables → GoldMachineHourlyFeature (truncate + full rebuild)."""

from __future__ import annotations

from datetime import timezone
from typing import Any

import pandas as pd
import sqlalchemy as sa
from loguru import logger
from sqlalchemy.orm import Session

from indusense.db.models import (
    GoldMachineHourlyFeature,
    INCIDENT_TYPE_COLS,
    Machine,
    SensorType,
    SilverIncident,
    SilverSensorReading,
    SilverWeatherReading,
    SplitSet,
)
from indusense.processing import build_gold_dataset_candidate


def _to_naive(series: pd.Series) -> pd.Series:
    """Strip timezone info from a datetime series for processing functions."""
    s = pd.to_datetime(series)
    if s.dt.tz is not None:
        return s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def _to_utc(val: Any) -> Any:
    """Convert a pandas Timestamp or datetime to UTC-aware datetime."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    ts = pd.Timestamp(val)
    if ts is pd.NaT:
        return None
    return ts.to_pydatetime().replace(tzinfo=timezone.utc)


def _f(val: Any) -> float | None:
    return float(val) if pd.notna(val) else None


def _i(val: Any) -> int | None:
    return int(val) if pd.notna(val) else None


_SPLIT_MAP: dict[str, SplitSet] = {
    "train": SplitSet.TRAIN,
    "validation": SplitSet.VALIDATION,
    "test": SplitSet.TEST,
}


def build_gold(session: Session) -> int:
    """Truncate GoldMachineHourlyFeature and rebuild from all Silver data."""
    machine_map: dict[int, str] = {
        row.machine_id: row.machine_code
        for row in session.execute(sa.select(Machine.machine_id, Machine.machine_code)).all()
    }
    code_to_id: dict[str, int] = {v: k for k, v in machine_map.items()}

    # Load Silver temperature
    temp_rows = session.execute(
        sa.select(
            SilverSensorReading.machine_id,
            SilverSensorReading.observed_at,
            SilverSensorReading.sensor_value,
            SilverSensorReading.is_missing,
            SilverSensorReading.is_duplicate,
        ).where(SilverSensorReading.sensor_type == SensorType.TEMPERATURE)
    ).all()

    temp_df = pd.DataFrame(
        [
            (machine_map.get(r.machine_id), r.observed_at,
             float(r.sensor_value) if r.sensor_value is not None else None,
             r.is_missing, r.is_duplicate)
            for r in temp_rows
        ],
        columns=["machine_id_std", "event_ts", "sensor_value", "is_missing", "is_duplicate"],
    )
    temp_df["event_ts"] = _to_naive(temp_df["event_ts"])
    temp_df = temp_df.dropna(subset=["machine_id_std"])

    # Load Silver pressure
    press_rows = session.execute(
        sa.select(
            SilverSensorReading.machine_id,
            SilverSensorReading.observed_at,
            SilverSensorReading.sensor_value,
            SilverSensorReading.is_missing,
            SilverSensorReading.is_duplicate,
        ).where(SilverSensorReading.sensor_type == SensorType.PRESSURE)
    ).all()

    press_df = pd.DataFrame(
        [
            (machine_map.get(r.machine_id), r.observed_at,
             float(r.sensor_value) if r.sensor_value is not None else None,
             r.is_missing, r.is_duplicate)
            for r in press_rows
        ],
        columns=["machine_id_std", "event_ts", "sensor_value", "is_missing", "is_duplicate"],
    )
    press_df["event_ts"] = _to_naive(press_df["event_ts"])
    press_df = press_df.dropna(subset=["machine_id_std"])

    # Load Silver incidents WITH type columns
    type_col_attrs = [getattr(SilverIncident, col) for col in INCIDENT_TYPE_COLS]
    inc_rows = session.execute(
        sa.select(
            SilverIncident.machine_id,
            SilverIncident.occurred_at,
            SilverIncident.severity,
            SilverIncident.incident_code,
            *type_col_attrs,
        )
    ).all()

    inc_df = pd.DataFrame(
        [
            (machine_map.get(r.machine_id), r.occurred_at, r.severity, r.incident_code,
             *[getattr(r, col) for col in INCIDENT_TYPE_COLS])
            for r in inc_rows
        ],
        columns=["machine_id_std", "event_ts", "severity", "incident_id", *INCIDENT_TYPE_COLS],
    )
    inc_df["event_ts"] = _to_naive(inc_df["event_ts"])
    inc_df = inc_df.dropna(subset=["machine_id_std"])

    # Feature engineering (rolling windows, labels, split)
    gold_df = build_gold_dataset_candidate(temp_df, press_df, inc_df)

    # ── Incident type 24h rolling counts ──────────────────────────────────────
    inc_types = inc_df.copy()
    inc_types["window_start"] = inc_types["event_ts"].dt.floor("h")
    for col in INCIDENT_TYPE_COLS:
        inc_types[col] = pd.to_numeric(inc_types[col], errors="coerce").fillna(0).astype(int)

    type_hourly = (
        inc_types.groupby(["machine_id_std", "window_start"])[INCIDENT_TYPE_COLS]
        .sum()
        .reset_index()
    )
    gold_df = gold_df.merge(type_hourly, on=["machine_id_std", "window_start"], how="left")
    for col in INCIDENT_TYPE_COLS:
        gold_df[col] = gold_df[col].fillna(0).astype(int)
        gold_df[f"{col}_count_prev_24h"] = (
            gold_df.groupby("machine_id_std")[col]
            .transform(lambda s: s.rolling(24, min_periods=1).sum())
            .fillna(0)
            .astype(int)
        )

    # ── Ambient weather join (by UTC hour) ────────────────────────────────────
    weather_rows = session.execute(
        sa.select(
            SilverWeatherReading.observed_at,
            SilverWeatherReading.temp_celsius,
            SilverWeatherReading.humidity_pct,
            SilverWeatherReading.pressure_hpa,
        )
    ).all()

    if weather_rows:
        weather_df = pd.DataFrame(
            weather_rows,
            columns=["observed_at", "temp_celsius", "humidity_pct", "pressure_hpa"],
        )
        weather_df["window_start"] = _to_naive(weather_df["observed_at"]).dt.floor("h")
        weather_df = (
            weather_df.drop(columns=["observed_at"])
            .drop_duplicates(subset=["window_start"])
        )
        gold_df = gold_df.merge(weather_df, on="window_start", how="left")
    else:
        gold_df["temp_celsius"] = None
        gold_df["humidity_pct"] = None
        gold_df["pressure_hpa"] = None

    # Truncate Gold table before full rebuild
    session.execute(sa.delete(GoldMachineHourlyFeature))
    session.flush()

    # Prepare insert dicts
    rows: list[dict[str, Any]] = []
    for _, r in gold_df.iterrows():
        machine_id = code_to_id.get(r.get("machine_id_std"))
        if machine_id is None:
            continue
        type_counts = {
            f"{col}_count_prev_24h": int(r.get(f"{col}_count_prev_24h") or 0)
            for col in INCIDENT_TYPE_COLS
        }
        rows.append({
            "machine_id": machine_id,
            "window_start": _to_utc(r.get("window_start")),
            "window_end": _to_utc(r.get("window_end")),
            # 6h features
            "temp_mean_6h": _f(r.get("temp_mean_6h")),
            "temp_max_6h": _f(r.get("temp_max_6h")),
            "temp_std_6h": _f(r.get("temp_std_6h")),
            "pressure_mean_6h": _f(r.get("pressure_mean_6h")),
            "pressure_max_6h": _f(r.get("pressure_max_6h")),
            "pressure_std_6h": _f(r.get("pressure_std_6h")),
            # 12h features
            "temp_mean_12h": _f(r.get("temp_mean_12h")),
            "temp_max_12h": _f(r.get("temp_max_12h")),
            "temp_std_12h": _f(r.get("temp_std_12h")),
            "pressure_mean_12h": _f(r.get("pressure_mean_12h")),
            "pressure_max_12h": _f(r.get("pressure_max_12h")),
            "pressure_std_12h": _f(r.get("pressure_std_12h")),
            # 24h features
            "temp_mean_24h": _f(r.get("temp_mean_24h")),
            "temp_max_24h": _f(r.get("temp_max_24h")),
            "temp_std_24h": _f(r.get("temp_std_24h")),
            "pressure_mean_24h": _f(r.get("pressure_mean_24h")),
            "pressure_max_24h": _f(r.get("pressure_max_24h")),
            "pressure_std_24h": _f(r.get("pressure_std_24h")),
            # Trend and anomaly
            "temp_trend_6h": _f(r.get("temp_trend_6h")),
            "pressure_trend_6h": _f(r.get("pressure_trend_6h")),
            "temp_zscore_24h": _f(r.get("temp_zscore_24h")),
            # Incident lookback
            "incident_count_prev_24h": int(r.get("incident_count_prev_24h") or 0),
            "incident_max_severity_prev_24h": _i(r.get("incident_max_severity_prev_24h")),
            "incident_count_prev_7d": int(r.get("incident_count_prev_7d") or 0),
            "hours_since_last_incident": _f(r.get("hours_since_last_incident")),
            # Incident type counts
            **type_counts,
            # Ambient weather
            "ambient_temp_c": _f(r.get("temp_celsius")),
            "ambient_humidity_pct": _f(r.get("humidity_pct")),
            "ambient_pressure_hpa": _f(r.get("pressure_hpa")),
            # Multi-horizon labels
            "label_failure_next_6h": bool(r.get("label_failure_next_6h", False)),
            "label_failure_next_12h": bool(r.get("label_failure_next_12h", False)),
            "label_failure_next_24h": bool(r.get("label_failure_next_24h", False)),
            "label_failure_next_48h": bool(r.get("label_failure_next_48h", False)),
            # Split
            "split_set": _SPLIT_MAP.get(str(r.get("split_set", "train")), SplitSet.TRAIN),
        })

    batch_size = 500
    for i in range(0, len(rows), batch_size):
        session.execute(sa.insert(GoldMachineHourlyFeature), rows[i : i + batch_size])
    session.flush()

    logger.info("Gold dataset: {} rows inserted", len(rows))
    return len(rows)
