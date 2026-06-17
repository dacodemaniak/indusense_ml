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
    Maintenance,
    SensorType,
    SilverIncident,
    SilverSensorReading,
    SilverTelemetryReading,
    SilverWeatherReading,
    SplitSet,
)
from indusense.processing import build_gold_dataset_candidate, build_gold_from_telemetry


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


def _gold_row(r: Any, machine_id: int, type_counts: dict[str, int]) -> dict[str, Any]:
    """Convert a gold DataFrame row to a GoldMachineHourlyFeature insert dict."""
    return {
        "machine_id": machine_id,
        "window_start": _to_utc(r.get("window_start")),
        "window_end": _to_utc(r.get("window_end")),
        # 6h
        "temp_mean_6h": _f(r.get("temp_mean_6h")),
        "temp_max_6h": _f(r.get("temp_max_6h")),
        "temp_std_6h": _f(r.get("temp_std_6h")),
        "pressure_mean_6h": _f(r.get("pressure_mean_6h")),
        "pressure_max_6h": _f(r.get("pressure_max_6h")),
        "pressure_std_6h": _f(r.get("pressure_std_6h")),
        # 12h
        "temp_mean_12h": _f(r.get("temp_mean_12h")),
        "temp_max_12h": _f(r.get("temp_max_12h")),
        "temp_std_12h": _f(r.get("temp_std_12h")),
        "pressure_mean_12h": _f(r.get("pressure_mean_12h")),
        "pressure_max_12h": _f(r.get("pressure_max_12h")),
        "pressure_std_12h": _f(r.get("pressure_std_12h")),
        # 24h
        "temp_mean_24h": _f(r.get("temp_mean_24h")),
        "temp_max_24h": _f(r.get("temp_max_24h")),
        "temp_std_24h": _f(r.get("temp_std_24h")),
        "pressure_mean_24h": _f(r.get("pressure_mean_24h")),
        "pressure_max_24h": _f(r.get("pressure_max_24h")),
        "pressure_std_24h": _f(r.get("pressure_std_24h")),
        # Trend + anomaly
        "temp_trend_6h": _f(r.get("temp_trend_6h")),
        "pressure_trend_6h": _f(r.get("pressure_trend_6h")),
        "temp_zscore_24h": _f(r.get("temp_zscore_24h")),
        # Voltage rolling
        "voltage_mean_6h": _f(r.get("voltage_mean_6h")),
        "voltage_std_6h": _f(r.get("voltage_std_6h")),
        "voltage_mean_12h": _f(r.get("voltage_mean_12h")),
        "voltage_std_12h": _f(r.get("voltage_std_12h")),
        "voltage_mean_24h": _f(r.get("voltage_mean_24h")),
        "voltage_std_24h": _f(r.get("voltage_std_24h")),
        # Rotation rolling
        "rotation_mean_6h": _f(r.get("rotation_mean_6h")),
        "rotation_std_6h": _f(r.get("rotation_std_6h")),
        "rotation_mean_12h": _f(r.get("rotation_mean_12h")),
        "rotation_std_12h": _f(r.get("rotation_std_12h")),
        "rotation_mean_24h": _f(r.get("rotation_mean_24h")),
        "rotation_std_24h": _f(r.get("rotation_std_24h")),
        # Production / maintenance
        "pieces_produced_sum_24h": int(r.get("pieces_produced_sum_24h") or 0),
        "capacity_utilization_pct": _f(r.get("capacity_utilization_pct")),
        "days_since_last_maintenance": _f(r.get("days_since_last_maintenance")),
        "maintenance_count_prev_30d": int(r.get("maintenance_count_prev_30d") or 0),
        # Incident lookback
        "incident_count_prev_24h": int(r.get("incident_count_prev_24h") or 0),
        "incident_max_severity_prev_24h": _i(r.get("incident_max_severity_prev_24h")),
        "incident_count_prev_7d": int(r.get("incident_count_prev_7d") or 0),
        "hours_since_last_incident": _f(r.get("hours_since_last_incident")),
        # Incident type counts
        **type_counts,
        # Ambient weather
        "ambient_temp_c": _f(r.get("temp_celsius") or r.get("ambient_temp_c")),
        "ambient_humidity_pct": _f(r.get("humidity_pct") or r.get("ambient_humidity_pct")),
        "ambient_pressure_hpa": _f(r.get("pressure_hpa") or r.get("ambient_pressure_hpa")),
        # Short-lag delta features
        "temp_delta_1h": _f(r.get("temp_delta_1h")),
        "temp_delta_3h": _f(r.get("temp_delta_3h")),
        "pressure_delta_1h": _f(r.get("pressure_delta_1h")),
        "pressure_delta_3h": _f(r.get("pressure_delta_3h")),
        "rotation_delta_1h": _f(r.get("rotation_delta_1h")),
        "rotation_delta_3h": _f(r.get("rotation_delta_3h")),
        "voltage_delta_1h": _f(r.get("voltage_delta_1h")),
        # Per-machine z-scores
        "temp_zscore_machine": _f(r.get("temp_zscore_machine")),
        "pressure_zscore_machine": _f(r.get("pressure_zscore_machine")),
        # Labels
        "label_failure_next_6h": bool(r.get("label_failure_next_6h", False)),
        "label_failure_next_12h": bool(r.get("label_failure_next_12h", False)),
        "label_failure_next_24h": bool(r.get("label_failure_next_24h", False)),
        "label_failure_next_48h": bool(r.get("label_failure_next_48h", False)),
        "split_set": _SPLIT_MAP.get(str(r.get("split_set", "train")), SplitSet.TRAIN),
    }


def _load_telemetry_silver(session: Session, machine_map: dict[int, str]) -> pd.DataFrame:
    """Load SilverTelemetryReading as a DataFrame with machine_id_std column."""
    rows = session.execute(
        sa.select(
            SilverTelemetryReading.machine_id,
            SilverTelemetryReading.observed_at,
            SilverTelemetryReading.temperature_c,
            SilverTelemetryReading.pressure_bar,
            SilverTelemetryReading.voltage_mean_v,
            SilverTelemetryReading.rotation_mean_rpm,
            SilverTelemetryReading.pieces_produced,
            SilverTelemetryReading.is_missing_temp,
            SilverTelemetryReading.is_missing_pressure,
        )
    ).all()
    df = pd.DataFrame(
        [
            (
                machine_map.get(r.machine_id), r.observed_at,
                float(r.temperature_c) if r.temperature_c is not None else None,
                float(r.pressure_bar) if r.pressure_bar is not None else None,
                float(r.voltage_mean_v) if r.voltage_mean_v is not None else None,
                float(r.rotation_mean_rpm) if r.rotation_mean_rpm is not None else None,
                int(r.pieces_produced) if r.pieces_produced is not None else None,
                r.is_missing_temp,
                r.is_missing_pressure,
            )
            for r in rows
        ],
        columns=[
            "machine_id_std", "event_ts",
            "temperature_c", "pressure_bar", "voltage_mean_v", "rotation_mean_rpm",
            "pieces_produced", "is_missing_temp", "is_missing_pressure",
        ],
    )
    df["event_ts"] = _to_naive(df["event_ts"])
    return df.dropna(subset=["machine_id_std"])


def _load_maintenance(session: Session, machine_map: dict[int, str]) -> pd.DataFrame:
    rows = session.execute(
        sa.select(Maintenance.machine_id, Maintenance.maintenance_at)
    ).all()
    df = pd.DataFrame(
        [(machine_map.get(r.machine_id), r.maintenance_at) for r in rows],
        columns=["machine_id_std", "maintenance_at"],
    )
    df["maintenance_at"] = _to_naive(df["maintenance_at"])
    return df.dropna(subset=["machine_id_std"])


def _load_incidents(session: Session, machine_map: dict[int, str]) -> pd.DataFrame:
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
    df = pd.DataFrame(
        [
            (machine_map.get(r.machine_id), r.occurred_at, r.severity, r.incident_code,
             *[getattr(r, col) for col in INCIDENT_TYPE_COLS])
            for r in inc_rows
        ],
        columns=["machine_id_std", "event_ts", "severity", "incident_id", *INCIDENT_TYPE_COLS],
    )
    df["event_ts"] = _to_naive(df["event_ts"])
    return df.dropna(subset=["machine_id_std"])


def _load_weather(session: Session) -> pd.DataFrame | None:
    rows = session.execute(
        sa.select(
            SilverWeatherReading.observed_at,
            SilverWeatherReading.temp_celsius,
            SilverWeatherReading.humidity_pct,
            SilverWeatherReading.pressure_hpa,
        )
    ).all()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["observed_at", "temp_celsius", "humidity_pct", "pressure_hpa"])
    df["window_start"] = _to_naive(df["observed_at"]).dt.floor("h")
    return df.drop(columns=["observed_at"]).drop_duplicates(subset=["window_start"])


def _insert_gold_df(session: Session, gold_df: pd.DataFrame, code_to_id: dict[str, int]) -> int:
    """Truncate Gold table and bulk-insert all rows from gold_df."""
    session.execute(sa.delete(GoldMachineHourlyFeature))
    session.flush()

    rows: list[dict[str, Any]] = []
    for _, r in gold_df.iterrows():
        machine_id = code_to_id.get(r.get("machine_id_std"))
        if machine_id is None:
            continue
        type_counts = {
            f"{col}_count_prev_24h": int(r.get(f"{col}_count_prev_24h") or 0)
            for col in INCIDENT_TYPE_COLS
        }
        rows.append(_gold_row(r, machine_id, type_counts))

    batch_size = 500
    for i in range(0, len(rows), batch_size):
        session.execute(sa.insert(GoldMachineHourlyFeature), rows[i : i + batch_size])
    session.flush()
    return len(rows)


def build_gold(session: Session) -> int:
    """Truncate GoldMachineHourlyFeature and rebuild from Silver data.

    Uses SilverTelemetryReading (unified telemetry) when available, otherwise
    falls back to the legacy SilverSensorReading path (separate temp/pressure).
    """
    machine_map: dict[int, str] = {
        row.machine_id: row.machine_code
        for row in session.execute(sa.select(Machine.machine_id, Machine.machine_code)).all()
    }
    code_to_id: dict[str, int] = {v: k for k, v in machine_map.items()}

    inc_df = _load_incidents(session, machine_map)
    weather_df = _load_weather(session)

    # ── Telemetry path (unified format) ───────────────────────────────────────
    tel_count = session.execute(sa.select(sa.func.count(SilverTelemetryReading.telemetry_reading_id))).scalar()
    if tel_count and tel_count > 0:
        logger.info("Gold: using SilverTelemetryReading ({} rows)", tel_count)
        tel_df = _load_telemetry_silver(session, machine_map)
        maint_df = _load_maintenance(session, machine_map)

        machine_meta = pd.DataFrame(
            [
                {"machine_id_std": row.machine_code, "max_hourly_capacity_pieces": row.max_hourly_capacity_pieces}
                for row in session.execute(
                    sa.select(Machine.machine_code, Machine.max_hourly_capacity_pieces)
                ).all()
            ]
        )

        gold_df = build_gold_from_telemetry(
            telemetry_silver=tel_df,
            incident_silver=inc_df,
            machine_meta=machine_meta if not machine_meta.empty else None,
            maintenance_df=maint_df if not maint_df.empty else None,
        )

    # ── Legacy path (separate temperature + pressure Silver) ──────────────────
    else:
        logger.info("Gold: SilverTelemetryReading empty, falling back to SilverSensorReading")
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
            [(machine_map.get(r.machine_id), r.observed_at,
              float(r.sensor_value) if r.sensor_value is not None else None,
              r.is_missing, r.is_duplicate) for r in temp_rows],
            columns=["machine_id_std", "event_ts", "sensor_value", "is_missing", "is_duplicate"],
        )
        temp_df["event_ts"] = _to_naive(temp_df["event_ts"])
        temp_df = temp_df.dropna(subset=["machine_id_std"])

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
            [(machine_map.get(r.machine_id), r.observed_at,
              float(r.sensor_value) if r.sensor_value is not None else None,
              r.is_missing, r.is_duplicate) for r in press_rows],
            columns=["machine_id_std", "event_ts", "sensor_value", "is_missing", "is_duplicate"],
        )
        press_df["event_ts"] = _to_naive(press_df["event_ts"])
        press_df = press_df.dropna(subset=["machine_id_std"])

        gold_df = build_gold_dataset_candidate(temp_df, press_df, inc_df)

        inc_types = inc_df.copy()
        inc_types["window_start"] = inc_types["event_ts"].dt.floor("h")
        for col in INCIDENT_TYPE_COLS:
            inc_types[col] = pd.to_numeric(inc_types[col], errors="coerce").fillna(0).astype(int)
        type_hourly = (
            inc_types.groupby(["machine_id_std", "window_start"])[INCIDENT_TYPE_COLS]
            .sum().reset_index()
        )
        gold_df = gold_df.merge(type_hourly, on=["machine_id_std", "window_start"], how="left")
        for col in INCIDENT_TYPE_COLS:
            gold_df[col] = gold_df[col].fillna(0).astype(int)
            gold_df[f"{col}_count_prev_24h"] = (
                gold_df.groupby("machine_id_std")[col]
                .transform(lambda s: s.rolling(24, min_periods=1).sum())
                .fillna(0).astype(int)
            )

    # ── Weather join (both paths) ─────────────────────────────────────────────
    if weather_df is not None:
        gold_df = gold_df.merge(weather_df, on="window_start", how="left")
    else:
        gold_df["temp_celsius"] = None
        gold_df["humidity_pct"] = None
        gold_df["pressure_hpa"] = None

    n = _insert_gold_df(session, gold_df, code_to_id)
    logger.info("Gold dataset: {} rows inserted", n)
    return n


def persist_gold_to_db(gold_df: pd.DataFrame, session_factory: Any = None) -> int:
    """Persist an in-memory Gold DataFrame to gold_machine_hourly_feature (truncate + reload).

    Designed to be called from a Jupyter notebook after build_gold_from_telemetry().
    Returns the number of rows inserted.
    """
    from indusense.db.session import SessionLocal

    if session_factory is None:
        session_factory = SessionLocal

    with session_factory() as session:
        machine_map: dict[int, str] = {
            row.machine_id: row.machine_code
            for row in session.execute(sa.select(Machine.machine_id, Machine.machine_code)).all()
        }
        code_to_id: dict[str, int] = {v: k for k, v in machine_map.items()}
        n = _insert_gold_df(session, gold_df, code_to_id)
        session.commit()

    logger.info("persist_gold_to_db: {} rows committed", n)
    return n
