"""Load GoldMachineHourlyFeature as a pandas DataFrame via SQLAlchemy."""

from __future__ import annotations

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from indusense.db.models import GoldMachineHourlyFeature, Machine
from indusense.db.session import create_postgres_engine


def load_gold_from_db(engine: Engine | None = None) -> pd.DataFrame:
    """Query gold_machine_hourly_feature joined with machine and return a DataFrame.

    The returned DataFrame is column-compatible with the in-memory Gold built by
    build_gold_from_telemetry(): machine_id_std contains the machine code (e.g.
    MACH-01), window_start/window_end are timezone-naive UTC, split_set is a
    plain string ("train" / "validation" / "test").

    Pass an explicit engine to override the default (useful in notebooks to reuse
    an already-created connection pool).
    """
    if engine is None:
        engine = create_postgres_engine()

    query = (
        sa.select(
            Machine.machine_code.label("machine_id_std"),
            GoldMachineHourlyFeature.window_start,
            GoldMachineHourlyFeature.window_end,
            GoldMachineHourlyFeature.split_set,
            # Temperature rolling
            GoldMachineHourlyFeature.temp_mean_6h,
            GoldMachineHourlyFeature.temp_max_6h,
            GoldMachineHourlyFeature.temp_std_6h,
            GoldMachineHourlyFeature.temp_mean_12h,
            GoldMachineHourlyFeature.temp_max_12h,
            GoldMachineHourlyFeature.temp_std_12h,
            GoldMachineHourlyFeature.temp_mean_24h,
            GoldMachineHourlyFeature.temp_max_24h,
            GoldMachineHourlyFeature.temp_std_24h,
            # Pressure rolling
            GoldMachineHourlyFeature.pressure_mean_6h,
            GoldMachineHourlyFeature.pressure_max_6h,
            GoldMachineHourlyFeature.pressure_std_6h,
            GoldMachineHourlyFeature.pressure_mean_12h,
            GoldMachineHourlyFeature.pressure_max_12h,
            GoldMachineHourlyFeature.pressure_std_12h,
            GoldMachineHourlyFeature.pressure_mean_24h,
            GoldMachineHourlyFeature.pressure_max_24h,
            GoldMachineHourlyFeature.pressure_std_24h,
            # Voltage rolling
            GoldMachineHourlyFeature.voltage_mean_6h,
            GoldMachineHourlyFeature.voltage_std_6h,
            GoldMachineHourlyFeature.voltage_mean_12h,
            GoldMachineHourlyFeature.voltage_std_12h,
            GoldMachineHourlyFeature.voltage_mean_24h,
            GoldMachineHourlyFeature.voltage_std_24h,
            # Rotation rolling
            GoldMachineHourlyFeature.rotation_mean_6h,
            GoldMachineHourlyFeature.rotation_std_6h,
            GoldMachineHourlyFeature.rotation_mean_12h,
            GoldMachineHourlyFeature.rotation_std_12h,
            GoldMachineHourlyFeature.rotation_mean_24h,
            GoldMachineHourlyFeature.rotation_std_24h,
            # Trend + 24h anomaly
            GoldMachineHourlyFeature.temp_trend_6h,
            GoldMachineHourlyFeature.pressure_trend_6h,
            GoldMachineHourlyFeature.temp_zscore_24h,
            # Short-lag delta features
            GoldMachineHourlyFeature.temp_delta_1h,
            GoldMachineHourlyFeature.temp_delta_3h,
            GoldMachineHourlyFeature.pressure_delta_1h,
            GoldMachineHourlyFeature.pressure_delta_3h,
            GoldMachineHourlyFeature.rotation_delta_1h,
            GoldMachineHourlyFeature.rotation_delta_3h,
            GoldMachineHourlyFeature.voltage_delta_1h,
            # Per-machine z-scores
            GoldMachineHourlyFeature.temp_zscore_machine,
            GoldMachineHourlyFeature.pressure_zscore_machine,
            # Production / maintenance
            GoldMachineHourlyFeature.pieces_produced_sum_24h,
            GoldMachineHourlyFeature.capacity_utilization_pct,
            GoldMachineHourlyFeature.days_since_last_maintenance,
            GoldMachineHourlyFeature.maintenance_count_prev_30d,
            # Incident lookback
            GoldMachineHourlyFeature.incident_count_prev_24h,
            GoldMachineHourlyFeature.incident_max_severity_prev_24h,
            GoldMachineHourlyFeature.incident_count_prev_7d,
            GoldMachineHourlyFeature.hours_since_last_incident,
            # Incident type counts
            GoldMachineHourlyFeature.type_surchauffe_count_prev_24h,
            GoldMachineHourlyFeature.type_baisse_pression_count_prev_24h,
            GoldMachineHourlyFeature.type_vibration_count_prev_24h,
            GoldMachineHourlyFeature.type_bruit_mecanique_count_prev_24h,
            GoldMachineHourlyFeature.type_surconsommation_count_prev_24h,
            GoldMachineHourlyFeature.type_blocage_mecanique_count_prev_24h,
            GoldMachineHourlyFeature.type_alarme_capteur_count_prev_24h,
            GoldMachineHourlyFeature.type_arret_urgence_count_prev_24h,
            GoldMachineHourlyFeature.type_defaut_qualite_count_prev_24h,
            # Ambient weather
            GoldMachineHourlyFeature.ambient_temp_c,
            GoldMachineHourlyFeature.ambient_humidity_pct,
            GoldMachineHourlyFeature.ambient_pressure_hpa,
            # Labels
            GoldMachineHourlyFeature.label_failure_next_6h,
            GoldMachineHourlyFeature.label_failure_next_12h,
            GoldMachineHourlyFeature.label_failure_next_24h,
            GoldMachineHourlyFeature.label_failure_next_48h,
        )
        .join(Machine, GoldMachineHourlyFeature.machine_id == Machine.machine_id)
        .order_by(GoldMachineHourlyFeature.machine_id, GoldMachineHourlyFeature.window_start)
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    df["window_start"] = pd.to_datetime(df["window_start"]).dt.tz_localize(None)
    df["window_end"] = pd.to_datetime(df["window_end"]).dt.tz_localize(None)
    # PostgreSQL enum → string plain
    df["split_set"] = df["split_set"].astype(str)

    return df


__all__ = ["load_gold_from_db"]
