"""Initial Bronze/Silver/Gold relational model."""

from __future__ import annotations

from datetime import date

from alembic import op
import sqlalchemy as sa


revision = "20260507_0001"
down_revision = None
branch_labels = None
depends_on = None


ingestion_status = sa.Enum(
    "pending",
    "running",
    "completed",
    "failed",
    "partial",
    name="ingestion_status",
    create_type=False,
)
data_quality_severity = sa.Enum(
    "info",
    "warning",
    "error",
    name="data_quality_severity",
    create_type=False,
)
sensor_type = sa.Enum(
    "temperature",
    "pressure",
    name="sensor_type",
    create_type=False,
)
split_set = sa.Enum(
    "train",
    "validation",
    "test",
    "holdout",
    name="split_set",
    create_type=False,
)


def upgrade() -> None:
    op.create_table(
        "ingestion_batch",
        sa.Column("ingestion_batch_id", sa.Uuid(), nullable=False),
        sa.Column("source_name", sa.String(length=64), nullable=False),
        sa.Column("source_file", sa.String(length=255), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rows_read", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rows_loaded", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rows_rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", ingestion_status, nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("ingestion_batch_id", name=op.f("pk_ingestion_batch")),
    )

    op.create_table(
        "machine",
        sa.Column("machine_id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("machine_code", sa.String(length=16), nullable=False),
        sa.Column("commissioning_date", sa.Date(), nullable=True),
        sa.Column("max_daily_capacity", sa.Integer(), nullable=True),
        sa.Column("model", sa.String(length=32), nullable=True),
        sa.Column("production_line", sa.String(length=16), nullable=True),
        sa.Column("location", sa.String(length=16), nullable=True),
        sa.Column("criticality", sa.String(length=8), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("criticality IN ('LOW', 'MEDIUM', 'HIGH')", name=op.f("ck_machine_machine_criticality_allowed")),
        sa.CheckConstraint("machine_code ~ '^MACH-[0-9]{2}$'", name=op.f("ck_machine_machine_code_format")),
        sa.CheckConstraint("max_daily_capacity > 0", name=op.f("ck_machine_machine_capacity_positive")),
        sa.PrimaryKeyConstraint("machine_id", name=op.f("pk_machine")),
        sa.UniqueConstraint("machine_code", name=op.f("uq_machine_machine_code")),
    )

    op.create_table(
        "operator",
        sa.Column("operator_id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("operator_key", sa.String(length=64), nullable=False),
        sa.Column("badge_hash", sa.String(length=128), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("operator_id", name=op.f("pk_operator")),
        sa.UniqueConstraint("operator_key", name=op.f("uq_operator_operator_key")),
    )

    op.create_table(
        "bronze_temperature_raw",
        sa.Column("temperature_raw_id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("ingestion_batch_id", sa.Uuid(), nullable=False),
        sa.Column("row_number", sa.Integer(), nullable=False),
        sa.Column("machine_id_raw", sa.String(length=64), nullable=True),
        sa.Column("timestamp_raw", sa.String(length=128), nullable=True),
        sa.Column("temperature_raw", sa.String(length=64), nullable=True),
        sa.Column("parse_ok", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("rejected_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["ingestion_batch_id"], ["ingestion_batch.ingestion_batch_id"], name=op.f("fk_bronze_temperature_raw_ingestion_batch_id_ingestion_batch"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("temperature_raw_id", name=op.f("pk_bronze_temperature_raw")),
    )

    op.create_table(
        "bronze_pressure_raw",
        sa.Column("pressure_raw_id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("ingestion_batch_id", sa.Uuid(), nullable=False),
        sa.Column("row_number", sa.Integer(), nullable=False),
        sa.Column("machine_id_raw", sa.String(length=64), nullable=True),
        sa.Column("timestamp_raw", sa.String(length=128), nullable=True),
        sa.Column("pressure_raw", sa.String(length=64), nullable=True),
        sa.Column("parse_ok", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("rejected_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["ingestion_batch_id"], ["ingestion_batch.ingestion_batch_id"], name=op.f("fk_bronze_pressure_raw_ingestion_batch_id_ingestion_batch"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("pressure_raw_id", name=op.f("pk_bronze_pressure_raw")),
    )

    op.create_table(
        "bronze_incident_raw",
        sa.Column("incident_raw_id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("ingestion_batch_id", sa.Uuid(), nullable=False),
        sa.Column("row_number", sa.Integer(), nullable=False),
        sa.Column("incident_code_raw", sa.String(length=64), nullable=True),
        sa.Column("machine_id_raw", sa.String(length=64), nullable=True),
        sa.Column("operator_name_raw", sa.String(length=128), nullable=True),
        sa.Column("operator_badge_raw", sa.String(length=64), nullable=True),
        sa.Column("occurred_at_raw", sa.String(length=128), nullable=True),
        sa.Column("severity_raw", sa.String(length=16), nullable=True),
        sa.Column("shift_raw", sa.String(length=32), nullable=True),
        sa.Column("comment_raw", sa.Text(), nullable=True),
        sa.Column("parse_ok", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("rejected_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["ingestion_batch_id"], ["ingestion_batch.ingestion_batch_id"], name=op.f("fk_bronze_incident_raw_ingestion_batch_id_ingestion_batch"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("incident_raw_id", name=op.f("pk_bronze_incident_raw")),
    )

    op.create_table(
        "data_quality_issue",
        sa.Column("dq_issue_id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("ingestion_batch_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_name", sa.String(length=64), nullable=False),
        sa.Column("rule_code", sa.String(length=64), nullable=False),
        sa.Column("severity", data_quality_severity, nullable=False),
        sa.Column("entity_key", sa.String(length=128), nullable=True),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["ingestion_batch_id"], ["ingestion_batch.ingestion_batch_id"], name=op.f("fk_data_quality_issue_ingestion_batch_id_ingestion_batch"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("dq_issue_id", name=op.f("pk_data_quality_issue")),
    )

    op.create_table(
        "silver_sensor_reading",
        sa.Column("sensor_reading_id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("machine_id", sa.BigInteger(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sensor_type", sensor_type, nullable=False),
        sa.Column("sensor_value", sa.Numeric(precision=12, scale=3), nullable=True),
        sa.Column("unit", sa.String(length=16), nullable=False),
        sa.Column("is_missing", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_duplicate", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_outlier", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("ingestion_batch_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["ingestion_batch_id"], ["ingestion_batch.ingestion_batch_id"], name=op.f("fk_silver_sensor_reading_ingestion_batch_id_ingestion_batch"), ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["machine_id"], ["machine.machine_id"], name=op.f("fk_silver_sensor_reading_machine_id_machine"), ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("sensor_reading_id", name=op.f("pk_silver_sensor_reading")),
        sa.UniqueConstraint("machine_id", "observed_at", "sensor_type", name="silver_sensor_observation"),
    )

    op.create_table(
        "silver_incident",
        sa.Column("incident_id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("incident_code", sa.String(length=64), nullable=False),
        sa.Column("machine_id", sa.BigInteger(), nullable=False),
        sa.Column("operator_id", sa.BigInteger(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("severity", sa.SmallInteger(), nullable=False),
        sa.Column("shift", sa.String(length=32), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("is_label_event", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("ingestion_batch_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("severity BETWEEN 1 AND 5", name=op.f("ck_silver_incident_silver_incident_severity_range")),
        sa.ForeignKeyConstraint(["ingestion_batch_id"], ["ingestion_batch.ingestion_batch_id"], name=op.f("fk_silver_incident_ingestion_batch_id_ingestion_batch"), ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["machine_id"], ["machine.machine_id"], name=op.f("fk_silver_incident_machine_id_machine"), ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["operator_id"], ["operator.operator_id"], name=op.f("fk_silver_incident_operator_id_operator"), ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("incident_id", name=op.f("pk_silver_incident")),
        sa.UniqueConstraint("incident_code", name=op.f("uq_silver_incident_incident_code")),
    )

    op.create_table(
        "gold_machine_hourly_feature",
        sa.Column("feature_row_id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("machine_id", sa.BigInteger(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("temp_mean_24h", sa.Numeric(precision=12, scale=3), nullable=True),
        sa.Column("temp_max_24h", sa.Numeric(precision=12, scale=3), nullable=True),
        sa.Column("temp_std_24h", sa.Numeric(precision=12, scale=3), nullable=True),
        sa.Column("pressure_mean_24h", sa.Numeric(precision=12, scale=3), nullable=True),
        sa.Column("pressure_std_24h", sa.Numeric(precision=12, scale=3), nullable=True),
        sa.Column("incident_count_prev_24h", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("incident_max_severity_prev_24h", sa.SmallInteger(), nullable=True),
        sa.Column("label_failure_next_24h", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("split_set", split_set, nullable=False, server_default="train"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("window_end > window_start", name=op.f("ck_gold_machine_hourly_feature_gold_window_order")),
        sa.ForeignKeyConstraint(["machine_id"], ["machine.machine_id"], name=op.f("fk_gold_machine_hourly_feature_machine_id_machine"), ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("feature_row_id", name=op.f("pk_gold_machine_hourly_feature")),
        sa.UniqueConstraint("machine_id", "window_start", name="gold_machine_window_start"),
    )

    op.create_index(op.f("ix_machine_location"), "machine", ["location"], unique=False)
    op.create_index(op.f("ix_machine_production_line"), "machine", ["production_line"], unique=False)
    op.create_index(op.f("ix_silver_sensor_reading_machine_id"), "silver_sensor_reading", ["machine_id"], unique=False)
    op.create_index(op.f("ix_silver_sensor_reading_observed_at"), "silver_sensor_reading", ["observed_at"], unique=False)
    op.create_index(op.f("ix_silver_incident_machine_id"), "silver_incident", ["machine_id"], unique=False)
    op.create_index(op.f("ix_silver_incident_occurred_at"), "silver_incident", ["occurred_at"], unique=False)
    op.create_index(op.f("ix_gold_machine_hourly_feature_machine_id"), "gold_machine_hourly_feature", ["machine_id"], unique=False)
    op.create_index(op.f("ix_gold_machine_hourly_feature_window_start"), "gold_machine_hourly_feature", ["window_start"], unique=False)

    op.bulk_insert(
        sa.table(
            "machine",
            sa.column("machine_code", sa.String()),
            sa.column("commissioning_date", sa.Date()),
            sa.column("max_daily_capacity", sa.Integer()),
            sa.column("model", sa.String()),
            sa.column("production_line", sa.String()),
            sa.column("location", sa.String()),
            sa.column("criticality", sa.String()),
            sa.column("is_active", sa.Boolean()),
        ),
        [
            {"machine_code": "MACH-01", "commissioning_date": date(2021, 5, 12), "max_daily_capacity": 770, "model": "InduPress-X2", "production_line": "Ligne-A", "location": "Atelier-2", "criticality": "MEDIUM", "is_active": True},
            {"machine_code": "MACH-02", "commissioning_date": date(2024, 9, 7), "max_daily_capacity": 800, "model": "InduPress-X2", "production_line": "Ligne-A", "location": "Atelier-1", "criticality": "LOW", "is_active": True},
            {"machine_code": "MACH-03", "commissioning_date": date(2019, 7, 23), "max_daily_capacity": 1405, "model": "InduPress-X1", "production_line": "Ligne-B", "location": "Atelier-1", "criticality": "HIGH", "is_active": True},
            {"machine_code": "MACH-04", "commissioning_date": date(2023, 1, 7), "max_daily_capacity": 750, "model": "InduPress-Z1", "production_line": "Ligne-C", "location": "Atelier-3", "criticality": "LOW", "is_active": True},
            {"machine_code": "MACH-05", "commissioning_date": date(2024, 3, 11), "max_daily_capacity": 1380, "model": "InduPress-X2", "production_line": "Ligne-C", "location": "Atelier-1", "criticality": "HIGH", "is_active": True},
            {"machine_code": "MACH-06", "commissioning_date": date(2022, 1, 1), "max_daily_capacity": 1351, "model": "InduPress-X3", "production_line": "Ligne-A", "location": "Atelier-1", "criticality": "LOW", "is_active": True},
            {"machine_code": "MACH-07", "commissioning_date": date(2025, 5, 25), "max_daily_capacity": 1428, "model": "InduPress-Z1", "production_line": "Ligne-A", "location": "Atelier-3", "criticality": "MEDIUM", "is_active": True},
            {"machine_code": "MACH-08", "commissioning_date": date(2023, 10, 18), "max_daily_capacity": 1158, "model": "InduPress-X1", "production_line": "Ligne-B", "location": "Atelier-2", "criticality": "HIGH", "is_active": True},
            {"machine_code": "MACH-09", "commissioning_date": date(2023, 1, 15), "max_daily_capacity": 1056, "model": "InduPress-X2", "production_line": "Ligne-B", "location": "Atelier-3", "criticality": "MEDIUM", "is_active": True},
            {"machine_code": "MACH-10", "commissioning_date": date(2021, 4, 16), "max_daily_capacity": 778, "model": "InduPress-X3", "production_line": "Ligne-A", "location": "Atelier-3", "criticality": "LOW", "is_active": True},
            {"machine_code": "MACH-11", "commissioning_date": date(2022, 9, 15), "max_daily_capacity": 984, "model": "InduPress-X2", "production_line": "Ligne-B", "location": "Atelier-1", "criticality": "MEDIUM", "is_active": True},
            {"machine_code": "MACH-12", "commissioning_date": date(2024, 2, 21), "max_daily_capacity": 838, "model": "InduPress-Z1", "production_line": "Ligne-B", "location": "Atelier-2", "criticality": "MEDIUM", "is_active": True},
            {"machine_code": "MACH-13", "commissioning_date": date(2019, 12, 30), "max_daily_capacity": 907, "model": "InduPress-X3", "production_line": "Ligne-C", "location": "Atelier-3", "criticality": "MEDIUM", "is_active": True},
            {"machine_code": "MACH-14", "commissioning_date": date(2021, 10, 21), "max_daily_capacity": 1191, "model": "InduPress-X2", "production_line": "Ligne-A", "location": "Atelier-3", "criticality": "LOW", "is_active": True},
            {"machine_code": "MACH-15", "commissioning_date": date(2022, 3, 16), "max_daily_capacity": 1027, "model": "InduPress-X2", "production_line": "Ligne-A", "location": "Atelier-3", "criticality": "MEDIUM", "is_active": True},
        ],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_gold_machine_hourly_feature_window_start"), table_name="gold_machine_hourly_feature")
    op.drop_index(op.f("ix_gold_machine_hourly_feature_machine_id"), table_name="gold_machine_hourly_feature")
    op.drop_index(op.f("ix_silver_incident_occurred_at"), table_name="silver_incident")
    op.drop_index(op.f("ix_silver_incident_machine_id"), table_name="silver_incident")
    op.drop_index(op.f("ix_silver_sensor_reading_observed_at"), table_name="silver_sensor_reading")
    op.drop_index(op.f("ix_silver_sensor_reading_machine_id"), table_name="silver_sensor_reading")
    op.drop_index(op.f("ix_machine_production_line"), table_name="machine")
    op.drop_index(op.f("ix_machine_location"), table_name="machine")

    op.drop_table("gold_machine_hourly_feature")
    op.drop_table("silver_incident")
    op.drop_table("silver_sensor_reading")
    op.drop_table("data_quality_issue")
    op.drop_table("bronze_incident_raw")
    op.drop_table("bronze_pressure_raw")
    op.drop_table("bronze_temperature_raw")
    op.drop_table("operator")
    op.drop_table("machine")
    op.drop_table("ingestion_batch")

    op.execute("DROP TYPE IF EXISTS split_set")
    op.execute("DROP TYPE IF EXISTS sensor_type")
    op.execute("DROP TYPE IF EXISTS data_quality_severity")
    op.execute("DROP TYPE IF EXISTS ingestion_status")
