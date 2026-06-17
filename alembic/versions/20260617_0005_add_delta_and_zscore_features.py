"""Add short-lag delta features and per-machine z-score columns to gold_machine_hourly_feature."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260617_0005"
down_revision = "20260611_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for col in (
        "temp_delta_1h",
        "temp_delta_3h",
        "pressure_delta_1h",
        "pressure_delta_3h",
        "rotation_delta_1h",
        "rotation_delta_3h",
        "voltage_delta_1h",
    ):
        op.add_column(
            "gold_machine_hourly_feature",
            sa.Column(col, sa.Numeric(10, 3), nullable=True),
        )

    for col in ("temp_zscore_machine", "pressure_zscore_machine"):
        op.add_column(
            "gold_machine_hourly_feature",
            sa.Column(col, sa.Numeric(8, 4), nullable=True),
        )


def downgrade() -> None:
    for col in (
        "temp_delta_1h",
        "temp_delta_3h",
        "pressure_delta_1h",
        "pressure_delta_3h",
        "rotation_delta_1h",
        "rotation_delta_3h",
        "voltage_delta_1h",
        "temp_zscore_machine",
        "pressure_zscore_machine",
    ):
        op.drop_column("gold_machine_hourly_feature", col)
