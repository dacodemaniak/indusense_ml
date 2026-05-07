"""Pydantic validation schemas for source ingestion."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def normalize_machine_code(value: str) -> str:
    digits = re.findall(r"\d+", value)
    if not digits:
        raise ValueError("machine_id does not contain digits")
    return f"MACH-{int(digits[-1]):02d}"


class TemperatureInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    machine_id_raw: str = Field(min_length=1)
    timestamp_raw: str = Field(min_length=1)
    temperature_raw: str | float | int | Decimal | None = None

    machine_code: str | None = None
    observed_at: datetime | None = None
    temperature: Decimal | None = None

    @model_validator(mode="after")
    def populate_derived_fields(self):
        self.machine_code = normalize_machine_code(self.machine_id_raw)
        self.observed_at = datetime.fromisoformat(self.timestamp_raw.replace("T", " ").replace("Z", ""))
        if self.temperature_raw in (None, ""):
            self.temperature = None
        else:
            self.temperature = Decimal(str(self.temperature_raw))
        return self


class PressureInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    machine_id_raw: str = Field(min_length=1)
    timestamp_raw: str = Field(min_length=1)
    pressure_raw: str | float | int | Decimal | None = None

    machine_code: str | None = None
    observed_at: datetime | None = None
    pressure_bar: Decimal | None = None

    @model_validator(mode="after")
    def populate_derived_fields(self):
        self.machine_code = normalize_machine_code(self.machine_id_raw)
        cleaned_timestamp = re.sub(r"(?:Z|[+-]\d{2}:\d{2})$", "", self.timestamp_raw.replace("T", " "))
        self.observed_at = datetime.fromisoformat(cleaned_timestamp)
        if self.pressure_raw in (None, ""):
            self.pressure_bar = None
        else:
            self.pressure_bar = Decimal(str(self.pressure_raw))
        return self


class IncidentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    incident_code_raw: str = Field(min_length=1)
    machine_id_raw: str = Field(min_length=1)
    date_raw: str = Field(min_length=1)
    time_raw: str = Field(min_length=1)
    severity_raw: str | int = Field(min_length=1)
    operator_name_raw: str | None = None
    operator_badge_raw: str | None = None
    shift_raw: str | None = None
    comment_raw: str | None = None

    incident_code: str | None = None
    machine_code: str | None = None
    occurred_at: datetime | None = None
    severity: int | None = None

    @field_validator("incident_code_raw")
    @classmethod
    def uppercase_incident_code(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def populate_derived_fields(self):
        self.incident_code = self.incident_code_raw
        self.machine_code = normalize_machine_code(self.machine_id_raw)
        self.occurred_at = datetime.fromisoformat(f"{self.date_raw} {self.time_raw}")
        self.severity = int(self.severity_raw)
        if not 1 <= self.severity <= 5:
            raise ValueError("incident severity must be between 1 and 5")
        return self
