"""SQLAlchemy engine and session helpers."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from indusense.core.settings import build_database_url, get_database_settings


def get_database_url() -> str:
    """Return the configured PostgreSQL URL."""
    return build_database_url()


def create_postgres_engine(echo: bool = False):
    """Create the project SQLAlchemy engine."""
    return create_engine(get_database_url(), echo=echo, future=True)


SessionLocal = sessionmaker(
    bind=create_postgres_engine(),
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


__all__ = [
    "SessionLocal",
    "create_postgres_engine",
    "get_database_settings",
    "get_database_url",
]
