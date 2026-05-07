"""Project settings and database URL helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.engine import URL


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DOCKER_ENV_PATH = PROJECT_ROOT / ".docker" / ".env"


class DatabaseSettings(BaseModel):
    """Database connection settings loaded from environment files."""

    model_config = ConfigDict(str_strip_whitespace=True)

    db_name: str = Field(alias="DB_NAME", min_length=1)
    db_user: str = Field(alias="DB_USER", min_length=1)
    db_password: str = Field(alias="DB_PASSWORD", min_length=1)
    host: str = Field(alias="HOST", min_length=1)
    port: int = Field(alias="PORT", ge=1, le=65535)

    def sqlalchemy_url(self, driver: str = "postgresql+psycopg") -> str:
        """Build a SQLAlchemy URL for PostgreSQL."""
        return URL.create(
            drivername=driver,
            username=self.db_user,
            password=self.db_password,
            host=self.host,
            port=self.port,
            database=self.db_name,
        ).render_as_string(hide_password=False)

    def masked_sqlalchemy_url(self, driver: str = "postgresql+psycopg") -> str:
        """Build a SQLAlchemy URL with the password masked."""
        return (
            f"{driver}://{self.db_user}:***"
            f"@{self.host}:{self.port}/{self.db_name}"
        )


def _load_env_file(env_path: Path | None = None) -> Path | None:
    selected_path = env_path or DEFAULT_ENV_PATH
    if selected_path.exists():
        load_dotenv(selected_path, override=False)
        return selected_path
    if env_path is None and DOCKER_ENV_PATH.exists():
        load_dotenv(DOCKER_ENV_PATH, override=False)
        return DOCKER_ENV_PATH
    return None


@lru_cache(maxsize=1)
def get_database_settings(env_path: str | Path | None = None) -> DatabaseSettings:
    """Load database settings from `.env`, with a fallback to `.docker/.env`."""
    selected_path = Path(env_path) if env_path else None
    loaded_path = _load_env_file(selected_path)

    if loaded_path is not None:
        raw_values = dotenv_values(loaded_path)
    else:
        raw_values = {}

    try:
        return DatabaseSettings.model_validate(raw_values)
    except ValidationError:
        # Retry from process environment if values were already exported.
        import os

        return DatabaseSettings.model_validate(os.environ)


def build_database_url(env_path: str | Path | None = None) -> str:
    """Return the SQLAlchemy URL computed from project settings."""
    return get_database_settings(env_path).sqlalchemy_url()
