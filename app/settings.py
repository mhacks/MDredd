from typing import ClassVar, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cors_origins() -> list[str]:
    return ["http://localhost:8000"]


class ApiKey(BaseModel):
    key_hash: str
    user_id: str
    role: Literal["admin", "judge"]
    pair_capacity: int | None = None
    pair_refill_per_second: float | None = None
    submit_capacity: int | None = None
    submit_refill_per_second: float | None = None
    admin_capacity: int | None = None
    admin_refill_per_second: float | None = None


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_prefix="MDREDD_",
    )

    DB_FILE: str = "mdredd.db"
    SNAPSHOT_INTERVAL: int = 10
    MAX_SNAPSHOTS: int = 10
    ENABLE_CRASH_ROUTE: bool = False
    API_KEYS: list[ApiKey] = Field(default_factory=list)
    CORS_ORIGINS: list[str] = Field(default_factory=_default_cors_origins)
    PAIR_CAPACITY: int = 6
    PAIR_REFILL_PER_SECOND: float = 0.2
    SUBMIT_CAPACITY: int = 2
    SUBMIT_REFILL_PER_SECOND: float = 1 / 60
    ADMIN_CAPACITY: int = 4
    ADMIN_REFILL_PER_SECOND: float = 1 / 30


settings = Settings()
