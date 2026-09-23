from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cors_origins() -> list[str]:
    return ["http://localhost:8000"]


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_prefix="MDREDD_",
    )

    DB_FILE: str = "mdredd.db"
    ENABLE_CRASH_ROUTE: bool = False
    API_TOKEN: str = ""
    CORS_ORIGINS: list[str] = Field(default_factory=_default_cors_origins)
    PAIR_CAPACITY: int = 6
    PAIR_REFILL_PER_SECOND: float = 0.2
    SUBMIT_CAPACITY: int = 2
    SUBMIT_REFILL_PER_SECOND: float = 1 / 60
    ADMIN_CAPACITY: int = 4
    ADMIN_REFILL_PER_SECOND: float = 1 / 30


settings = Settings()
