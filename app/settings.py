from typing import ClassVar

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cors_origins() -> list[str]:
    return ["http://localhost:8000"]


MIN_API_TOKEN_LENGTH = 32


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
    WORKER_TIMEOUT_SECONDS: float = 10
    WORKER_STUCK_SECONDS: float = 30
    WORKER_QUEUE_SIZE: int = 8
    WATCHDOG_INTERVAL_SECONDS: float = 5
    STRIKE_LIMIT: int = 3
    MIN_JUDGMENTS: int = Field(default=3, ge=0)
    # Sent to Devpost when resolving submission URLs, so private submissions resolve.
    DEVPOST_COOKIE: str = ""
    DEVPOST_CONCURRENCY: int = Field(default=4, ge=1)

    @field_validator("API_TOKEN")
    @classmethod
    def require_strong_token(cls, token: str) -> str:
        # Without a token every request is rejected, so fail at startup instead.
        if len(token) < MIN_API_TOKEN_LENGTH:
            raise ValueError(
                f"MDREDD_API_TOKEN must be set to at least {MIN_API_TOKEN_LENGTH} "
                "characters, e.g. the output of `openssl rand -hex 32`"
            )
        return token


settings = Settings()
