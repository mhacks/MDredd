from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_prefix="MDREDD_",
    )

    DB_FILE: str = "mdredd.db"
    ENABLE_CRASH_ROUTE: bool = False
    API_TOKEN: str = ""
    PAIR_CAPACITY: int = 6
    PAIR_REFILL_PER_SECOND: float = 0.2
    SUBMIT_CAPACITY: int = 2
    SUBMIT_REFILL_PER_SECOND: float = 1 / 60
    ADMIN_CAPACITY: int = 4
    ADMIN_REFILL_PER_SECOND: float = 1 / 30


settings = Settings()
