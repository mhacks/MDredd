from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MDREDD_",
    )

    DB_FILE: str = "mdredd.db"
    SNAPSHOT_INTERVAL: int = 10
    MAX_SNAPSHOTS: int = 10


settings = Settings()
