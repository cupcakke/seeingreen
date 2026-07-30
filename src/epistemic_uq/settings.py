from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EUQ_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./var/epistemic_uq.db"
    artifact_root: Path = Path("./var/artifacts")
    api_keys: str = "local-development-key"
    redis_url: str = "redis://localhost:6379/0"
    rate_limit_per_minute: int = Field(default=120, ge=1)
    log_level: str = "INFO"
    redact_patterns: str = "email,phone,credit_card"
    default_config: Path = Path("./config/default.yaml")
    allowed_data_root: Path = Path("./data")

    @property
    def api_key_set(self) -> set[str]:
        return {value.strip() for value in self.api_keys.split(",") if value.strip()}

    @property
    def redact_pattern_set(self) -> set[str]:
        return {value.strip() for value in self.redact_patterns.split(",") if value.strip()}

    def load_pipeline_config(self, path: Path | None = None) -> dict[str, Any]:
        source = path or self.default_config
        with source.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("Pipeline configuration must be a mapping")
        return loaded


settings = Settings()
