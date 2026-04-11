from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    PORT: int = 8000

    # AI Config — OpenRouter
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_FALLBACK_MODEL_1: str = "openai/gpt-4o-mini"
    OPENROUTER_FALLBACK_MODEL_2: str = ""  # e.g. "mistralai/mistral-7b-instruct"

    # Backend Config — REQUIRED.  The service cannot operate without this.
    BACKEND_BASE_URL: str = ""

    # Redis Config
    REDIS_URL: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("BACKEND_BASE_URL")
    @classmethod
    def _require_backend_url(cls, v: str) -> str:
        """Refuse to start if the .NET backend URL is not configured."""
        if not v or not v.strip():
            raise ValueError(
                "BACKEND_BASE_URL is not set. "
                "Set it in your .env file or environment variables before starting the service. "
                "Example: BACKEND_BASE_URL=http://localhost:5000"
            )
        return v.rstrip("/")   # normalize: strip trailing slash once, at load time

    @field_validator("OPENROUTER_API_KEY")
    @classmethod
    def _warn_openrouter_key(cls, v: str) -> str:
        """OpenRouter key is optional at config load but validated again at startup."""
        return v


settings = Settings()

