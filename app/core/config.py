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

    # CORS — comma-separated allowed origins (leave empty to default to BACKEND_BASE_URL)
    # Example: "https://app.example.com,http://localhost:3000"
    ALLOWED_ORIGINS: str = ""

    # Backend Config — REQUIRED.  The service cannot operate without this.
    BACKEND_BASE_URL: str = ""

    # Redis Config — primary URL or individual parts (Railway exposes both)
    REDIS_URL: str = ""
    REDIS_PUBLIC_URL: str = ""   # Railway public proxy URL (fallback)
    REDISHOST: str = ""
    REDISPORT: int = 6379
    REDISUSER: str = "default"
    REDISPASSWORD: str = ""

    # Rate limiting — requests per minute per user_id (0 = disabled)
    RATE_LIMIT_RPM: int = 30

    # Request timeout — seconds to wait for Agent.run() before 504
    REQUEST_TIMEOUT_SECONDS: int = 60

    # OpenRouter per-call timeout — seconds before we abandon a single LLM call.
    # Must be < REQUEST_TIMEOUT_SECONDS to leave room for fallback chain.
    OPENROUTER_TIMEOUT_SECONDS: float = 45.0

    # Backend (.NET) HTTP call timeout — per request
    BACKEND_TIMEOUT_SECONDS: float = 30.0

    # Backend circuit breaker — opens after this many consecutive failures,
    # stays open for BACKEND_BREAKER_RESET_SECONDS, then half-open trial.
    BACKEND_BREAKER_FAIL_THRESHOLD: int = 5
    BACKEND_BREAKER_RESET_SECONDS: float = 30.0

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

