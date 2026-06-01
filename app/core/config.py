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

    # Embeddings — for RAG pipeline.
    # OPENAI_API_KEY: direct OpenAI key for text-embedding-3-small.
    # EMBEDDING_BASE_URL: override to point at any OpenAI-compatible provider.
    # EMBEDDING_MODEL: override to use a different embedding model.
    # If none are set the service falls back to keyword-overlap (no real vectors).
    OPENAI_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

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

    @field_validator("REDISPORT", mode="before")
    @classmethod
    def _coerce_redis_port(cls, v):
        if v == "" or v is None:
            return 6379
        return v
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

    # ── AI Pipeline — Layer 1/2/3 flags ──────────────────────────────────────
    # EMBEDDING_CLASSIFIER_ENABLED:
    #   True  → Layer 1 (embedding fast path) is active.
    #   False → Skip Layer 1, always use LLM classification (old behavior).
    EMBEDDING_CLASSIFIER_ENABLED: bool = True

    # EMBEDDING_CLASSIFIER_SHADOW_MODE:
    #   True  → Embedding classifier runs and logs results but does NOT change
    #            execution path.  Use this for 1-week calibration.
    #   False → Embedding classifier drives the fast path (production mode).
    EMBEDDING_CLASSIFIER_SHADOW_MODE: bool = True

    # EMBEDDING_HIGH_CONFIDENCE_THRESHOLD:
    #   Cosine similarity threshold above which the embedding result is trusted
    #   directly, skipping the LLM call entirely.
    EMBEDDING_HIGH_CONFIDENCE_THRESHOLD: float = 0.82

    # LLM confidence thresholds (function-call classifier, Layer 2)
    LLM_CONFIDENCE_EXECUTE_THRESHOLD: float = 0.78
    LLM_CONFIDENCE_CLARIFY_THRESHOLD: float = 0.55

    # ACTION_GUARD_ENABLED:
    #   True  → Layer 4 critical action guard is active (enrollment confirmation).
    #   False → Write operations execute immediately (old behavior).
    ACTION_GUARD_ENABLED: bool = True

    # OPENROUTER_MODEL: model used for classification (planner) calls.
    OPENROUTER_MODEL: str = "openai/gpt-4o-mini"

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

