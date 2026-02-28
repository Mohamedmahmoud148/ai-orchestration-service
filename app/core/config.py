from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    PORT: int = 8000
    
    # AI Config
    GEMINI_API_KEY: str = ""
    
    # Backend Config
    DOTNET_BACKEND_URL: str = ""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()
