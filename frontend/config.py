"""
Frontend configuration — loaded once at startup.
All frontend modules import `settings` from here.
"""
import os
from pydantic_settings import BaseSettings


class FrontendSettings(BaseSettings):
    API_BASE_URL: str = "http://localhost:8000/api/v1"
    API_TIMEOUT: float = 30.0
    HEALTH_POLL_INTERVAL: int = 30   # seconds
    CHAT_POLL_INTERVAL: int = 2       # seconds
    MAX_FILE_SIZE_MB: int = 50
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = FrontendSettings()
