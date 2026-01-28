from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://localhost/edgebet"

    # External APIs
    THE_ODDS_API_KEY: str = ""  # Get from https://the-odds-api.com

    # CORS
    CORS_ORIGINS: str = '["http://localhost:3000"]'

    # Environment
    ENVIRONMENT: str = "development"

    # Model paths
    MODEL_DIR: str = "models"

    @property
    def database_url_with_pgbouncer_fix(self) -> str:
        """Add pgbouncer compatibility parameter to DATABASE_URL."""
        url = self.DATABASE_URL
        if "prepared_statement_cache_size" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}prepared_statement_cache_size=0"
        return url

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.CORS_ORIGINS)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
