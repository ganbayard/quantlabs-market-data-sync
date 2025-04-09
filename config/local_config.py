from pydantic import BaseSettings
from typing import Optional

class LocalConfig(BaseSettings):
    """Local development database configuration"""
    host: str
    port: int
    database: str
    user: str
    password: str

    class Config:
        env_prefix = "LOCAL_DB_"
        env_file = ".env" 