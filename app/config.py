from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    api_key: str
    api_base_url: str

    class Config:
        env_file = '.env'