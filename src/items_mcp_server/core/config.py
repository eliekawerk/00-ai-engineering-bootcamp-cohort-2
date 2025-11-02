from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPEN_API_KEY: str

    model_config_dict = SettingsConfigDict(env_file=".env")


config = Config()
