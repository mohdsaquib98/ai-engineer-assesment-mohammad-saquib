from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    GROQ_API_KEY: str
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    SUPERHERO_API_TOKEN: str

    DATASET_PATH: str = "data/facts.json"
    DATASET_SEARCH_THRESHOLD: float = 0.1
    DATASET_TOP_K: int = 3

    MIN_QUESTION_LENGTH: int = 2
    MAX_QUESTION_LENGTH: int = 1000

    SUPERHERO_API_TIMEOUT: float = 5.0
    SUPERHERO_API_MAX_FAILURES: int = 5


settings = Settings()
