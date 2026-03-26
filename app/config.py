from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Text-to-Speech Service"
    app_version: str = "0.1.0"
    port: int = Field(default=8000, alias="PORT")
    tts_provider: str = Field(default="openai", alias="TTS_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_tts_model: str = Field(default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", alias="OPENAI_TTS_MODEL")
    openai_tts_voice: str | None = Field(default=None, alias="OPENAI_TTS_VOICE")
    openai_analysis_model: str = Field(default="openai/gpt-oss-120b", alias="OPENAI_ANALYSIS_MODEL")
    openai_stt_model: str = Field(default="openai/whisper-large-v3-turbo", alias="OPENAI_STT_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_timeout_seconds: float = Field(default=120.0, alias="OPENAI_TIMEOUT_SECONDS")
    openai_max_retries: int = Field(default=0, alias="OPENAI_MAX_RETRIES")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    analysis_retry_on_invalid_json: bool = Field(default=False, alias="ANALYSIS_RETRY_ON_INVALID_JSON")
    ffmpeg_path: str = Field(default="ffmpeg", alias="FFMPEG_PATH")
    ffprobe_path: str = Field(default="ffprobe", alias="FFPROBE_PATH")
    stt_chunk_duration_seconds: int = Field(default=300, alias="STT_CHUNK_DURATION_SECONDS")
    stt_max_parallel_chunks: int = Field(default=2, alias="STT_MAX_PARALLEL_CHUNKS")
    max_text_length: int = Field(default=5000, alias="MAX_TEXT_LENGTH")


@lru_cache
def get_settings() -> Settings:
    return Settings()
