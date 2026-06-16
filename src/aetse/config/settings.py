"""
AET-SE Configuration — Pydantic BaseSettings.

All settings are loaded from environment variables / .env file.
Import this module anywhere: `from aetse.config.settings import settings`
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = 3 levels up from this file (src/aetse/config/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class OllamaSettings(BaseSettings):
    """Ollama LLM runtime configuration."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = "http://localhost:11434"
    primary_model: str = "llama3.1:8b-instruct-q4_K_M"
    fallback_model: str = "phi3:mini"
    timeout_seconds: int = 120
    max_retries: int = 3


class EmbeddingSettings(BaseSettings):
    """Sentence-transformer embedding configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model_name: str = "BAAI/bge-small-en-v1.5"
    device: Literal["cpu", "cuda"] = "cpu"


class ChromaSettings(BaseSettings):
    """ChromaDB vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="CHROMA_")

    persist_dir: Path = PROJECT_ROOT / "data" / "chroma_db"
    collection_name: str = "meddra_pts"
    batch_size: int = 100


class DuckDBSettings(BaseSettings):
    """DuckDB SQL engine configuration."""

    model_config = SettingsConfigDict(env_prefix="DUCKDB_")

    path: Path = PROJECT_ROOT / "data" / "duckdb" / "faers.duckdb"
    memory_limit: str = "2GB"
    threads: int = 4


class DataSettings(BaseSettings):
    """Data directory paths."""

    model_config = SettingsConfigDict(env_prefix="")

    faers_raw_dir: Path = PROJECT_ROOT / "data" / "raw" / "faers"
    faers_quarters: list[str] = ["2024Q4", "2025Q1"]
    reviews_raw_dir: Path = PROJECT_ROOT / "data" / "raw" / "reviews"
    rxnorm_data_dir: Path = PROJECT_ROOT / "data" / "raw" / "rxnorm"
    ground_truth_dir: Path = PROJECT_ROOT / "data" / "ground_truth"
    eval_results_dir: Path = PROJECT_ROOT / "data" / "eval_results"

    @field_validator("faers_quarters", mode="before")
    @classmethod
    def parse_quarters(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated quarters from env var."""
        if isinstance(v, str):
            return [q.strip() for q in v.split(",")]
        return v


class CacheSettings(BaseSettings):
    """Caching configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    llm_cache_dir: Path = PROJECT_ROOT / "data" / "cache" / "llm_extractions"
    embedding_cache_dir: Path = PROJECT_ROOT / "data" / "cache" / "embeddings"


class ProcessingSettings(BaseSettings):
    """Pipeline processing configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    max_reviews_to_process: int = 30_000
    batch_size: int = 50
    confidence_threshold: float = 0.75


class StreamlitSettings(BaseSettings):
    """Streamlit dashboard configuration."""

    model_config = SettingsConfigDict(env_prefix="STREAMLIT_")

    port: int = 8501
    theme: Literal["dark", "light"] = "dark"


class LogSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = "INFO"
    dir: Path = PROJECT_ROOT / "logs"


class Settings(BaseSettings):
    """Root settings object aggregating all subsystem configs."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configs
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    duckdb: DuckDBSettings = Field(default_factory=DuckDBSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    streamlit: StreamlitSettings = Field(default_factory=StreamlitSettings)
    log: LogSettings = Field(default_factory=LogSettings)

    # Derived
    project_root: Path = PROJECT_ROOT

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        dirs = [
            self.data.faers_raw_dir,
            self.data.reviews_raw_dir,
            self.data.rxnorm_data_dir,
            self.data.ground_truth_dir,
            self.data.eval_results_dir,
            self.cache.llm_cache_dir,
            self.cache.embedding_cache_dir,
            self.chroma.persist_dir,
            self.duckdb.path.parent,
            self.log.dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Singleton instance — import this everywhere
settings = Settings()
