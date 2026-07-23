"""
Centralized configuration for the Personalized Prompt Refiner.

All tunables — LLM params, HuggingFace model names, retry/timeout
settings, paths — live here. Change a value once; every module picks it up.

Usage:
    from app.config import settings
    client.chat(..., temperature=settings.llm_temperature)

Environment variable precedence (highest → lowest):
    1. Shell env vars  (export LLM_TEMPERATURE=0.3)
    2. .env file       (LLM_TEMPERATURE=0.3)
    3. Field defaults  (defined below)
"""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings resolved from environment variables / .env.

    All fields map directly to an uppercase env var of the same name.
    Example: ``llm_temperature`` ↔ ``LLM_TEMPERATURE=0.5`` in .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ── LLM Provider ────────────────────────────────────────────────────────
    llm_provider: str = Field(
        default="groq",
        description="Active LLM backend. Options: 'groq', 'ollama'.",
    )

    # ── Groq Configuration ──────────────────────────────────────────────────
    groq_api_key: str = Field(
        default="",
        description="Groq API key. Get a free key at https://console.groq.com",
    )
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model identifier.",
    )

    # ── Ollama Configuration ────────────────────────────────────────────────
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )
    ollama_model: str = Field(
        default="llama3.1",
        description="Ollama model name (must be pulled locally).",
    )

    # ── Shared LLM Generation Parameters ───────────────────────────────────
    # Change these once here — all clients pick them up automatically.
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower = more deterministic (0.0–2.0).",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=64,
        le=8192,
        description="Max tokens in the LLM response.",
    )
    llm_timeout_seconds: int = Field(
        default=60,
        ge=5,
        description="Per-request timeout in seconds for LLM API calls.",
    )

    # ── Retry / Back-off (tenacity) ─────────────────────────────────────────
    llm_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retry attempts for transient LLM errors.",
    )
    llm_retry_min_wait: float = Field(
        default=1.0,
        ge=0.1,
        description="Minimum seconds to wait between retries (exponential back-off base).",
    )
    llm_retry_max_wait: float = Field(
        default=30.0,
        ge=1.0,
        description="Maximum seconds to wait between retries.",
    )

    # ── ChromaDB ────────────────────────────────────────────────────────────
    chroma_db_path: str = Field(
        default="./chroma_db",
        description="Path for persistent ChromaDB vector storage.",
    )

    # ── HuggingFace / Specialist Pipelines ──────────────────────────────────
    # Set HF_USE_PIPELINE=false to disable all HF pipelines (low-RAM machines).
    hf_use_pipeline: bool = Field(
        default=True,
        description=(
            "Master switch for HuggingFace specialist pipelines. "
            "Set to false on low-RAM machines to fall back to LLM for all tasks."
        ),
    )
    hf_translation_model: str = Field(
        default="Helsinki-NLP/opus-mt-ar-en",
        description="HuggingFace model for Arabic → English translation (primary).",
    )
    hf_finbert_model: str = Field(
        default="ProsusAI/finbert",
        description="FinBERT model for marketing/business domain signal extraction.",
    )
    hf_summarization_model: str = Field(
        default="facebook/bart-large-cnn",
        description="Summarization model for research domain intent condensation.",
    )
    hf_code_model: str = Field(
        default="microsoft/codebert-base",
        description="CodeBERT model for dev domain context anchoring.",
    )
    hf_zeroshot_model: str = Field(
        default="facebook/bart-large-mnli",
        description="Zero-shot classification model for education domain.",
    )
    hf_sentiment_model: str = Field(
        default="distilbert-base-uncased-finetuned-sst-2-english",
        description="Sentiment model for data_analysis domain signal.",
    )
    hf_creative_model: str = Field(
        default="gpt2",
        description="Lightweight text-generation model for creative_writing enrichment.",
    )

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    log_dir: str = Field(
        default="logs",
        description="Directory for rotating log files.",
    )
    log_max_bytes: int = Field(
        default=5 * 1024 * 1024,   # 5 MB
        description="Max size in bytes per log file before rotation.",
    )
    log_backup_count: int = Field(
        default=3,
        description="Number of rotated log file backups to keep.",
    )

    # ── Validators ──────────────────────────────────────────────────────────

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"groq", "ollama"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"LLM_PROVIDER must be one of {allowed}, got '{v}'.")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper().strip()
        if v not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{v}'.")
        return v

    # ── Derived / Convenience Properties ───────────────────────────────────

    @property
    def chroma_db_resolved(self) -> Path:
        """Return ChromaDB path resolved relative to CWD."""
        return Path(self.chroma_db_path).resolve()

    @property
    def log_dir_resolved(self) -> Path:
        """Return log directory path resolved relative to CWD."""
        return Path(self.log_dir).resolve()

    @property
    def groq_api_key_is_set(self) -> bool:
        """True if a real Groq API key is configured."""
        return bool(self.groq_api_key) and self.groq_api_key != "your_groq_api_key_here"


# ── Singleton ──────────────────────────────────────────────────────────────
# Import `settings` everywhere — do NOT instantiate Settings() again.
settings = Settings()
