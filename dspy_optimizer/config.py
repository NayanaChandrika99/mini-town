from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application configuration sourced from environment variables.

    Environment variables are prefixed with ``DSPY_OPT_`` by default.
    Example: ``DSPY_OPT_AUTH_TOKEN`` or ``DSPY_OPT_COMPILED_DIR``.
    """

    # Base directories
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    compiled_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "compiled")

    # LLM configuration (optional for read-only mode)
    provider: str = Field(default="together", description="Default LLM provider name.")
    model: str = Field(default="meta-llama/Llama-3.2-3B-Instruct-Turbo", description="Default LLM model identifier.")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    together_api_key: Optional[str] = Field(default=None, alias="TOGETHER_API_KEY")

    # Service toggles
    allow_compile: bool = Field(
        default=False,
        description="Set to true to enable runtime compilation endpoints."
    )
    auth_token: Optional[str] = Field(
        default=None,
        description="Shared secret token required on incoming requests."
    )

    model_config = SettingsConfigDict(env_prefix="DSPY_OPT_", env_file=".env", extra="ignore")


class AppState(BaseModel):
    """Mutable state cached on the FastAPI app instance."""

    settings: Settings
    dspy_configured: bool = False


def load_settings() -> Settings:
    """Helper to instantiate settings and emit a short diagnostic log."""
    settings = Settings()
    logger.info(
        "Optimizer settings loaded (compiled_dir=%s, allow_compile=%s)",
        settings.compiled_dir,
        settings.allow_compile,
    )
    return settings
