"""Configuration module for the WhisperX FastAPI application."""

import os
import sys
from functools import lru_cache
from typing import Optional

# CRITICAL: Try to set LD_PRELOAD before importing torch
# This is a fallback in case the startup script didn't execute
# Note: This usually won't work because LD_PRELOAD must be set before process starts,
# but we try anyway as a last resort
if "LD_PRELOAD" not in os.environ or not os.environ.get("LD_PRELOAD"):
    # Try to find NCCL library at runtime
    import glob
    
    # Check common NCCL locations
    torch_lib = "/usr/local/lib/python3.11/dist-packages/torch/lib"
    system_nccl = f"/usr/lib/{os.uname().machine}-linux-gnu/libnccl.so.2"
    
    nccl_lib = None
    # Try system NCCL first
    if os.path.exists(system_nccl):
        nccl_lib = system_nccl
    # Try PyTorch bundled NCCL
    elif os.path.exists(torch_lib):
        nccl_files = glob.glob(os.path.join(torch_lib, "libnccl*.so*"))
        if nccl_files:
            nccl_lib = nccl_files[0]
    
    if nccl_lib and os.path.exists(nccl_lib):
        # Try to set LD_PRELOAD (may not work, but worth trying)
        os.environ["LD_PRELOAD"] = nccl_lib
        nccl_dir = os.path.dirname(nccl_lib)
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{nccl_dir}:{torch_lib}:/usr/local/cuda/lib64:{current_ld_path}"

try:
    import torch
except (ImportError, OSError, RuntimeError) as e:
    # Provide helpful error message for NCCL-related import errors
    error_msg = str(e)
    if "ncclGroupSimulateEnd" in error_msg or "undefined symbol" in error_msg:
        ld_preload = os.environ.get("LD_PRELOAD", "not set")
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "not set")
        raise ImportError(
            f"PyTorch import failed due to NCCL compatibility issue: {error_msg}\n"
            f"This usually indicates that PyTorch 2.8+ requires NCCL 2.18+ with the "
            f"'ncclGroupSimulateEnd' symbol.\n"
            f"Diagnostic information:\n"
            f"  LD_PRELOAD: {ld_preload}\n"
            f"  LD_LIBRARY_PATH: {ld_library_path}\n"
            f"  Python attempted to set LD_PRELOAD but it may be too late.\n"
            f"Please ensure the startup script correctly sets LD_PRELOAD to a compatible NCCL library."
        ) from e
    raise

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas import ComputeType, Device, WhisperModel


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    DB_URL: str = Field(
        default="sqlite:///records.db",
        description="Database connection URL",
    )
    DB_ECHO: bool = Field(
        default=False,
        description="Echo SQL queries for debugging",
    )


class WhisperSettings(BaseSettings):
    """WhisperX ML model configuration settings."""

    HF_TOKEN: Optional[str] = Field(
        default=None,
        description="HuggingFace API token for model downloads",
    )
    WHISPER_MODEL: WhisperModel = Field(
        default=WhisperModel.tiny,
        description="Whisper model size to use",
    )
    DEFAULT_LANG: str = Field(
        default="en",
        description="Default language for transcription",
    )
    DEVICE: Device = Field(
        default_factory=lambda: Device.cuda
        if torch.cuda.is_available()
        else Device.cpu,
        description="Device to use for computation (cuda or cpu)",
    )
    COMPUTE_TYPE: ComputeType = Field(
        default_factory=lambda: (
            ComputeType.float16 if torch.cuda.is_available() else ComputeType.int8
        ),
        description="Compute type for model inference",
    )

    AUDIO_EXTENSIONS: set[str] = {
        ".mp3",
        ".wav",
        ".awb",
        ".aac",
        ".ogg",
        ".oga",
        ".m4a",
        ".wma",
        ".amr",
    }
    VIDEO_EXTENSIONS: set[str] = {".mp4", ".mov", ".avi", ".wmv", ".mkv"}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ALLOWED_EXTENSIONS(self) -> set[str]:
        """Compute allowed extensions by combining audio and video."""
        return self.AUDIO_EXTENSIONS | self.VIDEO_EXTENSIONS

    @model_validator(mode="after")
    def validate_compute_type_for_cpu(self) -> "WhisperSettings":
        """Validate that CPU device uses int8 compute type."""
        if self.DEVICE == Device.cpu and self.COMPUTE_TYPE != ComputeType.int8:
            # Auto-correct instead of raising error
            self.COMPUTE_TYPE = ComputeType.int8
        return self


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    LOG_FORMAT: str = Field(
        default="text",
        description="Log format: text or json",
    )
    FILTER_WARNING: bool = Field(
        default=True,
        description="Filter specific warnings",
    )


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_nested_delimiter="__",
    )

    ENVIRONMENT: str = Field(
        default="production",
        description="Environment: development, testing, production",
    )
    DEV: bool = Field(
        default=False,
        description="Development mode flag",
    )

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def normalize_environment(cls, v: str) -> str:
        """Normalize environment to lowercase."""
        return str(v).lower() if v else "production"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance (singleton pattern).

    Returns:
        Settings: The application settings instance.
    """
    return Settings()


# Legacy Config class for backward compatibility during migration
# This will be removed once all references are updated
class Config:
    """DEPRECATED: Legacy configuration class. Use get_settings() instead."""

    _settings = get_settings()

    # Delegate to new settings
    LANG = _settings.whisper.DEFAULT_LANG
    HF_TOKEN = _settings.whisper.HF_TOKEN
    WHISPER_MODEL = _settings.whisper.WHISPER_MODEL
    DEVICE = _settings.whisper.DEVICE
    COMPUTE_TYPE = _settings.whisper.COMPUTE_TYPE
    ENVIRONMENT = _settings.ENVIRONMENT
    LOG_LEVEL = _settings.logging.LOG_LEVEL
    AUDIO_EXTENSIONS = _settings.whisper.AUDIO_EXTENSIONS
    VIDEO_EXTENSIONS = _settings.whisper.VIDEO_EXTENSIONS
    ALLOWED_EXTENSIONS = _settings.whisper.ALLOWED_EXTENSIONS
    DB_URL = _settings.database.DB_URL
