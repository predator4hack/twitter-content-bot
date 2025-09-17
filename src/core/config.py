"""
Configuration management for the YouTube to Twitter clipper.

This module handles environment variables, API keys, and application settings
with proper validation and defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Clear proxy environment variables if BYPASS_PROXY is enabled
if os.getenv("BYPASS_PROXY", "true").lower() == "true":
    # Clear all possible proxy environment variables
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]
    for var in proxy_vars:
        if var in os.environ:
            del os.environ[var]


class Config:
    """Application configuration with environment variable support."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    SRC_DIR = PROJECT_ROOT / "src"
    TEMP_DIR = Path(os.getenv("TEMP_DIR", "./temp"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
    CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
    
    # API Configuration
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    
    # Application Settings
    MAX_VIDEO_DURATION: int = int(os.getenv("MAX_VIDEO_DURATION", "600"))
    MAX_CLIP_DURATION: int = int(os.getenv("MAX_CLIP_DURATION", "140"))
    DEFAULT_CLIP_COUNT: int = int(os.getenv("DEFAULT_CLIP_COUNT", "3"))
    
    # Processing Settings
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")  # Force CPU to avoid CUDA issues
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # CPU-optimized
    PARALLEL_PROCESSING: bool = os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
    MAX_CONCURRENT_CLIPS: int = int(os.getenv("MAX_CONCURRENT_CLIPS", "3"))
    
    # Storage Settings
    CLEANUP_TEMP_FILES: bool = os.getenv("CLEANUP_TEMP_FILES", "true").lower() == "true"
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")
    
    # YouTube Download Settings
    YTDL_QUALITY: str = os.getenv("YTDL_QUALITY", "720p")
    YTDL_FORMAT: str = os.getenv("YTDL_FORMAT", "mp4")
    EXTRACT_THUMBNAILS: bool = os.getenv("EXTRACT_THUMBNAILS", "true").lower() == "true"
    
    # Twitter Optimization Settings
    TARGET_FILESIZE_MB: int = int(os.getenv("TARGET_FILESIZE_MB", "50"))
    VIDEO_BITRATE: str = os.getenv("VIDEO_BITRATE", "1000k")
    AUDIO_BITRATE: str = os.getenv("AUDIO_BITRATE", "128k")
    
    # Network Settings
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "5"))
    BYPASS_PROXY: bool = os.getenv("BYPASS_PROXY", "true").lower() == "true"
    
    # Bot Detection Avoidance
    RANDOMIZE_USER_AGENTS: bool = os.getenv("RANDOMIZE_USER_AGENTS", "true").lower() == "true"
    USE_MOBILE_CLIENTS: bool = os.getenv("USE_MOBILE_CLIENTS", "true").lower() == "true"
    RATE_LIMIT_DELAY: float = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))
    
    @classmethod
    def validate_config(cls) -> dict[str, bool]:
        """
        Validate configuration and return status of required settings.
        
        Returns:
            Dictionary with validation results for each component.
        """
        validation_results = {
            "directories": True,
            "api_keys": False,
            "whisper_model": True,
            "youtube_settings": True,
        }
        
        # Check if directories can be created
        try:
            cls.TEMP_DIR.mkdir(exist_ok=True)
            cls.OUTPUT_DIR.mkdir(exist_ok=True)
            cls.CACHE_DIR.mkdir(exist_ok=True)
        except Exception:
            validation_results["directories"] = False
        
        # Check API keys
        if cls.DEFAULT_LLM_PROVIDER == "gemini" and cls.GOOGLE_API_KEY:
            validation_results["api_keys"] = True
        elif cls.DEFAULT_LLM_PROVIDER == "groq" and cls.GROQ_API_KEY:
            validation_results["api_keys"] = True
        
        # Validate Whisper model
        valid_whisper_models = ["tiny", "base", "small", "medium", "large"]
        if cls.WHISPER_MODEL not in valid_whisper_models:
            validation_results["whisper_model"] = False
        
        return validation_results
    
    @classmethod
    def setup_directories(cls) -> bool:
        """
        Create necessary directories for the application.
        
        Returns:
            True if all directories were created successfully.
        """
        try:
            cls.TEMP_DIR.mkdir(exist_ok=True)
            cls.OUTPUT_DIR.mkdir(exist_ok=True)
            cls.CACHE_DIR.mkdir(exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directories: {e}")
            return False
    
    @classmethod
    def get_summary(cls) -> dict:
        """Get a summary of current configuration."""
        return {
            "llm_provider": cls.DEFAULT_LLM_PROVIDER,
            "whisper_model": cls.WHISPER_MODEL,
            "max_video_duration": f"{cls.MAX_VIDEO_DURATION}s",
            "max_clip_duration": f"{cls.MAX_CLIP_DURATION}s",
            "ytdl_quality": cls.YTDL_QUALITY,
            "parallel_processing": cls.PARALLEL_PROCESSING,
            "api_keys_configured": bool(cls.GOOGLE_API_KEY or cls.GROQ_API_KEY),
        }


# Global config instance
config = Config()

# Ensure directories exist
config.setup_directories()
