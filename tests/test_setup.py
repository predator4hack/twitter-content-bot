"""
Test suite for Task 1.1: Project Setup & Dependencies

This module tests the basic project setup, configuration loading,
and dependency installation validation.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import Config, config
from src.core.logger import setup_logger, get_logger


class TestProjectSetup:
    """Test basic project setup and structure."""
    
    def test_project_structure_exists(self):
        """Test that all required directories exist."""
        project_root = Path(__file__).parent.parent
        
        required_dirs = [
            "src",
            "src/core",
            "src/downloader", 
            "src/transcription",
            "src/analyzer",
            "src/clipper",
            "src/ui",
            "tests"
        ]
        
        for dir_path in required_dirs:
            assert (project_root / dir_path).exists(), f"Directory {dir_path} does not exist"
            assert (project_root / dir_path).is_dir(), f"{dir_path} is not a directory"
    
    def test_init_files_exist(self):
        """Test that all __init__.py files exist."""
        project_root = Path(__file__).parent.parent
        
        required_init_files = [
            "src/__init__.py",
            "src/core/__init__.py",
            "src/downloader/__init__.py",
            "src/transcription/__init__.py", 
            "src/analyzer/__init__.py",
            "src/clipper/__init__.py",
            "src/ui/__init__.py"
        ]
        
        for init_file in required_init_files:
            init_path = project_root / init_file
            assert init_path.exists(), f"Init file {init_file} does not exist"
            assert init_path.is_file(), f"{init_file} is not a file"
    
    def test_config_files_exist(self):
        """Test that configuration files exist."""
        project_root = Path(__file__).parent.parent
        
        config_files = [
            "pyproject.toml",
            ".env.template",
            "requirements.txt"
        ]
        
        for config_file in config_files:
            assert (project_root / config_file).exists(), f"Config file {config_file} does not exist"


class TestConfiguration:
    """Test configuration module functionality."""
    
    def test_config_import(self):
        """Test that config module can be imported."""
        from src.core.config import Config, config
        assert Config is not None
        assert config is not None
    
    def test_config_attributes(self):
        """Test that config has all required attributes."""
        required_attrs = [
            "PROJECT_ROOT",
            "TEMP_DIR", 
            "OUTPUT_DIR",
            "CACHE_DIR",
            "DEFAULT_LLM_PROVIDER",
            "WHISPER_MODEL",
            "LOG_LEVEL"
        ]
        
        for attr in required_attrs:
            assert hasattr(Config, attr), f"Config missing attribute: {attr}"
    
    def test_config_validation(self):
        """Test config validation functionality."""
        validation_results = Config.validate_config()
        
        assert isinstance(validation_results, dict)
        required_keys = ["directories", "api_keys", "whisper_model", "youtube_settings"]
        
        for key in required_keys:
            assert key in validation_results, f"Validation missing key: {key}"
    
    def test_directory_setup(self):
        """Test that directories can be created."""
        result = Config.setup_directories()
        assert result is True, "Directory setup failed"
        
        # Check that directories actually exist
        assert Config.TEMP_DIR.exists(), "TEMP_DIR was not created"
        assert Config.OUTPUT_DIR.exists(), "OUTPUT_DIR was not created" 
        assert Config.CACHE_DIR.exists(), "CACHE_DIR was not created"
    
    def test_config_summary(self):
        """Test config summary generation."""
        summary = Config.get_summary()
        
        assert isinstance(summary, dict)
        required_keys = [
            "llm_provider",
            "whisper_model", 
            "max_video_duration",
            "max_clip_duration",
            "ytdl_quality",
            "parallel_processing",
            "api_keys_configured"
        ]
        
        for key in required_keys:
            assert key in summary, f"Summary missing key: {key}"


class TestLogging:
    """Test logging configuration and functionality."""
    
    def test_logger_import(self):
        """Test that logger module can be imported."""
        from src.core.logger import setup_logger, get_logger
        assert setup_logger is not None
        assert get_logger is not None
    
    def test_logger_creation(self):
        """Test logger creation with default settings."""
        logger = setup_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_logger_levels(self):
        """Test that logger handles different log levels."""
        logger = setup_logger("test_levels", level="DEBUG")
        
        # Test that these don't raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
    
    def test_get_logger(self):
        """Test the get_logger convenience function."""
        logger = get_logger("test_get_logger")
        assert logger is not None
        assert logger.name == "test_get_logger"


class TestDependencyImports:
    """Test that all required dependencies can be imported."""
    
    def test_core_dependencies(self):
        """Test importing core dependencies."""
        try:
            import streamlit
            import yt_dlp
            import faster_whisper
            import google.generativeai
            import groq
            import ffmpeg_python
            import dotenv
            import requests
            import PIL
            import numpy
            import pandas
        except ImportError as e:
            pytest.fail(f"Failed to import core dependency: {e}")
    
    def test_optional_dependencies(self):
        """Test importing optional development dependencies."""
        optional_deps = ["pytest", "black", "isort", "flake8"]
        
        for dep in optional_deps:
            try:
                __import__(dep)
            except ImportError:
                # Optional dependencies are allowed to fail
                pass


def test_environment_template():
    """Test that .env.template has required variables."""
    project_root = Path(__file__).parent.parent
    env_template = project_root / ".env.template"
    
    assert env_template.exists(), ".env.template file not found"
    
    content = env_template.read_text()
    required_vars = [
        "GOOGLE_API_KEY",
        "GROQ_API_KEY", 
        "DEFAULT_LLM_PROVIDER",
        "WHISPER_MODEL",
        "LOG_LEVEL"
    ]
    
    for var in required_vars:
        assert var in content, f"Required environment variable {var} not in template"


def test_main_module_import():
    """Test that the main src module can be imported."""
    try:
        import src
        assert hasattr(src, "__version__")
    except ImportError as e:
        pytest.fail(f"Failed to import main src module: {e}")


if __name__ == "__main__":
    # Run tests manually for immediate validation
    print("🧪 Running Task 1.1 Setup Tests...")
    
    # Test configuration
    print("\n📋 Testing Configuration...")
    try:
        from src.core.config import Config, config
        validation = Config.validate_config()
        print(f"✅ Config loaded successfully")
        print(f"📊 Validation results: {validation}")
        
        summary = Config.get_summary() 
        print(f"📋 Config summary: {summary}")
    except Exception as e:
        print(f"❌ Config test failed: {e}")
    
    # Test logging
    print("\n📝 Testing Logging...")
    try:
        from src.core.logger import setup_logger
        logger = setup_logger("test")
        logger.info("✅ Logger test successful")
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
    
    # Test imports
    print("\n📦 Testing Dependencies...")
    dependencies = [
        "streamlit", "yt_dlp", "faster_whisper", "google.generativeai", 
        "groq", "ffmpeg_python", "dotenv", "requests", "PIL", "numpy", "pandas"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - Not installed")
    
    print("\n🎉 Task 1.1 setup testing complete!")
