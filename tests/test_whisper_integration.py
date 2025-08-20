"""
Simple integration test for Whisper transcriber.
"""

import pytest
from src.transcription import WhisperTranscriber


def test_whisper_transcriber_creation():
    """Test that we can create a WhisperTranscriber instance."""
    transcriber = WhisperTranscriber()
    assert transcriber is not None
    assert transcriber.model_size == "base"
    assert transcriber.device in ["cpu", "cuda"]
    

def test_whisper_transcriber_languages():
    """Test that we can get supported languages."""
    transcriber = WhisperTranscriber()
    languages = transcriber.get_supported_languages()
    assert isinstance(languages, list)
    assert len(languages) > 0
    assert 'en' in languages


def test_whisper_transcriber_info():
    """Test getting transcriber info."""
    transcriber = WhisperTranscriber()
    info = transcriber.get_info()
    assert isinstance(info, dict)
    assert 'class' in info
    assert 'supported_languages' in info
