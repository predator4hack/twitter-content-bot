"""
Transcription module for YouTube to Twitter Clipper.

This module provides speech-to-text functionality using OpenAI's Whisper model,
with support for multiple languages, speaker diarization, and timestamp accuracy.
"""

from .base import TranscriptionResult, TranscriptionSegment, BaseTranscriber
from .whisper_transcriber import WhisperTranscriber

__all__ = [
    'BaseTranscriber',
    'WhisperTranscriber',
    'TranscriptionResult', 
    'TranscriptionSegment'
]
