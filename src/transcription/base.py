"""
Base classes for transcription functionality.

This module defines the abstract base classes and data structures used
throughout the transcription system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union


@dataclass
class TranscriptionSegment:
    """
    Represents a single segment of transcribed text with timing information.
    
    Attributes:
        text: The transcribed text for this segment
        start_time: Start time in seconds
        end_time: End time in seconds
        confidence: Confidence score (0.0 to 1.0)
        words: Optional word-level timing information
    """
    text: str
    start_time: float
    end_time: float
    confidence: float = 0.0
    words: Optional[List[Dict[str, Any]]] = None
    
    @property
    def duration(self) -> float:
        """Get the duration of this segment in seconds."""
        return self.end_time - self.start_time
    
    def __str__(self) -> str:
        """String representation of the segment."""
        return f"[{self.start_time:.2f}-{self.end_time:.2f}]: {self.text}"


@dataclass
class TranscriptionResult:
    """
    Complete transcription result containing all segments and metadata.
    
    Attributes:
        text: Full transcribed text
        segments: List of individual segments
        language: Detected or specified language
        confidence: Overall confidence score
        metadata: Additional information about the transcription
    """
    text: str
    segments: List[TranscriptionSegment]
    language: str
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization to set defaults."""
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def duration(self) -> float:
        """Get the total duration of the transcription."""
        if not self.segments:
            return 0.0
        return max(segment.end_time for segment in self.segments)
    
    @property
    def word_count(self) -> int:
        """Get the total word count."""
        return len(self.text.split())
    
    def get_text_at_time(self, timestamp: float) -> Optional[str]:
        """
        Get the text that is being spoken at a specific timestamp.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Text at the given timestamp, or None if no speech at that time
        """
        for segment in self.segments:
            if segment.start_time <= timestamp <= segment.end_time:
                return segment.text
        return None
    
    def get_segments_in_range(self, start_time: float, end_time: float) -> List[TranscriptionSegment]:
        """
        Get all segments that overlap with the given time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of segments that overlap with the range
        """
        result = []
        for segment in self.segments:
            # Check if segment overlaps with the range
            if (segment.start_time <= end_time and segment.end_time >= start_time):
                result.append(segment)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'language': self.language,
            'confidence': self.confidence,
            'duration': self.duration,
            'word_count': self.word_count,
            'segments': [
                {
                    'text': seg.text,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'confidence': seg.confidence,
                    'words': seg.words
                }
                for seg in self.segments
            ],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionResult':
        """Create from dictionary representation."""
        segments = [
            TranscriptionSegment(
                text=seg['text'],
                start_time=seg['start_time'],
                end_time=seg['end_time'],
                confidence=seg.get('confidence', 0.0),
                words=seg.get('words')
            )
            for seg in data.get('segments', [])
        ]
        
        return cls(
            text=data['text'],
            segments=segments,
            language=data['language'],
            confidence=data.get('confidence', 0.0),
            metadata=data.get('metadata', {})
        )


class BaseTranscriber(ABC):
    """
    Abstract base class for all transcription implementations.
    
    This class defines the interface that all transcribers must implement,
    ensuring consistency across different transcription backends.
    """
    
    def __init__(self):
        """Initialize the transcriber."""
        pass
    
    @abstractmethod
    def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio or video file.
        
        Args:
            file_path: Path to the file to transcribe
            language: Language code (e.g., 'en', 'es'). Auto-detect if None
            **kwargs: Additional transcriber-specific options
            
        Returns:
            TranscriptionResult containing the transcribed text and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If transcription fails
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get a list of supported language codes.
        
        Returns:
            List of language codes (e.g., ['en', 'es', 'fr'])
        """
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Check if the transcriber is healthy and ready to use.
        
        Returns:
            True if the transcriber is working properly, False otherwise
        """
        pass
    
    def transcribe_audio_data(
        self,
        audio_data: bytes,
        format: str = "wav",
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio data directly (optional implementation).
        
        This is an optional method that subclasses can override to support
        transcribing audio data without writing to a temporary file.
        
        Args:
            audio_data: Raw audio data
            format: Audio format (wav, mp3, etc.)
            language: Language code
            **kwargs: Additional options
            
        Returns:
            TranscriptionResult
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("This transcriber doesn't support audio data transcription")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this transcriber.
        
        Returns:
            Dictionary with transcriber information
        """
        return {
            'class': self.__class__.__name__,
            'module': self.__class__.__module__,
            'supported_languages': self.get_supported_languages(),
            'healthy': self.is_healthy()
        }
