"""
OpenAI Whisper speech-to-text implementation using faster-whisper.

This module provides transcription functionality using faster-whisper,
which is more efficient than the original OpenAI Whisper implementation.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import av
import ffmpeg
from faster_whisper import WhisperModel, download_model
from faster_whisper.transcribe import Segment

from .base import TranscriptionResult, TranscriptionSegment, BaseTranscriber

logger = logging.getLogger(__name__)


class WhisperTranscriber(BaseTranscriber):
    """
    Whisper-based speech-to-text transcriber using faster-whisper.
    
    This implementation provides high-quality transcription with timestamp accuracy
    and supports multiple languages and model sizes.
    """
    
    SUPPORTED_MODELS = [
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large"
    ]
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        download_root: Optional[str] = None,
        local_files_only: bool = False
    ):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_size: Size of the Whisper model to use
            device: Device to run inference on ("cpu", "cuda", "auto")
            compute_type: Computation type ("int8", "int16", "float16", "float32", "auto")
            download_root: Directory to save downloaded models
            local_files_only: Only use local model files, don't download
        """
        super().__init__()
        
        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model size: {model_size}. "
                           f"Supported models: {', '.join(self.SUPPORTED_MODELS)}")
        
        self.model_size = model_size
        self.device = device if device != "auto" else self._detect_device()
        self.compute_type = compute_type if compute_type != "auto" else self._detect_compute_type()
        # Use project-local cache directory for better control
        if download_root:
            self.download_root = download_root
        else:
            # Store models in project directory for persistence
            project_root = Path(__file__).parent.parent.parent
            self.download_root = str(project_root / "models" / "whisper")
            Path(self.download_root).mkdir(parents=True, exist_ok=True)
        
        self.local_files_only = local_files_only
        
        self.model: Optional[WhisperModel] = None
        
        logger.info(f"Initialized Whisper transcriber with model={model_size}, "
                   f"device={self.device}, compute_type={self.compute_type}")
    
    def _detect_device(self) -> str:
        """Detect the best available device for inference."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    def _detect_compute_type(self) -> str:
        """Detect the best compute type based on device."""
        if self.device == "cuda":
            return "float16"
        return "int8"
    
    def _load_model(self) -> None:
        """Load the Whisper model if not already loaded."""
        if self.model is not None:
            return
        
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Check if model already exists locally
            model_path = Path(self.download_root) / self.model_size
            model_exists = model_path.exists() and any(model_path.iterdir())
            
            if model_exists:
                logger.info(f"Found existing model at: {model_path}")
                # Try to use local model first
                try:
                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root=self.download_root,
                        local_files_only=True  # Use local only first
                    )
                    logger.info(f"Successfully loaded cached model: {self.model_size}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cached model: {e}. Will try downloading.")
            
            # Download model if needed
            if not self.local_files_only:
                logger.info(f"Downloading Whisper model: {self.model_size}")
                try:
                    download_model(self.model_size, cache_dir=self.download_root)
                    logger.info(f"Successfully downloaded model to: {self.download_root}")
                except Exception as e:
                    logger.warning(f"Failed to download model: {e}. Trying to use any local files.")
            
            # Load the model
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
                local_files_only=self.local_files_only
            )
            
            logger.info(f"Successfully loaded Whisper model: {self.model_size} from {self.download_root}")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Could not load Whisper model '{self.model_size}': {e}")
    
    def _extract_audio_from_video(self, video_path: str, temp_dir: str) -> str:
        """
        Extract audio from video file using ffmpeg.
        
        Args:
            video_path: Path to the video file
            temp_dir: Temporary directory for audio file
            
        Returns:
            Path to the extracted audio file
        """
        audio_path = os.path.join(temp_dir, "audio.wav")
        
        try:
            # Use ffmpeg to extract audio
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(
                stream,
                audio_path,
                acodec='pcm_s16le',  # 16-bit PCM
                ac=1,               # Mono
                ar='16000'          # 16kHz sample rate (Whisper's preferred)
            )
            ffmpeg.run(stream, quiet=True, overwrite_output=True)
            
            logger.debug(f"Extracted audio to: {audio_path}")
            return audio_path
            
        except Exception as e:
            # Handle both ffmpeg.Error and general exceptions
            logger.error(f"FFmpeg error extracting audio: {e}")
            raise RuntimeError(f"Failed to extract audio from video: {e}")
    
    def _validate_audio_file(self, audio_path: str) -> None:
        """
        Validate that the audio file exists and is readable.
        
        Args:
            audio_path: Path to the audio file
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If audio file is invalid
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if os.path.getsize(audio_path) == 0:
            raise RuntimeError(f"Audio file is empty: {audio_path}")
        
        # Try to read the audio file with av to validate format
        try:
            with av.open(audio_path) as container:
                if not container.streams.audio:
                    raise RuntimeError(f"No audio streams found in file: {audio_path}")
        except Exception as e:
            raise RuntimeError(f"Invalid audio file format: {e}")
    
    def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio or video file.
        
        Args:
            file_path: Path to audio or video file
            language: Language code (e.g., 'en', 'es'). Auto-detect if None
            task: Task type ('transcribe' or 'translate')
            **kwargs: Additional arguments passed to the model
            
        Returns:
            TranscriptionResult with segments and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load model if needed
        self._load_model()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Check if it's a video file that needs audio extraction
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']:
                logger.info(f"Extracting audio from video file: {file_path}")
                audio_path = self._extract_audio_from_video(file_path, temp_dir)
            else:
                audio_path = file_path
            
            # Validate audio file
            self._validate_audio_file(audio_path)
            
            return self._transcribe_audio(audio_path, language, task, **kwargs)
    
    def _transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
        **kwargs
    ) -> TranscriptionResult:
        """
        Internal method to transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            task: Task type
            **kwargs: Additional arguments
            
        Returns:
            TranscriptionResult
        """
        logger.info(f"Starting transcription: {audio_path}")
        
        try:
            # Set default parameters
            transcribe_kwargs = {
                'language': language,
                'task': task,
                'beam_size': kwargs.get('beam_size', 5),
                'best_of': kwargs.get('best_of', 5),
                'temperature': kwargs.get('temperature', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                'compression_ratio_threshold': kwargs.get('compression_ratio_threshold', 2.4),
                'log_prob_threshold': kwargs.get('log_prob_threshold', -1.0),
                'no_speech_threshold': kwargs.get('no_speech_threshold', 0.6),
                'condition_on_previous_text': kwargs.get('condition_on_previous_text', True),
                'word_timestamps': kwargs.get('word_timestamps', True),
                'prepend_punctuations': kwargs.get('prepend_punctuations', "\"'¿([{-"),
                'append_punctuations': kwargs.get('append_punctuations', "\"'.。,，!！?？:：\")]}、"),
            }
            
            # Remove None values
            transcribe_kwargs = {k: v for k, v in transcribe_kwargs.items() if v is not None}
            
            # Transcribe
            segments_iter, info = self.model.transcribe(audio_path, **transcribe_kwargs)
            
            # Convert segments to our format
            segments = []
            full_text_parts = []
            
            for segment in segments_iter:
                # Create word-level timestamps if available
                words = []
                if hasattr(segment, 'words') and segment.words:
                    words = [
                        {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        }
                        for word in segment.words
                    ]
                
                transcription_segment = TranscriptionSegment(
                    text=segment.text.strip(),
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=getattr(segment, 'avg_logprob', 0.0),
                    words=words
                )
                
                segments.append(transcription_segment)
                full_text_parts.append(segment.text.strip())
            
            # Combine all segments into full text
            full_text = ' '.join(full_text_parts).strip()
            
            # Create metadata
            metadata = {
                'model': self.model_size,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'duration_after_vad': getattr(info, 'duration_after_vad', None),
                'all_language_probs': getattr(info, 'all_language_probs', None),
                'device': self.device,
                'compute_type': self.compute_type
            }
            
            result = TranscriptionResult(
                text=full_text,
                segments=segments,
                language=info.language,
                confidence=sum(s.confidence for s in segments) / len(segments) if segments else 0.0,
                metadata=metadata
            )
            
            logger.info(f"Transcription completed. Language: {info.language}, "
                       f"Duration: {info.duration:.2f}s, Segments: {len(segments)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of language codes
        """
        # Whisper supports these languages
        return [
            'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca',
            'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr',
            'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it',
            'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv',
            'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no',
            'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 'sl', 'sn',
            'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr',
            'tt', 'uk', 'ur', 'uz', 'vi', 'yi', 'yo', 'zh'
        ]
    
    def is_healthy(self) -> bool:
        """
        Check if the transcriber is healthy and ready to use.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to load the model
            self._load_model()
            return self.model is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
