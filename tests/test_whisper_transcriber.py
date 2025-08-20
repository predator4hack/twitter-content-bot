"""
Test basic Whisper transcriber functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.transcription.whisper_transcriber import WhisperTranscriber
from src.transcription.base import TranscriptionResult, TranscriptionSegment


class TestWhisperTranscriber:
    """Test Whisper transcriber implementation."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        transcriber = WhisperTranscriber()
        assert transcriber.model_size == "base"
        assert transcriber.device in ["cpu", "cuda"]
        assert transcriber.compute_type in ["int8", "float16"]
        assert transcriber.model is None
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        transcriber = WhisperTranscriber(
            model_size="small",
            device="cpu",
            compute_type="int8",
            local_files_only=True
        )
        assert transcriber.model_size == "small"
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"
        assert transcriber.local_files_only is True
    
    def test_init_invalid_model(self):
        """Test initialization with invalid model size."""
        with pytest.raises(ValueError, match="Unsupported model size"):
            WhisperTranscriber(model_size="invalid")
    
    def test_detect_device_with_cuda(self):
        """Test device detection when CUDA is available."""
        transcriber = WhisperTranscriber()
        
        with patch('torch.cuda.is_available', return_value=True):
            device = transcriber._detect_device()
            assert device == "cuda"
    
    def test_detect_device_without_cuda(self):
        """Test device detection when CUDA is not available."""
        transcriber = WhisperTranscriber()
        
        with patch('torch.cuda.is_available', return_value=False):
            device = transcriber._detect_device()
            assert device == "cpu"
    
    def test_detect_device_no_torch(self):
        """Test device detection when torch is not available."""
        transcriber = WhisperTranscriber()
        
        with patch('builtins.__import__', side_effect=ImportError):
            device = transcriber._detect_device()
            assert device == "cpu"
    
    def test_detect_compute_type_cuda(self):
        """Test compute type detection for CUDA device."""
        transcriber = WhisperTranscriber()
        transcriber.device = "cuda"
        compute_type = transcriber._detect_compute_type()
        assert compute_type == "float16"
    
    def test_detect_compute_type_cpu(self):
        """Test compute type detection for CPU device."""
        transcriber = WhisperTranscriber()
        transcriber.device = "cpu"
        compute_type = transcriber._detect_compute_type()
        assert compute_type == "int8"
    
    @patch('src.transcription.whisper_transcriber.WhisperModel')
    @patch('src.transcription.whisper_transcriber.download_model')
    def test_load_model_success(self, mock_download, mock_model_class):
        """Test successful model loading."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        transcriber = WhisperTranscriber()
        transcriber._load_model()
        
        mock_download.assert_called_once()
        mock_model_class.assert_called_once()
        assert transcriber.model == mock_model
    
    @patch('src.transcription.whisper_transcriber.WhisperModel')
    @patch('src.transcription.whisper_transcriber.download_model')
    def test_load_model_download_failure(self, mock_download, mock_model_class):
        """Test model loading when download fails but local model exists."""
        mock_download.side_effect = Exception("Download failed")
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        transcriber = WhisperTranscriber()
        transcriber._load_model()
        
        mock_download.assert_called_once()
        mock_model_class.assert_called_once()
        assert transcriber.model == mock_model
    
    @patch('src.transcription.whisper_transcriber.WhisperModel')
    @patch('src.transcription.whisper_transcriber.download_model')
    def test_load_model_failure(self, mock_download, mock_model_class):
        """Test model loading failure."""
        mock_model_class.side_effect = Exception("Model loading failed")
        
        transcriber = WhisperTranscriber()
        
        with pytest.raises(RuntimeError, match="Could not load Whisper model"):
            transcriber._load_model()
    
    @patch('src.transcription.whisper_transcriber.ffmpeg')
    def test_extract_audio_success(self, mock_ffmpeg):
        """Test successful audio extraction from video."""
        mock_ffmpeg.input.return_value = Mock()
        mock_ffmpeg.output.return_value = Mock()
        mock_ffmpeg.run.return_value = None
        
        transcriber = WhisperTranscriber()
        result = transcriber._extract_audio_from_video("/path/to/video.mp4", "/tmp")
        
        assert result == "/tmp/audio.wav"
        mock_ffmpeg.input.assert_called_once_with("/path/to/video.mp4")
        mock_ffmpeg.run.assert_called_once()
    
    @patch('src.transcription.whisper_transcriber.ffmpeg')
    def test_extract_audio_failure(self, mock_ffmpeg):
        """Test audio extraction failure."""
        mock_ffmpeg.run.side_effect = Exception("FFmpeg failed")
        
        transcriber = WhisperTranscriber()
        
        with pytest.raises(RuntimeError, match="Failed to extract audio"):
            transcriber._extract_audio_from_video("/path/to/video.mp4", "/tmp")
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('src.transcription.whisper_transcriber.av')
    def test_validate_audio_file_success(self, mock_av, mock_getsize, mock_exists):
        """Test successful audio file validation."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1000
        
        mock_container = Mock()
        mock_container.streams.audio = [Mock()]
        mock_av.open.return_value.__enter__.return_value = mock_container
        
        transcriber = WhisperTranscriber()
        transcriber._validate_audio_file("/path/to/audio.wav")  # Should not raise
    
    @patch('os.path.exists')
    def test_validate_audio_file_not_found(self, mock_exists):
        """Test audio file validation when file doesn't exist."""
        mock_exists.return_value = False
        
        transcriber = WhisperTranscriber()
        
        with pytest.raises(FileNotFoundError):
            transcriber._validate_audio_file("/path/to/audio.wav")
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_validate_audio_file_empty(self, mock_getsize, mock_exists):
        """Test audio file validation when file is empty."""
        mock_exists.return_value = True
        mock_getsize.return_value = 0
        
        transcriber = WhisperTranscriber()
        
        with pytest.raises(RuntimeError, match="Audio file is empty"):
            transcriber._validate_audio_file("/path/to/audio.wav")
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        transcriber = WhisperTranscriber()
        languages = transcriber.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert 'en' in languages
        assert 'es' in languages
        assert 'fr' in languages
    
    @patch.object(WhisperTranscriber, '_load_model')
    def test_is_healthy_success(self, mock_load):
        """Test health check when transcriber is healthy."""
        transcriber = WhisperTranscriber()
        transcriber.model = Mock()
        
        assert transcriber.is_healthy() is True
        mock_load.assert_called_once()
    
    @patch.object(WhisperTranscriber, '_load_model')
    def test_is_healthy_failure(self, mock_load):
        """Test health check when transcriber is not healthy."""
        mock_load.side_effect = Exception("Model loading failed")
        
        transcriber = WhisperTranscriber()
        
        assert transcriber.is_healthy() is False
    
    @patch('os.path.exists')
    def test_transcribe_file_not_found(self, mock_exists):
        """Test transcription when file doesn't exist."""
        mock_exists.return_value = False
        
        transcriber = WhisperTranscriber()
        
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe_file("/path/to/nonexistent.wav")
    
    @patch('os.path.exists')
    @patch.object(WhisperTranscriber, '_load_model')
    @patch.object(WhisperTranscriber, '_validate_audio_file')
    @patch.object(WhisperTranscriber, '_transcribe_audio')
    def test_transcribe_audio_file_success(self, mock_transcribe, mock_validate, mock_load, mock_exists):
        """Test successful transcription of audio file."""
        mock_exists.return_value = True
        
        expected_result = TranscriptionResult(
            text="Hello world",
            segments=[],
            language="en",
            confidence=0.9
        )
        mock_transcribe.return_value = expected_result
        
        transcriber = WhisperTranscriber()
        result = transcriber.transcribe_file("/path/to/audio.wav")
        
        assert result == expected_result
        mock_load.assert_called_once()
        mock_validate.assert_called_once_with("/path/to/audio.wav")
        mock_transcribe.assert_called_once()
    
    @patch('os.path.exists')
    @patch.object(WhisperTranscriber, '_load_model')
    @patch.object(WhisperTranscriber, '_extract_audio_from_video')
    @patch.object(WhisperTranscriber, '_validate_audio_file')
    @patch.object(WhisperTranscriber, '_transcribe_audio')
    def test_transcribe_video_file_success(self, mock_transcribe, mock_validate, mock_extract, mock_load, mock_exists):
        """Test successful transcription of video file."""
        mock_exists.return_value = True
        mock_extract.return_value = "/tmp/audio.wav"
        
        expected_result = TranscriptionResult(
            text="Hello world",
            segments=[],
            language="en",
            confidence=0.9
        )
        mock_transcribe.return_value = expected_result
        
        transcriber = WhisperTranscriber()
        result = transcriber.transcribe_file("/path/to/video.mp4")
        
        assert result == expected_result
        mock_load.assert_called_once()
        mock_extract.assert_called_once()
        mock_validate.assert_called_once_with("/tmp/audio.wav")
        mock_transcribe.assert_called_once()
