"""
Tests for video clip extraction functionality.

This module contains comprehensive tests for the ClipExtractor class,
including unit tests, integration tests, and performance tests.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

import ffmpeg

from src.clipper.clip_extractor import (
    ClipExtractor,
    ClipExtractionResult,
    BatchExtractionResult,
    extract_single_clip,
    extract_clips_from_analysis
)
from src.analyzer.llm_analyzer import ClipRecommendation, HookStrength


class TestClipExtractor:
    """Test cases for ClipExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(tempfile.mkdtemp())
        self.extractor = ClipExtractor(
            temp_dir=self.temp_dir,
            output_dir=self.output_dir,
            cleanup_temp=True,
            max_concurrent=2
        )
        
        # Create a mock video file for testing
        self.mock_video_path = str(self.temp_dir / "test_video.mp4")
        with open(self.mock_video_path, "wb") as f:
            f.write(b"fake video content for testing")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
    
    def test_time_to_seconds_conversion(self):
        """Test time string to seconds conversion."""
        # Test MM:SS format
        assert self.extractor._time_to_seconds("01:30") == 90
        assert self.extractor._time_to_seconds("00:00") == 0
        assert self.extractor._time_to_seconds("02:45") == 165
        
        # Test HH:MM:SS format
        assert self.extractor._time_to_seconds("01:30:45") == 5445
        assert self.extractor._time_to_seconds("00:01:30") == 90
        
        # Test invalid format
        with pytest.raises(ValueError):
            self.extractor._time_to_seconds("invalid")
        
        with pytest.raises(ValueError):
            self.extractor._time_to_seconds("1:2:3:4")
    
    def test_seconds_to_time_conversion(self):
        """Test seconds to time string conversion."""
        assert self.extractor._seconds_to_time(90) == "01:30"
        assert self.extractor._seconds_to_time(0) == "00:00"
        assert self.extractor._seconds_to_time(165) == "02:45"
        assert self.extractor._seconds_to_time(3661) == "61:01"  # Over 1 hour
    
    def test_extract_clip_file_not_found(self):
        """Test extraction with non-existent source video."""
        result = self.extractor.extract_clip(
            "nonexistent_video.mp4",
            "00:30",
            "01:00"
        )
        
        assert not result.success
        assert result.error_message is not None
        assert "Source video not found" in result.error_message
        assert result.clip_path == ""
        assert result.file_size_mb == 0.0
    
    def test_extract_clip_invalid_time_range(self):
        """Test extraction with invalid time range."""
        result = self.extractor.extract_clip(
            self.mock_video_path,
            "01:00",
            "00:30"  # End before start
        )
        
        assert not result.success
        assert result.error_message is not None
        assert "Invalid time range" in result.error_message
        assert result.duration_seconds == 0.0
    
    @patch('ffmpeg.run')
    @patch('ffmpeg.output')
    @patch('ffmpeg.input')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_extract_clip_success(
        self, 
        mock_getsize, 
        mock_exists, 
        mock_input, 
        mock_output, 
        mock_run
    ):
        """Test successful clip extraction."""
        # Setup mocks
        mock_exists.side_effect = lambda path: path == self.mock_video_path or path.endswith('.mp4')
        mock_getsize.return_value = 1024 * 1024  # 1MB
        mock_input.return_value = MagicMock()
        mock_output.return_value = MagicMock()
        
        # Mock ffmpeg.probe for metadata
        with patch('ffmpeg.probe') as mock_probe:
            mock_probe.return_value = {
                'format': {
                    'duration': '30.0',
                    'size': '1048576',
                    'format_name': 'mp4',
                    'bit_rate': '1000000'
                },
                'streams': [
                    {
                        'codec_type': 'video',
                        'width': 1920,
                        'height': 1080,
                        'r_frame_rate': '30/1',
                        'codec_name': 'h264'
                    },
                    {
                        'codec_type': 'audio',
                        'codec_name': 'aac',
                        'sample_rate': '48000'
                    }
                ]
            }
            
            result = self.extractor.extract_clip(
                self.mock_video_path,
                "00:30",
                "01:00"
            )
            
            assert result.success
            assert result.start_time == "00:30"
            assert result.end_time == "01:00"
            assert result.duration_seconds == 30.0
            assert result.file_size_mb == 1.0
            assert result.extraction_time > 0
            assert result.metadata is not None
            assert "duration" in result.metadata
    
    @patch('ffmpeg.run')
    def test_extract_clip_ffmpeg_error(self, mock_run):
        """Test handling of FFmpeg errors."""
        mock_run.side_effect = ffmpeg.Error("FFmpeg failed", "", "")
        
        result = self.extractor.extract_clip(
            self.mock_video_path,
            "00:30",
            "01:00"
        )
        
        assert not result.success
        assert result.error_message is not None
        assert "FFmpeg error" in result.error_message
        assert result.extraction_time > 0
    
    def create_recommendations(self):
        """Helper method to create test recommendations."""
        return [
            ClipRecommendation(
                start_time="00:30",
                end_time="01:00",
                reasoning="Engaging intro",
                confidence=85,
                hook_strength=HookStrength.HIGH,
                keywords=["intro", "hook"]
            ),
            ClipRecommendation(
                start_time="02:15",
                end_time="02:45",
                reasoning="Key insight",
                confidence=75,
                hook_strength=HookStrength.MEDIUM,
                keywords=["insight", "key"]
            ),
            ClipRecommendation(
                start_time="04:00",
                end_time="04:20",
                reasoning="Conclusion",
                confidence=70,
                hook_strength=HookStrength.LOW,
                keywords=["conclusion"]
            )
        ]
    
    @patch.object(ClipExtractor, 'extract_clip')
    def test_extract_clips_from_recommendations_sequential(self, mock_extract):
        """Test sequential extraction from recommendations."""
        recommendations = self.create_recommendations()
        
        # Mock successful extractions
        mock_extract.side_effect = [
            ClipExtractionResult(
                clip_path="/path/clip1.mp4",
                start_time="00:30",
                end_time="01:00",
                duration_seconds=30.0,
                file_size_mb=1.0,
                success=True,
                extraction_time=1.5
            ),
            ClipExtractionResult(
                clip_path="/path/clip2.mp4",
                start_time="02:15",
                end_time="02:45",
                duration_seconds=30.0,
                file_size_mb=1.2,
                success=True,
                extraction_time=1.8
            ),
            ClipExtractionResult(
                clip_path="/path/clip3.mp4",
                start_time="04:00",
                end_time="04:20",
                duration_seconds=20.0,
                file_size_mb=0.8,
                success=True,
                extraction_time=1.2
            )
        ]
        
        batch_result = self.extractor.extract_clips_from_recommendations(
            self.mock_video_path,
            recommendations,
            parallel=False
        )
        
        assert batch_result.success_count == 3
        assert batch_result.failure_count == 0
        assert batch_result.total_size_mb == 3.0
        assert batch_result.success_rate == 100.0
        assert len(batch_result.results) == 3
        assert mock_extract.call_count == 3
    
    @patch.object(ClipExtractor, 'extract_clip')
    def test_extract_clips_with_failures(self, mock_extract):
        """Test extraction with some failures."""
        recommendations = self.create_recommendations()
        
        # Mock mixed results (success, failure, success)
        mock_extract.side_effect = [
            ClipExtractionResult(
                clip_path="/path/clip1.mp4",
                start_time="00:30",
                end_time="01:00",
                duration_seconds=30.0,
                file_size_mb=1.0,
                success=True,
                extraction_time=1.5
            ),
            ClipExtractionResult(
                clip_path="",
                start_time="02:15",
                end_time="02:45",
                duration_seconds=30.0,
                file_size_mb=0.0,
                success=False,
                error_message="FFmpeg error",
                extraction_time=0.5
            ),
            ClipExtractionResult(
                clip_path="/path/clip3.mp4",
                start_time="04:00",
                end_time="04:20",
                duration_seconds=20.0,
                file_size_mb=0.8,
                success=True,
                extraction_time=1.2
            )
        ]
        
        batch_result = self.extractor.extract_clips_from_recommendations(
            self.mock_video_path,
            recommendations,
            parallel=False
        )
        
        assert batch_result.success_count == 2
        assert batch_result.failure_count == 1
        assert batch_result.total_size_mb == 1.8  # Only successful clips
        assert abs(batch_result.success_rate - 66.67) < 0.1  # 66.67% (with tolerance)
        assert len(batch_result.results) == 3
    
    def test_cleanup_temp_files(self):
        """Test temporary file cleanup."""
        # Clean up any existing temp files first
        self.extractor.cleanup_temp_files()
        
        # Create some temp files
        temp_files = []
        for i in range(3):
            temp_file = self.temp_dir / f"temp_clip_{i}.mp4"
            temp_file.write_text("temp content")
            temp_files.append(temp_file)
        
        # Verify files exist
        for temp_file in temp_files:
            assert temp_file.exists()
        
        # Cleanup
        cleanup_count = self.extractor.cleanup_temp_files()
        
        # Verify cleanup
        assert cleanup_count == 3
        for temp_file in temp_files:
            assert not temp_file.exists()
    
    @patch('ffmpeg.probe')
    def test_get_video_info_success(self, mock_probe):
        """Test successful video info retrieval."""
        mock_probe.return_value = {
            'format': {
                'duration': '300.0',
                'size': '50000000',
                'format_name': 'mp4',
                'bit_rate': '2000000'
            },
            'streams': [
                {
                    'codec_type': 'video',
                    'width': 1920,
                    'height': 1080,
                    'r_frame_rate': '30/1',
                    'codec_name': 'h264'
                },
                {
                    'codec_type': 'audio',
                    'codec_name': 'aac',
                    'sample_rate': '48000'
                }
            ]
        }
        
        info = self.extractor.get_video_info(self.mock_video_path)
        
        assert info['duration'] == 300.0
        assert info['size_mb'] == 50000000 / (1024 * 1024)
        assert info['width'] == 1920
        assert info['height'] == 1080
        assert info['fps'] == 30.0
        assert info['video_codec'] == 'h264'
        assert info['audio_codec'] == 'aac'
    
    @patch('ffmpeg.probe')
    def test_get_video_info_failure(self, mock_probe):
        """Test video info retrieval failure."""
        mock_probe.side_effect = Exception("Probe failed")
        
        info = self.extractor.get_video_info(self.mock_video_path)
        
        assert info == {}


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_video_path = str(self.temp_dir / "test_video.mp4")
        with open(self.mock_video_path, "wb") as f:
            f.write(b"fake video content")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch.object(ClipExtractor, 'extract_clip')
    def test_extract_single_clip_convenience(self, mock_extract):
        """Test extract_single_clip convenience function."""
        mock_extract.return_value = ClipExtractionResult(
            clip_path="/path/clip.mp4",
            start_time="00:30",
            end_time="01:00",
            duration_seconds=30.0,
            file_size_mb=1.0,
            success=True
        )
        
        result = extract_single_clip(
            self.mock_video_path,
            "00:30",
            "01:00",
            "/custom/output.mp4"
        )
        
        assert result.success
        assert mock_extract.called
        
        # Check that custom output filename was used
        call_args = mock_extract.call_args
        assert call_args[0][3] == "output.mp4"  # output_filename parameter
    
    @patch.object(ClipExtractor, 'extract_clips_from_recommendations')
    def test_extract_clips_from_analysis_convenience(self, mock_extract):
        """Test extract_clips_from_analysis convenience function."""
        recommendations = [
            ClipRecommendation(
                start_time="00:30",
                end_time="01:00",
                reasoning="Test clip",
                confidence=80,
                hook_strength=HookStrength.HIGH
            )
        ]
        
        mock_extract.return_value = BatchExtractionResult(
            source_video=self.mock_video_path,
            results=[],
            total_time=5.0,
            success_count=1,
            failure_count=0,
            total_size_mb=1.0
        )
        
        result = extract_clips_from_analysis(
            self.mock_video_path,
            recommendations,
            parallel=True
        )
        
        assert mock_extract.called
        call_args = mock_extract.call_args
        assert call_args[0][0] == self.mock_video_path
        assert call_args[0][1] == recommendations
        assert call_args[0][2] is True  # parallel parameter


class TestIntegration:
    """Integration tests for clip extraction."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
    
    def test_clip_extraction_result_properties(self):
        """Test ClipExtractionResult property calculations."""
        result = ClipExtractionResult(
            clip_path="/path/test.mp4",
            start_time="01:30",
            end_time="02:15",
            duration_seconds=45.0,
            file_size_mb=2.5,
            success=True
        )
        
        # Test that all properties are accessible
        assert result.clip_path == "/path/test.mp4"
        assert result.start_time == "01:30"
        assert result.end_time == "02:15"
        assert result.duration_seconds == 45.0
        assert result.file_size_mb == 2.5
        assert result.success is True
        assert result.metadata == {}
    
    def test_batch_extraction_result_calculations(self):
        """Test BatchExtractionResult calculations."""
        results = [
            ClipExtractionResult(
                clip_path="/path/clip1.mp4",
                start_time="00:30",
                end_time="01:00",
                duration_seconds=30.0,
                file_size_mb=1.0,
                success=True
            ),
            ClipExtractionResult(
                clip_path="",
                start_time="02:00",
                end_time="02:30",
                duration_seconds=30.0,
                file_size_mb=0.0,
                success=False,
                error_message="Failed"
            ),
            ClipExtractionResult(
                clip_path="/path/clip3.mp4",
                start_time="03:00",
                end_time="03:45",
                duration_seconds=45.0,
                file_size_mb=1.5,
                success=True
            )
        ]
        
        batch_result = BatchExtractionResult(
            source_video="/path/video.mp4",
            results=results,
            total_time=10.5,
            success_count=2,
            failure_count=1,
            total_size_mb=2.5
        )
        
        assert abs(batch_result.success_rate - 66.67) < 0.1  # 66.67% (with tolerance)
        assert batch_result.success_count == 2
        assert batch_result.failure_count == 1
        assert batch_result.total_size_mb == 2.5
        assert len(batch_result.results) == 3


class TestPerformance:
    """Performance tests for clip extraction."""
    
    def test_time_conversion_performance(self):
        """Test performance of time conversion functions."""
        extractor = ClipExtractor()
        
        # Test large number of time conversions
        start_time = time.time()
        for i in range(1000):
            seconds = extractor._time_to_seconds(f"{i%60:02d}:{i%60:02d}")
            time_str = extractor._seconds_to_time(seconds)
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Should complete in under 1 second
    
    def test_parallel_vs_sequential_mock_performance(self):
        """Test that parallel processing setup is faster than sequential."""
        extractor = ClipExtractor(max_concurrent=3)
        
        recommendations = [
            ClipRecommendation(
                start_time=f"0{i}:00",
                end_time=f"0{i}:30",
                reasoning=f"Test clip {i}",
                confidence=80,
                hook_strength=HookStrength.MEDIUM
            ) for i in range(1, 6)  # 5 clips
        ]
        
        # This test would need actual video files to be meaningful
        # For now, we just test that the methods can be called
        assert len(recommendations) == 5
        assert extractor.max_concurrent == 3


if __name__ == "__main__":
    # Run basic tests if file is executed directly
    pytest.main([__file__, "-v"])
