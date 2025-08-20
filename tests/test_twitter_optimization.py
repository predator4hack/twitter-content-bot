"""
Tests for Twitter video format optimization functionality.

This module contains comprehensive tests for the TwitterOptimizer class,
including unit tests, integration tests, and performance tests.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

import ffmpeg

from src.clipper.twitter_optimizer import (
    TwitterOptimizer,
    TwitterSpecs,
    TwitterAspectRatio,
    TwitterResolution,
    VideoQuality,
    OptimizationResult,
    BatchOptimizationResult,
    optimize_single_clip,
    optimize_extracted_clips
)


class TestTwitterSpecs:
    """Test cases for TwitterSpecs configuration."""
    
    def test_default_specs(self):
        """Test default Twitter specifications."""
        specs = TwitterSpecs()
        
        assert specs.max_file_size_mb == 512
        assert specs.max_duration_seconds == 140
        assert specs.video_codec == "h264"
        assert specs.audio_codec == "aac"
        assert specs.container_format == "mp4"
        assert specs.target_file_size_mb == 50
        assert specs.max_width == 1920
        assert specs.max_height == 1080
        assert specs.audio_bitrate_kbps == 128
    
    def test_custom_specs(self):
        """Test custom Twitter specifications."""
        specs = TwitterSpecs(
            max_file_size_mb=256,
            target_file_size_mb=25,
            max_bitrate_kbps=10000
        )
        
        assert specs.max_file_size_mb == 256
        assert specs.target_file_size_mb == 25
        assert specs.max_bitrate_kbps == 10000
        # Other values should remain default
        assert specs.video_codec == "h264"
        assert specs.audio_codec == "aac"


class TestTwitterOptimizer:
    """Test cases for TwitterOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(tempfile.mkdtemp())
        self.specs = TwitterSpecs(target_file_size_mb=10)  # Small target for testing
        self.optimizer = TwitterOptimizer(
            specs=self.specs,
            temp_dir=self.temp_dir,
            output_dir=self.output_dir
        )
        
        # Create a mock video file for testing
        self.mock_video_path = str(self.temp_dir / "test_video.mp4")
        with open(self.mock_video_path, "wb") as f:
            f.write(b"fake video content for testing" * 1000)  # Make it larger
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.specs.target_file_size_mb == 10
        assert self.optimizer.temp_dir == self.temp_dir
        assert self.optimizer.output_dir == self.output_dir
        assert self.temp_dir.exists()
        assert self.output_dir.exists()
    
    def test_optimize_file_not_found(self):
        """Test optimization with non-existent input file."""
        result = self.optimizer.optimize_for_twitter(
            "nonexistent_video.mp4",
            aspect_ratio=TwitterAspectRatio.LANDSCAPE
        )
        
        assert not result.success
        assert result.error_message is not None
        assert "Input file not found" in result.error_message
        assert result.optimized_path == ""
        assert result.original_size_mb == 0.0
        assert result.optimized_size_mb == 0.0
    
    @patch('ffmpeg.probe')
    @patch('ffmpeg.run')
    @patch('ffmpeg.output')
    @patch('ffmpeg.input')
    @patch('os.path.getsize')
    def test_optimize_success(
        self,
        mock_getsize,
        mock_input,
        mock_output,
        mock_run,
        mock_probe
    ):
        """Test successful Twitter optimization."""
        # Setup mocks
        def mock_getsize_func(path):
            path_str = str(path)
            if "twitter_optimized" in path_str:
                return 10 * 1024 * 1024  # 10MB for optimized file
            elif "test_video" in path_str:
                return 50 * 1024 * 1024  # 50MB for original file
            else:
                return 1 * 1024 * 1024   # 1MB for other files
        
        mock_getsize.side_effect = mock_getsize_func
        mock_input.return_value = MagicMock()
        mock_output.return_value = MagicMock()
        
        # Mock video info
        mock_probe.return_value = {
            'format': {
                'duration': '60.0',
                'size': '52428800',  # 50MB
                'format_name': 'mov,mp4,m4a,3gp,3g2,mj2',
                'bit_rate': '7000000'
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
        
        result = self.optimizer.optimize_for_twitter(
            self.mock_video_path,
            aspect_ratio=TwitterAspectRatio.LANDSCAPE,
            quality=VideoQuality.TWITTER_OPTIMIZED
        )
        
        assert result.success
        assert result.original_size_mb == 50.0
        assert result.optimized_size_mb == 10.0
        assert result.compression_ratio == 5.0
        assert result.optimization_time > 0
        assert result.metadata is not None
        assert "original_info" in result.metadata
        assert "optimized_info" in result.metadata
    
    def test_calculate_target_resolution_landscape(self):
        """Test target resolution calculation for landscape aspect ratio."""
        original_info = {'width': 1920, 'height': 1080}
        
        width, height = self.optimizer._calculate_target_resolution(
            original_info,
            TwitterAspectRatio.LANDSCAPE,
            TwitterResolution.HD_720P
        )
        
        assert width == 1280  # 720 * 16/9
        assert height == 720
    
    def test_calculate_target_resolution_square(self):
        """Test target resolution calculation for square aspect ratio."""
        original_info = {'width': 1920, 'height': 1080}
        
        width, height = self.optimizer._calculate_target_resolution(
            original_info,
            TwitterAspectRatio.SQUARE,
            TwitterResolution.HD_720P
        )
        
        assert width == 720
        assert height == 720
    
    def test_calculate_target_resolution_portrait(self):
        """Test target resolution calculation for portrait aspect ratio."""
        original_info = {'width': 1920, 'height': 1080}
        
        width, height = self.optimizer._calculate_target_resolution(
            original_info,
            TwitterAspectRatio.PORTRAIT,
            TwitterResolution.HD_720P
        )
        
        assert width == 404  # 720 * 9/16, rounded to even
        assert height == 720
    
    def test_calculate_target_resolution_original(self):
        """Test target resolution calculation preserving original aspect ratio."""
        original_info = {'width': 1920, 'height': 1080}
        
        width, height = self.optimizer._calculate_target_resolution(
            original_info,
            TwitterAspectRatio.ORIGINAL,
            TwitterResolution.HD_720P
        )
        
        # Should maintain 16:9 aspect ratio
        assert width == 1280  # 720 * (1920/1080)
        assert height == 720
    
    def test_get_quality_settings(self):
        """Test quality settings for different presets."""
        # Test high quality
        crf, bitrate = self.optimizer._get_quality_settings(
            VideoQuality.HIGH, 1920, 1080
        )
        assert crf == self.specs.crf_high
        assert bitrate is not None
        assert bitrate > 0
        
        # Test medium quality
        crf, bitrate = self.optimizer._get_quality_settings(
            VideoQuality.MEDIUM, 1920, 1080
        )
        assert crf == self.specs.crf_medium
        
        # Test Twitter optimized
        crf, bitrate = self.optimizer._get_quality_settings(
            VideoQuality.TWITTER_OPTIMIZED, 1920, 1080
        )
        assert crf == self.specs.crf_twitter
    
    def test_calculate_target_bitrate(self):
        """Test target bitrate calculation."""
        bitrate = self.optimizer._calculate_target_bitrate(1920, 1080)
        
        assert bitrate > 0
        assert bitrate <= self.specs.max_bitrate_kbps
        
        # Should be reasonable for the target file size
        expected_max = int((self.specs.target_file_size_mb * 8 * 1024) / 140)
        assert bitrate <= expected_max
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        # Good quality video
        video_info = {
            'width': 1920,
            'height': 1080,
            'bit_rate': 5000000
        }
        score = self.optimizer._calculate_quality_score(video_info, 5.0)
        assert 80 <= score <= 100
        
        # Oversized video
        score = self.optimizer._calculate_quality_score(video_info, 100.0)
        assert score <= 80  # Should be penalized for large size
        
        # Low quality video
        low_quality_info = {
            'width': 640,
            'height': 360,
            'bit_rate': 100000
        }
        score = self.optimizer._calculate_quality_score(low_quality_info, 5.0)
        assert score < 90  # Should be penalized for low bitrate
    
    def test_check_twitter_compatibility(self):
        """Test Twitter compatibility checking."""
        # Compatible video
        compatible_info = {
            'format_name': 'mov,mp4,m4a,3gp,3g2,mj2',
            'video_codec': 'h264',
            'audio_codec': 'aac',
            'width': 1920,
            'height': 1080,
            'duration': 60.0
        }
        assert self.optimizer._check_twitter_compatibility(compatible_info, 10.0)
        
        # Oversized file
        assert not self.optimizer._check_twitter_compatibility(compatible_info, 600.0)
        
        # Wrong codec
        incompatible_info = compatible_info.copy()
        incompatible_info['video_codec'] = 'vp9'
        assert not self.optimizer._check_twitter_compatibility(incompatible_info, 10.0)
        
        # Wrong format
        incompatible_info = compatible_info.copy()
        incompatible_info['format_name'] = 'avi'
        assert not self.optimizer._check_twitter_compatibility(incompatible_info, 10.0)
        
        # Too long duration
        incompatible_info = compatible_info.copy()
        incompatible_info['duration'] = 200.0
        assert not self.optimizer._check_twitter_compatibility(incompatible_info, 10.0)
    
    @patch('ffmpeg.probe')
    def test_get_video_info_success(self, mock_probe):
        """Test successful video info retrieval."""
        mock_probe.return_value = {
            'format': {
                'duration': '120.0',
                'size': '25000000',
                'format_name': 'mp4',
                'bit_rate': '1666667'
            },
            'streams': [
                {
                    'codec_type': 'video',
                    'width': 1280,
                    'height': 720,
                    'r_frame_rate': '24/1',
                    'codec_name': 'h264'
                },
                {
                    'codec_type': 'audio',
                    'codec_name': 'aac',
                    'sample_rate': '44100'
                }
            ]
        }
        
        info = self.optimizer._get_video_info(self.mock_video_path)
        
        assert info['duration'] == 120.0
        assert info['width'] == 1280
        assert info['height'] == 720
        assert info['video_codec'] == 'h264'
        assert info['audio_codec'] == 'aac'
        assert info['fps'] == 24.0
    
    @patch('ffmpeg.probe')
    def test_get_video_info_failure(self, mock_probe):
        """Test video info retrieval failure."""
        mock_probe.side_effect = Exception("Probe failed")
        
        info = self.optimizer._get_video_info(self.mock_video_path)
        
        assert info == {}
    
    def test_optimization_result_properties(self):
        """Test OptimizationResult property calculations."""
        result = OptimizationResult(
            optimized_path="/path/optimized.mp4",
            original_path="/path/original.mp4",
            original_size_mb=100.0,
            optimized_size_mb=25.0,
            compression_ratio=4.0,
            quality_score=85.0,
            twitter_compatible=True,
            success=True,
            optimization_time=5.2
        )
        
        assert result.size_reduction_percent == 75.0
        assert result.success
        assert result.optimization_time == 5.2
        assert result.metadata == {}
    
    @patch.object(TwitterOptimizer, 'optimize_for_twitter')
    def test_optimize_batch(self, mock_optimize):
        """Test batch optimization."""
        # Mock successful optimizations
        mock_optimize.side_effect = [
            OptimizationResult(
                optimized_path="/path/clip1_opt.mp4",
                original_path="/path/clip1.mp4",
                original_size_mb=50.0,
                optimized_size_mb=10.0,
                compression_ratio=5.0,
                quality_score=90.0,
                twitter_compatible=True,
                success=True,
                optimization_time=3.0
            ),
            OptimizationResult(
                optimized_path="/path/clip2_opt.mp4",
                original_path="/path/clip2.mp4",
                original_size_mb=75.0,
                optimized_size_mb=15.0,
                compression_ratio=5.0,
                quality_score=85.0,
                twitter_compatible=True,
                success=True,
                optimization_time=4.0
            ),
            OptimizationResult(
                optimized_path="",
                original_path="/path/clip3.mp4",
                original_size_mb=60.0,
                optimized_size_mb=0.0,
                compression_ratio=0.0,
                quality_score=0.0,
                twitter_compatible=False,
                success=False,
                error_message="Optimization failed",
                optimization_time=1.0
            )
        ]
        
        input_paths = ["/path/clip1.mp4", "/path/clip2.mp4", "/path/clip3.mp4"]
        batch_result = self.optimizer.optimize_batch(
            input_paths,
            TwitterAspectRatio.LANDSCAPE,
            TwitterResolution.HD_720P,
            VideoQuality.TWITTER_OPTIMIZED
        )
        
        assert batch_result.success_count == 2
        assert batch_result.failure_count == 1
        assert abs(batch_result.success_rate - 66.67) < 0.1
        assert batch_result.average_compression_ratio == 5.0
        assert batch_result.average_quality_score == 87.5
        assert batch_result.total_size_reduction_mb == 100.0  # (50-10) + (75-15)
        assert len(batch_result.results) == 3


class TestEnumTypes:
    """Test enum type definitions."""
    
    def test_twitter_aspect_ratio_enum(self):
        """Test TwitterAspectRatio enum values."""
        assert TwitterAspectRatio.LANDSCAPE.value == "16:9"
        assert TwitterAspectRatio.SQUARE.value == "1:1"
        assert TwitterAspectRatio.PORTRAIT.value == "9:16"
        assert TwitterAspectRatio.ORIGINAL.value == "original"
    
    def test_twitter_resolution_enum(self):
        """Test TwitterResolution enum values."""
        assert TwitterResolution.HD_720P.value == "720p"
        assert TwitterResolution.FHD_1080P.value == "1080p"
        assert TwitterResolution.AUTO.value == "auto"
    
    def test_video_quality_enum(self):
        """Test VideoQuality enum values."""
        assert VideoQuality.HIGH.value == "high"
        assert VideoQuality.MEDIUM.value == "medium"
        assert VideoQuality.LOW.value == "low"
        assert VideoQuality.TWITTER_OPTIMIZED.value == "twitter_optimized"


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
    
    @patch.object(TwitterOptimizer, 'optimize_for_twitter')
    def test_optimize_single_clip_convenience(self, mock_optimize):
        """Test optimize_single_clip convenience function."""
        mock_optimize.return_value = OptimizationResult(
            optimized_path="/path/optimized.mp4",
            original_path=self.mock_video_path,
            original_size_mb=50.0,
            optimized_size_mb=10.0,
            compression_ratio=5.0,
            quality_score=90.0,
            twitter_compatible=True,
            success=True
        )
        
        result = optimize_single_clip(
            self.mock_video_path,
            "/custom/output.mp4",
            TwitterAspectRatio.SQUARE,
            VideoQuality.HIGH
        )
        
        assert result.success
        assert mock_optimize.called
        
        # Check that parameters were passed correctly
        call_args = mock_optimize.call_args
        assert call_args[0][0] == self.mock_video_path  # input_path
        assert call_args[0][1] == "output.mp4"  # output_filename
        assert call_args[0][2] == TwitterAspectRatio.SQUARE  # aspect_ratio
    
    @patch.object(TwitterOptimizer, 'optimize_batch')
    def test_optimize_extracted_clips_convenience(self, mock_optimize):
        """Test optimize_extracted_clips convenience function."""
        clip_paths = ["/path/clip1.mp4", "/path/clip2.mp4"]
        
        mock_optimize.return_value = BatchOptimizationResult(
            results=[],
            total_time=10.0,
            success_count=2,
            failure_count=0,
            average_compression_ratio=4.0,
            average_quality_score=85.0,
            total_size_reduction_mb=80.0
        )
        
        result = optimize_extracted_clips(
            clip_paths,
            TwitterAspectRatio.PORTRAIT,
            VideoQuality.MEDIUM
        )
        
        assert mock_optimize.called
        call_args = mock_optimize.call_args
        assert call_args[0][0] == clip_paths
        assert call_args[0][1] == TwitterAspectRatio.PORTRAIT
        assert call_args[0][2] == TwitterResolution.AUTO
        assert call_args[0][3] == VideoQuality.MEDIUM


class TestIntegration:
    """Integration tests for Twitter optimization."""
    
    def test_batch_optimization_result_calculations(self):
        """Test BatchOptimizationResult calculations."""
        results = [
            OptimizationResult(
                optimized_path="/path/clip1.mp4",
                original_path="/path/orig1.mp4",
                original_size_mb=100.0,
                optimized_size_mb=20.0,
                compression_ratio=5.0,
                quality_score=90.0,
                twitter_compatible=True,
                success=True
            ),
            OptimizationResult(
                optimized_path="",
                original_path="/path/orig2.mp4",
                original_size_mb=80.0,
                optimized_size_mb=0.0,
                compression_ratio=0.0,
                quality_score=0.0,
                twitter_compatible=False,
                success=False,
                error_message="Failed"
            ),
            OptimizationResult(
                optimized_path="/path/clip3.mp4",
                original_path="/path/orig3.mp4",
                original_size_mb=60.0,
                optimized_size_mb=15.0,
                compression_ratio=4.0,
                quality_score=85.0,
                twitter_compatible=True,
                success=True
            )
        ]
        
        batch_result = BatchOptimizationResult(
            results=results,
            total_time=15.0,
            success_count=2,
            failure_count=1,
            average_compression_ratio=4.5,
            average_quality_score=87.5,
            total_size_reduction_mb=125.0
        )
        
        assert abs(batch_result.success_rate - 66.67) < 0.1
        assert batch_result.success_count == 2
        assert batch_result.failure_count == 1
        assert batch_result.total_time == 15.0
        assert len(batch_result.results) == 3


class TestPerformance:
    """Performance tests for Twitter optimization."""
    
    def test_resolution_calculation_performance(self):
        """Test performance of resolution calculation."""
        optimizer = TwitterOptimizer()
        original_info = {'width': 1920, 'height': 1080}
        
        # Test large number of calculations
        start_time = time.time()
        for i in range(1000):
            width, height = optimizer._calculate_target_resolution(
                original_info,
                TwitterAspectRatio.LANDSCAPE,
                TwitterResolution.HD_720P
            )
            assert width > 0 and height > 0
        
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should complete very quickly
    
    def test_quality_settings_performance(self):
        """Test performance of quality settings calculation."""
        optimizer = TwitterOptimizer()
        
        start_time = time.time()
        for i in range(1000):
            crf, bitrate = optimizer._get_quality_settings(
                VideoQuality.TWITTER_OPTIMIZED, 1920, 1080
            )
            assert crf > 0 and (bitrate is None or bitrate > 0)
        
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should complete very quickly


if __name__ == "__main__":
    # Run basic tests if file is executed directly
    pytest.main([__file__, "-v"])
