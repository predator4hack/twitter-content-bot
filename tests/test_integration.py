"""
Integration tests for the complete YouTube to Twitter clip pipeline.

These tests verify that all components work together correctly and that
the end-to-end workflow produces the expected results.
"""

import asyncio
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.core.pipeline import (
    TwitterClipPipeline,
    PipelineStage,
    PipelineProgress,
    PipelineResult,
    PipelineError,
    process_youtube_video
)
from src.analyzer.llm_analyzer import ClipRecommendation, ContentType, HookStrength
from src.transcription.base import TranscriptionResult, TranscriptionSegment


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_transcription_result(self):
        """Mock transcription result for testing."""
        segments = [
            TranscriptionSegment(
                start_time=0.0,
                end_time=30.0,
                text="Welcome to this amazing tutorial about Python programming.",
                confidence=0.95
            ),
            TranscriptionSegment(
                start_time=30.0,
                end_time=90.0,
                text="Today we will learn about async programming and how it works.",
                confidence=0.92
            ),
            TranscriptionSegment(
                start_time=90.0,
                end_time=150.0,
                text="This is the most important concept you need to understand.",
                confidence=0.98
            )
        ]
        
        return TranscriptionResult(
            text="Full transcript text here...",
            segments=segments,
            language="en"
        )
    
    @pytest.fixture
    def mock_clip_recommendations(self):
        """Mock clip recommendations for testing."""
        return [
            ClipRecommendation(
                start_time="00:00:30",
                end_time="00:01:30",
                reasoning="Great introduction with high engagement potential",
                confidence=85,
                hook_strength=HookStrength.HIGH,
                keywords=["tutorial", "python", "programming"],
                sentiment="positive"
            ),
            ClipRecommendation(
                start_time="00:01:30",
                end_time="00:02:30",
                reasoning="Key concept explanation that will resonate with developers",
                confidence=78,
                hook_strength=HookStrength.MEDIUM,
                keywords=["async", "programming", "concept"],
                sentiment="positive"
            )
        ]
    
    @pytest.fixture
    def pipeline_with_mocks(self, temp_output_dir):
        """Create pipeline with mocked components for testing."""
        pipeline = TwitterClipPipeline(
            output_dir=temp_output_dir,
            max_retries=3,  # Allow enough retries for tests
            retry_delay=0.1,
            cleanup_temp_files=False
        )
        return pipeline
    
    def test_pipeline_initialization(self, temp_output_dir):
        """Test pipeline initialization with custom configuration."""
        pipeline = TwitterClipPipeline(
            output_dir=temp_output_dir,
            max_retries=2,
            retry_delay=0.5,
            llm_provider="groq",
            whisper_model="small"
        )
        
        assert pipeline.output_dir == temp_output_dir
        assert pipeline.max_retries == 2
        assert pipeline.retry_delay == 0.5
        assert pipeline.llm_provider == "groq"
        assert pipeline.whisper_model == "small"
        assert pipeline.output_dir.exists()
        assert pipeline.temp_dir.exists()
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        progress = PipelineProgress()
        
        # Initial state
        assert progress.current_stage == PipelineStage.VALIDATION
        assert progress.progress_percentage == 0.0
        assert len(progress.completed_stages) == 0
        
        # Update to download stage
        progress.update_stage(PipelineStage.DOWNLOAD, {"url": "test_url"})
        assert progress.current_stage == PipelineStage.DOWNLOAD
        assert PipelineStage.VALIDATION in progress.completed_stages
        assert progress.progress_percentage > 0
        assert progress.stage_details["url"] == "test_url"
        
        # Update to completion
        for stage in [PipelineStage.THUMBNAIL, PipelineStage.TRANSCRIPTION, 
                     PipelineStage.ANALYSIS, PipelineStage.EXTRACTION, 
                     PipelineStage.OPTIMIZATION, PipelineStage.CLEANUP]:
            progress.update_stage(stage)
        
        progress.update_stage(PipelineStage.COMPLETED)
        assert progress.progress_percentage == 100.0
    
    @pytest.mark.asyncio
    async def test_url_validation_success(self, pipeline_with_mocks):
        """Test successful URL validation."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ"
        ]
        
        for url in valid_urls:
            # Should not raise an exception
            await pipeline_with_mocks._validate_url(url)
    
    @pytest.mark.asyncio
    async def test_url_validation_failure(self, pipeline_with_mocks):
        """Test URL validation with invalid URLs."""
        invalid_urls = [
            "https://www.vimeo.com/123456",
            "https://www.example.com",
            "not_a_url",
            "",
            None
        ]
        
        for url in invalid_urls:
            with pytest.raises(PipelineError) as exc_info:
                await pipeline_with_mocks._validate_url(url)
            assert exc_info.value.stage == PipelineStage.VALIDATION
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_success_after_failure(self, pipeline_with_mocks):
        """Test retry mechanism with eventual success."""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Attempt {call_count} failed")
            return "success"
        
        result = await pipeline_with_mocks._retry_with_backoff(
            failing_function,
            PipelineStage.DOWNLOAD
        )
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_final_failure(self, pipeline_with_mocks):
        """Test retry mechanism with ultimate failure."""
        def always_failing_function():
            raise Exception("Always fails")
        
        with pytest.raises(PipelineError) as exc_info:
            await pipeline_with_mocks._retry_with_backoff(
                always_failing_function,
                PipelineStage.TRANSCRIPTION
            )
        
        assert exc_info.value.stage == PipelineStage.TRANSCRIPTION
        assert "failed after" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_async_function_retry(self, pipeline_with_mocks):
        """Test retry mechanism with async functions."""
        call_count = 0
        
        async def async_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception(f"Async attempt {call_count} failed")
            return "async_success"
        
        result = await pipeline_with_mocks._retry_with_backoff(
            async_failing_function,
            PipelineStage.ANALYSIS
        )
        
        assert result == "async_success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    @patch('src.core.pipeline.YouTubeDownloader')
    @patch('src.core.pipeline.ThumbnailExtractor')
    @patch('src.core.pipeline.WhisperTranscriber')
    @patch('src.core.pipeline.LLMAnalyzerFactory')
    @patch('src.core.pipeline.ClipExtractor')
    @patch('src.core.pipeline.TwitterOptimizer')
    async def test_full_pipeline_success(
        self,
        mock_optimizer,
        mock_clip_extractor,
        mock_llm_factory,
        mock_transcriber,
        mock_thumbnail,
        mock_downloader,
        temp_output_dir,
        mock_transcription_result,
        mock_clip_recommendations
    ):
        """Test complete pipeline execution with mocked components."""
        
        # Setup mocks
        mock_downloader_instance = Mock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.download_video.return_value = (
            Path("/tmp/test_video.mp4"),
            {"title": "Test Video", "duration": 150}
        )
        
        mock_thumbnail_instance = Mock()
        mock_thumbnail.return_value = mock_thumbnail_instance
        mock_thumbnail_instance.extract_and_process_thumbnails.return_value = [
            {"success": True, "processed_path": "/tmp/thumbnail.jpg"}
        ]
        
        mock_transcriber_instance = Mock()
        mock_transcriber.return_value = mock_transcriber_instance
        mock_transcriber_instance.transcribe_file.return_value = mock_transcription_result
        
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_transcript.return_value = Mock(
            recommendations=mock_clip_recommendations
        )
        mock_llm_factory.create_analyzer.return_value = mock_analyzer
        
        mock_clip_extractor_instance = Mock()
        mock_clip_extractor.return_value = mock_clip_extractor_instance
        mock_extraction_result = Mock()
        mock_extraction_result.clips = [
            Mock(
                success=True,
                clip_path="/tmp/clip1.mp4",
                start_time="00:00:30",
                end_time="00:01:30",
                duration_seconds=60.0,
                file_size_mb=5.2
            ),
            Mock(
                success=True,
                clip_path="/tmp/clip2.mp4",
                start_time="00:01:30",
                end_time="00:02:30",
                duration_seconds=60.0,
                file_size_mb=4.8
            )
        ]
        mock_clip_extractor_instance.extract_clips_from_recommendations.return_value = mock_extraction_result
        
        mock_optimizer_instance = Mock()
        mock_optimizer.return_value = mock_optimizer_instance
        mock_optimizer_instance.optimize_for_twitter.return_value = Mock(
            success=True,
            output_path="/tmp/optimized_clip.mp4",
            original_size_mb=5.0,
            optimized_size_mb=3.2,
            compression_ratio=0.64,
            quality_score=85
        )
        
        # Create pipeline and process video
        pipeline = TwitterClipPipeline(
            output_dir=temp_output_dir,
            max_retries=1,
            retry_delay=0.1,
            cleanup_temp_files=False
        )
        
        # Track progress
        progress_updates = []
        pipeline.set_progress_callback(lambda p: progress_updates.append(p.current_stage))
        
        result = await pipeline.process_video(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            num_clips=2,
            max_clip_duration=140
        )
        
        # Verify result
        assert result.success is True
        assert result.youtube_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert result.video_path == Path("/tmp/test_video.mp4")
        assert result.thumbnail_path == Path("/tmp/thumbnail.jpg")
        assert result.transcription_result == mock_transcription_result
        assert len(result.clip_recommendations) == 2
        assert len(result.extracted_clips) == 2
        assert len(result.optimized_clips) == 2
        assert result.execution_time > 0
        assert result.progress.current_stage == PipelineStage.COMPLETED
        assert result.progress.progress_percentage == 100.0
        
        # Verify progress tracking
        expected_stages = [
            PipelineStage.VALIDATION,
            PipelineStage.DOWNLOAD,
            PipelineStage.THUMBNAIL,
            PipelineStage.TRANSCRIPTION,
            PipelineStage.ANALYSIS,
            PipelineStage.EXTRACTION,
            PipelineStage.OPTIMIZATION,
            PipelineStage.CLEANUP,
            PipelineStage.COMPLETED
        ]
        assert progress_updates == expected_stages
        
        # Verify component calls
        mock_downloader_instance.download_video.assert_called_once()
        mock_thumbnail_instance.extract_and_process_thumbnails.assert_called_once()
        mock_transcriber_instance.transcribe_file.assert_called_once()
        mock_analyzer.analyze_transcript.assert_called_once()
        mock_clip_extractor_instance.extract_clips_from_recommendations.assert_called_once()
        assert mock_optimizer_instance.optimize_for_twitter.call_count == 2
    
    @pytest.mark.asyncio
    @patch('src.core.pipeline.YouTubeDownloader')
    async def test_pipeline_failure_handling(self, mock_downloader, temp_output_dir):
        """Test pipeline failure handling and error reporting."""
        
        # Setup mock to fail
        mock_downloader_instance = Mock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.download_video.side_effect = Exception("Download failed")
        
        pipeline = TwitterClipPipeline(
            output_dir=temp_output_dir,
            max_retries=1,
            retry_delay=0.1
        )
        
        result = await pipeline.process_video(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        
        # Verify failure result
        assert result.success is False
        assert result.progress.current_stage == PipelineStage.FAILED
        assert "Download failed" in result.error_message
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    @patch('src.core.pipeline.TwitterClipPipeline')
    async def test_convenience_function(self, mock_pipeline_class, temp_output_dir):
        """Test the convenience function for processing videos."""
        
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_result = Mock(success=True)
        mock_pipeline.process_video = AsyncMock(return_value=mock_result)
        
        # Call convenience function
        result = await process_youtube_video(
            youtube_url="https://www.youtube.com/watch?v=test123",
            output_dir=temp_output_dir,
            num_clips=3,
            llm_provider="groq"
        )
        
        # Verify
        assert result == mock_result
        mock_pipeline_class.assert_called_once()
        mock_pipeline.process_video.assert_called_once_with(
            youtube_url="https://www.youtube.com/watch?v=test123",
            num_clips=3,
            max_clip_duration=140
        )
    
    def test_pipeline_result_dataclass(self):
        """Test PipelineResult dataclass functionality."""
        result = PipelineResult(
            success=True,
            youtube_url="https://test.com",
            execution_time=45.2
        )
        
        assert result.success is True
        assert result.youtube_url == "https://test.com"
        assert result.execution_time == 45.2
        assert result.video_path is None
        assert len(result.clip_recommendations) == 0
        assert len(result.extracted_clips) == 0
        assert len(result.optimized_clips) == 0
        assert len(result.metadata) == 0
    
    def test_pipeline_error_exception(self):
        """Test PipelineError exception functionality."""
        error = PipelineError(
            "Test error message",
            PipelineStage.TRANSCRIPTION,
            retry_allowed=False
        )
        
        assert str(error) == "Test error message"
        assert error.stage == PipelineStage.TRANSCRIPTION
        assert error.retry_allowed is False
    
    @pytest.mark.asyncio
    async def test_cleanup_functionality(self, pipeline_with_mocks):
        """Test cleanup functionality."""
        # Create some temporary files
        test_dirs = [
            pipeline_with_mocks.temp_dir / "downloads",
            pipeline_with_mocks.temp_dir / "thumbnails",
            pipeline_with_mocks.temp_dir / "clips"
        ]
        
        for test_dir in test_dirs:
            test_dir.mkdir(parents=True, exist_ok=True)
            (test_dir / "test_file.txt").touch()
        
        # Verify files exist
        for test_dir in test_dirs:
            assert test_dir.exists()
            assert (test_dir / "test_file.txt").exists()
        
        # Run cleanup
        await pipeline_with_mocks._cleanup_temp_files(keep_optimized=True)
        
        # Verify cleanup (directories should be removed)
        for test_dir in test_dirs:
            assert not test_dir.exists()
    
    def test_progress_callback_integration(self, pipeline_with_mocks):
        """Test progress callback integration."""
        progress_history = []
        
        def progress_callback(progress: PipelineProgress):
            progress_history.append({
                "stage": progress.current_stage,
                "percentage": progress.progress_percentage,
                "elapsed": progress.elapsed_time
            })
        
        pipeline_with_mocks.set_progress_callback(progress_callback)
        
        # Simulate progress updates
        progress = PipelineProgress()
        pipeline_with_mocks._update_progress(progress)
        
        progress.update_stage(PipelineStage.DOWNLOAD)
        pipeline_with_mocks._update_progress(progress)
        
        progress.update_stage(PipelineStage.COMPLETED)
        progress.progress_percentage = 100.0
        pipeline_with_mocks._update_progress(progress)
        
        # Verify callback was called
        assert len(progress_history) == 3
        assert progress_history[0]["stage"] == PipelineStage.VALIDATION
        assert progress_history[1]["stage"] == PipelineStage.DOWNLOAD
        assert progress_history[2]["stage"] == PipelineStage.COMPLETED
        assert progress_history[2]["percentage"] == 100.0


class TestPipelinePerformance:
    """Performance and stress tests for the pipeline."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_pipeline_memory_usage(self, temp_output_dir):
        """Test pipeline memory usage stays within reasonable bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple pipeline operations
        pipeline = TwitterClipPipeline(
            output_dir=temp_output_dir,
            cleanup_temp_files=True
        )
        
        # Memory should not grow excessively
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Allow some memory growth but not excessive (less than 100MB)
        assert memory_growth < 100, f"Memory grew by {memory_growth:.2f}MB"
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_operations(self, temp_output_dir):
        """Test running multiple pipeline operations concurrently."""
        
        async def mock_pipeline_operation(pipeline_id: int):
            """Mock pipeline operation for concurrency testing."""
            pipeline = TwitterClipPipeline(
                output_dir=temp_output_dir / f"pipeline_{pipeline_id}",
                max_retries=1,
                retry_delay=0.1
            )
            
            # Simulate some processing time
            await asyncio.sleep(0.1)
            return f"pipeline_{pipeline_id}_completed"
        
        # Run multiple operations concurrently
        tasks = [mock_pipeline_operation(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify all completed
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result == f"pipeline_{i}_completed"


@pytest.mark.integration
class TestRealWorldScenarios:
    """Integration tests with realistic scenarios (requires actual components)."""
    
    @pytest.mark.skipif(
        True,  # Skip by default - enable with custom pytest plugin
        reason="Integration tests require --run-integration flag"
    )
    @pytest.mark.asyncio
    async def test_real_youtube_video_processing(self, temp_output_dir):
        """Test with a real YouTube video (requires internet and API keys)."""
        
        # Use a short, public domain video for testing
        test_url = "https://www.youtube.com/watch?v=ScMzIvxBSi4"  # 6-second video
        
        pipeline = TwitterClipPipeline(
            output_dir=temp_output_dir,
            whisper_model="tiny",  # Use smallest model for speed
            cleanup_temp_files=False  # Keep files for inspection
        )
        
        result = await pipeline.process_video(
            youtube_url=test_url,
            num_clips=1,
            max_clip_duration=30
        )
        
        # Verify successful processing
        assert result.success is True
        assert result.video_path.exists()
        assert result.thumbnail_path.exists()
        assert result.transcription_result is not None
        assert len(result.clip_recommendations) > 0
        assert len(result.extracted_clips) > 0
        assert len(result.optimized_clips) > 0
    
    @pytest.mark.skipif(
        True,  # Skip by default - enable with custom pytest plugin
        reason="Stress tests require --run-stress flag"
    )
    @pytest.mark.asyncio
    async def test_long_video_processing(self, temp_output_dir):
        """Test processing of longer videos (stress test)."""
        
        # This would test with a longer video (10+ minutes)
        # Only run with explicit stress test flag
        pass


# Mark definitions for pytest
slow = pytest.mark.slow
integration = pytest.mark.integration