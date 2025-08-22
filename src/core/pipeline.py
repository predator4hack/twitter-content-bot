"""
End-to-End Pipeline for YouTube to Twitter Clip Extraction.

This module orchestrates the complete workflow from YouTube URL to optimized Twitter clips,
integrating all components with proper error handling, progress tracking, and retry mechanisms.
"""

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
from concurrent.futures import ThreadPoolExecutor
import shutil

from ..core.config import config
from ..core.logger import get_logger
from ..downloader.youtube_downloader import YouTubeDownloader, YouTubeURLValidator
from ..downloader.thumbnail_extractor import ThumbnailExtractor
from ..transcription.whisper_transcriber import WhisperTranscriber
from ..analyzer.llm_analyzer import LLMAnalyzerFactory, ClipRecommendation
from ..clipper.clip_extractor import ClipExtractor
from ..clipper.twitter_optimizer import TwitterOptimizer

logger = get_logger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    VALIDATION = "validation"
    DOWNLOAD = "download"
    THUMBNAIL = "thumbnail"
    TRANSCRIPTION = "transcription"
    ANALYSIS = "analysis"
    EXTRACTION = "extraction"
    OPTIMIZATION = "optimization"
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineProgress:
    """Tracks pipeline execution progress."""
    current_stage: PipelineStage = PipelineStage.VALIDATION
    completed_stages: List[PipelineStage] = field(default_factory=list)
    progress_percentage: float = 0.0
    stage_details: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    stage_start_time: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    
    def update_stage(self, stage: PipelineStage, details: Optional[Dict[str, Any]] = None):
        """Update current stage and progress."""
        if self.current_stage != PipelineStage.FAILED:
            if self.current_stage not in self.completed_stages:
                self.completed_stages.append(self.current_stage)
        
        self.current_stage = stage
        self.stage_start_time = time.time()
        
        if details:
            self.stage_details.update(details)
        
        # Calculate progress percentage (8 total stages)
        stage_weights = {
            PipelineStage.VALIDATION: 5,
            PipelineStage.DOWNLOAD: 15,
            PipelineStage.THUMBNAIL: 5,
            PipelineStage.TRANSCRIPTION: 25,
            PipelineStage.ANALYSIS: 20,
            PipelineStage.EXTRACTION: 20,
            PipelineStage.OPTIMIZATION: 5,
            PipelineStage.CLEANUP: 5
        }
        
        completed_weight = sum(stage_weights.get(stage, 0) for stage in self.completed_stages)
        self.progress_percentage = min(completed_weight, 100.0)
    
    @property
    def elapsed_time(self) -> float:
        """Total elapsed time since pipeline start."""
        return time.time() - self.start_time
    
    @property
    def stage_elapsed_time(self) -> float:
        """Elapsed time for current stage."""
        return time.time() - self.stage_start_time


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    success: bool
    youtube_url: str
    video_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None
    transcription_result: Optional[Any] = None
    clip_recommendations: List[ClipRecommendation] = field(default_factory=list)
    extracted_clips: List[Dict[str, Any]] = field(default_factory=list)
    optimized_clips: List[Dict[str, Any]] = field(default_factory=list)
    progress: Optional[PipelineProgress] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    
    def __init__(self, message: str, stage: PipelineStage, retry_allowed: bool = True):
        super().__init__(message)
        self.stage = stage
        self.retry_allowed = retry_allowed


class TwitterClipPipeline:
    """
    End-to-end pipeline for converting YouTube videos to Twitter clips.
    
    This class orchestrates the complete workflow with proper error handling,
    progress tracking, and retry mechanisms.
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        temp_dir: Optional[Path] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_progress_callback: bool = True,
        llm_provider: str = "gemini",
        whisper_model: str = "base",
        cleanup_temp_files: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Directory for final outputs
            temp_dir: Directory for temporary files
            max_retries: Maximum retry attempts per stage
            retry_delay: Delay between retries (seconds)
            enable_progress_callback: Enable progress tracking
            llm_provider: LLM provider ("gemini" or "groq")
            whisper_model: Whisper model size
            cleanup_temp_files: Clean up temporary files after completion
        """
        self.output_dir = output_dir or Path("outputs")
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "twitter_clip_pipeline"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_progress_callback = enable_progress_callback
        self.llm_provider = llm_provider
        self.whisper_model = whisper_model
        self.cleanup_temp_files = cleanup_temp_files
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        # Progress callback
        self.progress_callback: Optional[Callable[[PipelineProgress], None]] = None
    
    def _init_components(self):
        """Initialize all pipeline components."""
        try:
            self.downloader = YouTubeDownloader(
                output_dir=self.temp_dir / "downloads"
            )
            
            self.thumbnail_extractor = ThumbnailExtractor(
                output_dir=self.temp_dir / "thumbnails"
            )
            
            self.transcriber = WhisperTranscriber(
                model_size=self.whisper_model,
                device="auto"
            )
            
            self.clip_extractor = ClipExtractor(
                output_dir=self.temp_dir / "clips"
            )
            
            self.twitter_optimizer = TwitterOptimizer(
                output_dir=self.output_dir / "optimized_clips"
            )
            
            logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise PipelineError(
                f"Component initialization failed: {e}",
                PipelineStage.VALIDATION,
                retry_allowed=False
            )
    
    def set_progress_callback(self, callback: Callable[[PipelineProgress], None]):
        """Set progress callback function."""
        self.progress_callback = callback
    
    def _update_progress(self, progress: PipelineProgress):
        """Update progress and call callback if enabled."""
        if self.enable_progress_callback and self.progress_callback:
            self.progress_callback(progress)
    
    async def _retry_with_backoff(
        self,
        func: Callable,
        stage: PipelineStage,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic and exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"Stage {stage.value} failed after {self.max_retries} retries: {e}")
                    break
                
                retry_delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Stage {stage.value} attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
        
        raise PipelineError(
            f"Stage {stage.value} failed after {self.max_retries} retries: {last_exception}",
            stage,
            retry_allowed=False
        )
    
    async def process_video(
        self,
        youtube_url: str,
        num_clips: int = 2,
        max_clip_duration: int = 140,
        twitter_strategy: str = "viral"
    ) -> PipelineResult:
        """
        Process a YouTube video through the complete pipeline.
        
        Args:
            youtube_url: YouTube video URL
            num_clips: Number of clips to extract
            max_clip_duration: Maximum clip duration in seconds
            twitter_strategy: Twitter content strategy
            
        Returns:
            PipelineResult with all outputs and metadata
        """
        start_time = time.time()
        progress = PipelineProgress()
        result = PipelineResult(
            success=False,
            youtube_url=youtube_url,
            progress=progress
        )
        
        try:
            # Stage 1: URL Validation
            progress.update_stage(PipelineStage.VALIDATION)
            self._update_progress(progress)
            
            await self._retry_with_backoff(
                self._validate_url,
                PipelineStage.VALIDATION,
                youtube_url
            )
            
            # Stage 2: Video Download
            progress.update_stage(PipelineStage.DOWNLOAD)
            self._update_progress(progress)
            
            video_path, video_metadata = await self._retry_with_backoff(
                self._download_video,
                PipelineStage.DOWNLOAD,
                youtube_url
            )
            result.video_path = video_path
            result.metadata.update(video_metadata)
            
            # Stage 3: Thumbnail Extraction
            progress.update_stage(PipelineStage.THUMBNAIL)
            self._update_progress(progress)
            
            thumbnail_path = await self._retry_with_backoff(
                self._extract_thumbnail,
                PipelineStage.THUMBNAIL,
                youtube_url
            )
            result.thumbnail_path = thumbnail_path
            
            # Stage 4: Transcription
            progress.update_stage(PipelineStage.TRANSCRIPTION)
            self._update_progress(progress)
            
            transcription_result = await self._retry_with_backoff(
                self._transcribe_video,
                PipelineStage.TRANSCRIPTION,
                str(video_path)
            )
            result.transcription_result = transcription_result
            
            # Stage 5: LLM Analysis
            progress.update_stage(PipelineStage.ANALYSIS)
            self._update_progress(progress)
            
            clip_recommendations = await self._retry_with_backoff(
                self._analyze_content,
                PipelineStage.ANALYSIS,
                transcription_result,
                num_clips,
                twitter_strategy
            )
            result.clip_recommendations = clip_recommendations
            
            # Stage 6: Clip Extraction
            progress.update_stage(PipelineStage.EXTRACTION)
            self._update_progress(progress)
            
            extracted_clips = await self._retry_with_backoff(
                self._extract_clips,
                PipelineStage.EXTRACTION,
                str(video_path),
                clip_recommendations
            )
            result.extracted_clips = extracted_clips
            
            # Stage 7: Twitter Optimization
            progress.update_stage(PipelineStage.OPTIMIZATION)
            self._update_progress(progress)
            
            optimized_clips = await self._retry_with_backoff(
                self._optimize_clips,
                PipelineStage.OPTIMIZATION,
                extracted_clips,
                max_clip_duration
            )
            result.optimized_clips = optimized_clips
            
            # Stage 8: Cleanup
            progress.update_stage(PipelineStage.CLEANUP)
            self._update_progress(progress)
            
            if self.cleanup_temp_files:
                await self._cleanup_temp_files(keep_optimized=True)
            
            # Pipeline completed successfully
            progress.update_stage(PipelineStage.COMPLETED)
            progress.progress_percentage = 100.0
            self._update_progress(progress)
            
            result.success = True
            result.execution_time = time.time() - start_time
            
            logger.info(f"Pipeline completed successfully in {result.execution_time:.2f}s")
            
        except PipelineError as e:
            progress.current_stage = PipelineStage.FAILED
            progress.error_message = str(e)
            result.error_message = str(e)
            result.execution_time = time.time() - start_time
            self._update_progress(progress)
            
            logger.error(f"Pipeline failed at stage {e.stage.value}: {e}")
            
        except Exception as e:
            progress.current_stage = PipelineStage.FAILED
            progress.error_message = str(e)
            result.error_message = f"Unexpected error: {e}"
            result.execution_time = time.time() - start_time
            self._update_progress(progress)
            
            logger.error(f"Pipeline failed with unexpected error: {e}")
        
        return result
    
    async def _validate_url(self, url: str):
        """Validate YouTube URL."""
        if not YouTubeURLValidator.is_valid_youtube_url(url):
            raise PipelineError(
                f"Invalid YouTube URL: {url}",
                PipelineStage.VALIDATION,
                retry_allowed=False
            )
        
        logger.info(f"URL validation successful: {url}")
    
    async def _download_video(self, url: str) -> tuple:
        """Download video and return path and metadata."""
        try:
            video_path, metadata = self.downloader.download_video(url)
            logger.info(f"Video downloaded successfully: {video_path}")
            return video_path, metadata
        except Exception as e:
            raise PipelineError(f"Video download failed: {e}", PipelineStage.DOWNLOAD)
    
    async def _extract_thumbnail(self, url: str) -> Path:
        """Extract thumbnail and return path."""
        try:
            thumbnail_results = self.thumbnail_extractor.extract_and_process_thumbnails([url])
            if thumbnail_results and thumbnail_results[0].get("success"):
                thumbnail_path = Path(thumbnail_results[0]["processed_path"])
                logger.info(f"Thumbnail extracted successfully: {thumbnail_path}")
                return thumbnail_path
            else:
                raise Exception("Thumbnail extraction returned no results")
        except Exception as e:
            raise PipelineError(f"Thumbnail extraction failed: {e}", PipelineStage.THUMBNAIL)
    
    async def _transcribe_video(self, video_path: str):
        """Transcribe video and return transcription result."""
        try:
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                transcription_result = await loop.run_in_executor(
                    executor,
                    self.transcriber.transcribe_file,
                    video_path
                )
            
            logger.info(f"Transcription completed: {len(transcription_result.segments)} segments")
            return transcription_result
        except Exception as e:
            raise PipelineError(f"Transcription failed: {e}", PipelineStage.TRANSCRIPTION)
    
    async def _analyze_content(self, transcription_result, num_clips: int, strategy: str):
        """Analyze content with LLM and return recommendations."""
        try:
            analyzer = LLMAnalyzerFactory.create_analyzer(self.llm_provider)
            
            recommendations = await analyzer.analyze_transcript(
                transcription_result,
                max_clips=num_clips,
                strategy=strategy
            )
            
            logger.info(f"Content analysis completed: {len(recommendations.recommendations)} recommendations")
            return recommendations.recommendations
        except Exception as e:
            raise PipelineError(f"Content analysis failed: {e}", PipelineStage.ANALYSIS)
    
    async def _extract_clips(self, video_path: str, recommendations: List[ClipRecommendation]):
        """Extract clips based on recommendations."""
        try:
            # Run clip extraction in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                extraction_result = await loop.run_in_executor(
                    executor,
                    self.clip_extractor.extract_clips_from_recommendations,
                    video_path,
                    recommendations
                )
            
            extracted_clips = []
            for clip_result in extraction_result.clips:
                if clip_result.success:
                    extracted_clips.append({
                        "path": clip_result.clip_path,
                        "start_time": clip_result.start_time,
                        "end_time": clip_result.end_time,
                        "duration": clip_result.duration_seconds,
                        "file_size_mb": clip_result.file_size_mb
                    })
            
            logger.info(f"Clip extraction completed: {len(extracted_clips)} clips extracted")
            return extracted_clips
        except Exception as e:
            raise PipelineError(f"Clip extraction failed: {e}", PipelineStage.EXTRACTION)
    
    async def _optimize_clips(self, extracted_clips: List[Dict], max_duration: int):
        """Optimize clips for Twitter."""
        try:
            optimized_clips = []
            
            for clip_info in extracted_clips:
                clip_path = clip_info["path"]
                
                # Run optimization in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    optimization_result = await loop.run_in_executor(
                        executor,
                        self.twitter_optimizer.optimize_for_twitter,
                        clip_path,
                        max_duration
                    )
                
                if optimization_result.success:
                    optimized_clips.append({
                        "original_path": clip_path,
                        "optimized_path": optimization_result.output_path,
                        "original_size_mb": optimization_result.original_size_mb,
                        "optimized_size_mb": optimization_result.optimized_size_mb,
                        "compression_ratio": optimization_result.compression_ratio,
                        "quality_score": optimization_result.quality_score
                    })
            
            logger.info(f"Clip optimization completed: {len(optimized_clips)} clips optimized")
            return optimized_clips
        except Exception as e:
            raise PipelineError(f"Clip optimization failed: {e}", PipelineStage.OPTIMIZATION)
    
    async def _cleanup_temp_files(self, keep_optimized: bool = True):
        """Clean up temporary files."""
        try:
            # Keep optimized clips in output directory
            if keep_optimized:
                # Only clean up temp directory, not output directory
                temp_downloads = self.temp_dir / "downloads"
                temp_thumbnails = self.temp_dir / "thumbnails"
                temp_clips = self.temp_dir / "clips"
                
                for temp_path in [temp_downloads, temp_thumbnails, temp_clips]:
                    if temp_path.exists():
                        shutil.rmtree(temp_path, ignore_errors=True)
            else:
                # Clean up everything including output
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            logger.info("Temporary files cleaned up successfully")
        except Exception as e:
            logger.warning(f"Cleanup failed (non-critical): {e}")


# Convenience functions for direct usage
async def process_youtube_video(
    youtube_url: str,
    output_dir: Optional[Path] = None,
    num_clips: int = 2,
    max_clip_duration: int = 140,
    llm_provider: str = "gemini",
    progress_callback: Optional[Callable[[PipelineProgress], None]] = None
) -> PipelineResult:
    """
    Process a YouTube video through the complete pipeline (convenience function).
    
    Args:
        youtube_url: YouTube video URL
        output_dir: Output directory for final clips
        num_clips: Number of clips to extract
        max_clip_duration: Maximum clip duration in seconds
        llm_provider: LLM provider ("gemini" or "groq")
        progress_callback: Optional progress callback function
        
    Returns:
        PipelineResult with all outputs and metadata
    """
    pipeline = TwitterClipPipeline(
        output_dir=output_dir,
        llm_provider=llm_provider
    )
    
    if progress_callback:
        pipeline.set_progress_callback(progress_callback)
    
    return await pipeline.process_video(
        youtube_url=youtube_url,
        num_clips=num_clips,
        max_clip_duration=max_clip_duration
    )


def create_pipeline_with_config(config_overrides: Optional[Dict[str, Any]] = None) -> TwitterClipPipeline:
    """
    Create a pipeline instance with configuration overrides.
    
    Args:
        config_overrides: Dictionary of configuration overrides
        
    Returns:
        Configured TwitterClipPipeline instance
    """
    pipeline_config = {
        "output_dir": Path("outputs"),
        "max_retries": 3,
        "retry_delay": 1.0,
        "llm_provider": "gemini",
        "whisper_model": "base",
        "cleanup_temp_files": True
    }
    
    if config_overrides:
        pipeline_config.update(config_overrides)
    
    return TwitterClipPipeline(**pipeline_config)