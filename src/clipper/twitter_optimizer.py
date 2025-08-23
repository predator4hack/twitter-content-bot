"""
Twitter video format optimization for extracted clips.

This module provides functionality to optimize video clips for Twitter's
specific format requirements, file size limits, and quality standards.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

import ffmpeg

from ..core.config import config

logger = logging.getLogger(__name__)


class TwitterAspectRatio(Enum):
    """Supported Twitter aspect ratios."""
    LANDSCAPE = "16:9"
    SQUARE = "1:1"
    PORTRAIT = "9:16"
    ORIGINAL = "original"


class TwitterResolution(Enum):
    """Supported Twitter resolutions."""
    HD_720P = "720p"
    FHD_1080P = "1080p"
    AUTO = "auto"


class VideoQuality(Enum):
    """Video quality presets for optimization."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TWITTER_OPTIMIZED = "twitter_optimized"


@dataclass
class TwitterSpecs:
    """Twitter video specifications and constraints."""
    
    max_file_size_mb: int = 512
    max_duration_seconds: int = 140  # 2:20
    video_codec: str = "h264"
    audio_codec: str = "aac"
    container_format: str = "mp4"
    max_bitrate_kbps: int = 25000  # 25 Mbps
    target_file_size_mb: int = 50  # Target for optimization
    
    # Resolution constraints
    max_width: int = 1920
    max_height: int = 1080
    min_width: int = 320
    min_height: int = 240
    
    # Quality settings
    crf_high: int = 18      # High quality
    crf_medium: int = 23    # Medium quality  
    crf_low: int = 28       # Low quality
    crf_twitter: int = 20   # Twitter optimized
    
    # Audio settings
    audio_bitrate_kbps: int = 128
    audio_sample_rate: int = 48000


@dataclass
class OptimizationResult:
    """Result of Twitter optimization process."""
    
    optimized_path: str
    original_path: str
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    quality_score: float
    twitter_compatible: bool
    success: bool
    error_message: Optional[str] = None
    optimization_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization to set defaults."""
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def size_reduction_percent(self) -> float:
        """Calculate percentage of size reduction."""
        if self.original_size_mb == 0:
            return 0.0
        return ((self.original_size_mb - self.optimized_size_mb) / self.original_size_mb) * 100


@dataclass
class BatchOptimizationResult:
    """Result of batch Twitter optimization."""
    
    results: List[OptimizationResult]
    total_time: float
    success_count: int
    failure_count: int
    average_compression_ratio: float
    average_quality_score: float
    total_size_reduction_mb: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = len(self.results)
        return (self.success_count / total * 100) if total > 0 else 0.0


class TwitterOptimizer:
    """
    Video optimizer for Twitter format compliance and size optimization.
    
    Features:
    - Twitter format compliance (MP4, H.264, AAC)
    - File size optimization (under 512MB, target ~50MB)
    - Resolution optimization (720p, 1080p)
    - Aspect ratio handling (16:9, 1:1, 9:16)
    - Quality vs. size optimization
    - Batch processing support
    """
    
    def __init__(
        self,
        specs: Optional[TwitterSpecs] = None,
        temp_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize Twitter optimizer.
        
        Args:
            specs: Twitter specifications (uses defaults if None)
            temp_dir: Directory for temporary files
            output_dir: Directory for optimized clips
        """
        self.specs = specs or TwitterSpecs()
        self.temp_dir = temp_dir or config.TEMP_DIR
        self.output_dir = output_dir or config.OUTPUT_DIR
        
        # Ensure directories exist
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"TwitterOptimizer initialized with target size: {self.specs.target_file_size_mb}MB")
    
    def optimize_for_twitter(
        self,
        input_path: str,
        output_filename: Optional[str] = None,
        aspect_ratio: TwitterAspectRatio = TwitterAspectRatio.ORIGINAL,
        resolution: TwitterResolution = TwitterResolution.AUTO,
        quality: VideoQuality = VideoQuality.TWITTER_OPTIMIZED
    ) -> OptimizationResult:
        """
        Optimize a video clip for Twitter.
        
        Args:
            input_path: Path to input video file
            output_filename: Custom output filename (auto-generated if None)
            aspect_ratio: Target aspect ratio for Twitter
            resolution: Target resolution
            quality: Quality preset for optimization
            
        Returns:
            OptimizationResult with optimization details
        """
        start_time = time.time()
        temp_path = None
        original_size_mb = 0.0
        
        try:
            # Validate input
            if not os.path.exists(input_path):
                return OptimizationResult(
                    optimized_path="",
                    original_path=input_path,
                    original_size_mb=0.0,
                    optimized_size_mb=0.0,
                    compression_ratio=0.0,
                    quality_score=0.0,
                    twitter_compatible=False,
                    success=False,
                    error_message=f"Input file not found: {input_path}"
                )
            
            # Get original file info
            original_info = self._get_video_info(input_path)
            original_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            
            # Generate output filename if not provided
            if output_filename is None:
                input_name = Path(input_path).stem
                timestamp = int(time.time())
                output_filename = f"{input_name}_twitter_optimized_{timestamp}.mp4"
            
            output_path = self.output_dir / output_filename
            
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                dir=self.temp_dir,
                delete=False
            ) as temp_file:
                temp_path = temp_file.name
            
            logger.info(f"Optimizing {input_path} for Twitter (target: {self.specs.target_file_size_mb}MB)")
            
            # Build optimization pipeline
            success = self._optimize_video(
                input_path,
                str(temp_path),
                original_info,
                aspect_ratio,
                resolution,
                quality
            )
            
            if not success:
                return OptimizationResult(
                    optimized_path="",
                    original_path=input_path,
                    original_size_mb=original_size_mb,
                    optimized_size_mb=0.0,
                    compression_ratio=0.0,
                    quality_score=0.0,
                    twitter_compatible=False,
                    success=False,
                    error_message="Video optimization failed"
                )
            
            # Move to final location
            os.rename(str(temp_path), str(output_path))
            
            # Get optimized file info
            optimized_size_mb = os.path.getsize(str(output_path)) / (1024 * 1024)
            optimized_info = self._get_video_info(str(output_path))
            
            # Calculate metrics
            compression_ratio = original_size_mb / optimized_size_mb if optimized_size_mb > 0 else 0
            quality_score = self._calculate_quality_score(optimized_info, optimized_size_mb)
            twitter_compatible = self._check_twitter_compatibility(optimized_info, optimized_size_mb)
            
            optimization_time = time.time() - start_time
            
            logger.info(
                f"Optimization complete: {original_size_mb:.1f}MB → {optimized_size_mb:.1f}MB "
                f"({compression_ratio:.1f}x compression) in {optimization_time:.1f}s"
            )
            
            return OptimizationResult(
                optimized_path=str(output_path),
                original_path=input_path,
                original_size_mb=original_size_mb,
                optimized_size_mb=optimized_size_mb,
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                twitter_compatible=twitter_compatible,
                success=True,
                optimization_time=optimization_time,
                metadata={
                    "original_info": original_info,
                    "optimized_info": optimized_info,
                    "aspect_ratio": aspect_ratio.value,
                    "resolution": resolution.value,
                    "quality": quality.value
                }
            )
            
        except Exception as e:
            error_msg = f"Optimization error: {str(e)}"
            logger.error(error_msg)
            
            # Cleanup temp file if it exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            
            return OptimizationResult(
                optimized_path="",
                original_path=input_path,
                original_size_mb=original_size_mb,
                optimized_size_mb=0.0,
                compression_ratio=0.0,
                quality_score=0.0,
                twitter_compatible=False,
                success=False,
                error_message=error_msg,
                optimization_time=time.time() - start_time
            )
    
    def optimize_batch(
        self,
        input_paths: List[str],
        aspect_ratio: TwitterAspectRatio = TwitterAspectRatio.ORIGINAL,
        resolution: TwitterResolution = TwitterResolution.AUTO,
        quality: VideoQuality = VideoQuality.TWITTER_OPTIMIZED
    ) -> BatchOptimizationResult:
        """
        Optimize multiple videos for Twitter.
        
        Args:
            input_paths: List of input video file paths
            aspect_ratio: Target aspect ratio
            resolution: Target resolution  
            quality: Quality preset
            
        Returns:
            BatchOptimizationResult with all optimization results
        """
        start_time = time.time()
        
        logger.info(f"Starting batch optimization of {len(input_paths)} videos for Twitter")
        
        results = []
        for i, input_path in enumerate(input_paths):
            output_filename = f"twitter_clip_{i+1:02d}_{int(time.time())}.mp4"
            result = self.optimize_for_twitter(
                input_path,
                output_filename,
                aspect_ratio,
                resolution,
                quality
            )
            results.append(result)
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        failure_count = len(results) - success_count
        
        # Calculate averages for successful optimizations
        successful_results = [r for r in results if r.success]
        avg_compression = sum(r.compression_ratio for r in successful_results) / len(successful_results) if successful_results else 0
        avg_quality = sum(r.quality_score for r in successful_results) / len(successful_results) if successful_results else 0
        total_size_reduction = sum(r.original_size_mb - r.optimized_size_mb for r in successful_results)
        
        batch_result = BatchOptimizationResult(
            results=results,
            total_time=total_time,
            success_count=success_count,
            failure_count=failure_count,
            average_compression_ratio=avg_compression,
            average_quality_score=avg_quality,
            total_size_reduction_mb=total_size_reduction
        )
        
        logger.info(
            f"Batch optimization completed: {success_count}/{len(input_paths)} successful, "
            f"{total_size_reduction:.1f}MB saved, {total_time:.1f}s"
        )
        
        return batch_result
    
    def _optimize_video(
        self,
        input_path: str,
        output_path: str,
        original_info: Dict[str, Any],
        aspect_ratio: TwitterAspectRatio,
        resolution: TwitterResolution,
        quality: VideoQuality
    ) -> bool:
        """Perform the actual video optimization using ffmpeg."""
        try:
            # Start with input
            input_stream = ffmpeg.input(input_path)
            
            # Determine target resolution
            target_width, target_height = self._calculate_target_resolution(
                original_info,
                aspect_ratio,
                resolution
            )
            
            # Build video filter chain
            video_filters = []
            
            # Scale if needed
            if target_width != original_info.get('width', 0) or target_height != original_info.get('height', 0):
                video_filters.append(f"scale={target_width}:{target_height}")
            
            # Apply aspect ratio padding if needed
            if aspect_ratio != TwitterAspectRatio.ORIGINAL:
                video_filters.extend(self._get_aspect_ratio_filters(aspect_ratio, target_width, target_height))
            
            # Get quality settings
            crf, bitrate = self._get_quality_settings(quality, target_width, target_height)
            
            # Build output arguments
            output_args = {
                'vcodec': self.specs.video_codec,
                'acodec': self.specs.audio_codec,
                'crf': crf,
                'preset': 'medium',
                'profile:v': 'main',
                'level': '4.0',
                'pix_fmt': 'yuv420p',
                'movflags': '+faststart',  # Optimize for streaming
                'ab': f"{self.specs.audio_bitrate_kbps}k",
                'ar': self.specs.audio_sample_rate
            }
            
            # Add video filters if any
            if video_filters:
                output_args['vf'] = ','.join(video_filters)
            
            # Add bitrate constraint if needed
            if bitrate:
                output_args['maxrate'] = f"{bitrate}k"
                output_args['bufsize'] = f"{bitrate * 2}k"
            
            # Create output stream
            output_stream = ffmpeg.output(input_stream, str(output_path), **output_args)
            
            # Run ffmpeg
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            # Verify output exists and is valid
            if not os.path.exists(str(output_path)) or os.path.getsize(str(output_path)) == 0:
                logger.error("FFmpeg optimization produced no output")
                return False
            
            return True
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error during optimization: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during optimization: {e}")
            return False
    
    def _calculate_target_resolution(
        self,
        original_info: Dict[str, Any],
        aspect_ratio: TwitterAspectRatio,
        resolution: TwitterResolution
    ) -> Tuple[int, int]:
        """Calculate target resolution based on requirements."""
        original_width = original_info.get('width', 1920)
        original_height = original_info.get('height', 1080)
        
        # Handle resolution preset
        if resolution == TwitterResolution.HD_720P:
            base_height = 720
        elif resolution == TwitterResolution.FHD_1080P:
            base_height = 1080
        else:  # AUTO
            # Keep original if within limits, otherwise scale down
            base_height = min(original_height, self.specs.max_height)
        
        # Handle aspect ratio
        if aspect_ratio == TwitterAspectRatio.LANDSCAPE:  # 16:9
            target_width = int(base_height * 16 / 9)
            target_height = base_height
        elif aspect_ratio == TwitterAspectRatio.SQUARE:  # 1:1
            target_width = target_height = base_height
        elif aspect_ratio == TwitterAspectRatio.PORTRAIT:  # 9:16
            target_width = int(base_height * 9 / 16)
            target_height = base_height
        else:  # ORIGINAL
            # Maintain original aspect ratio
            original_aspect = original_width / original_height
            target_height = base_height
            target_width = int(target_height * original_aspect)
        
        # Ensure within Twitter limits
        target_width = max(self.specs.min_width, min(target_width, self.specs.max_width))
        target_height = max(self.specs.min_height, min(target_height, self.specs.max_height))
        
        # Ensure even dimensions (required for some codecs)
        target_width = target_width - (target_width % 2)
        target_height = target_height - (target_height % 2)
        
        return target_width, target_height
    
    def _get_aspect_ratio_filters(
        self,
        aspect_ratio: TwitterAspectRatio,
        target_width: int,
        target_height: int
    ) -> List[str]:
        """Get ffmpeg filters for aspect ratio adjustment."""
        filters = []
        
        if aspect_ratio == TwitterAspectRatio.SQUARE:
            # Add padding to make square
            filters.append(f"pad={max(target_width, target_height)}:{max(target_width, target_height)}:(ow-iw)/2:(oh-ih)/2:black")
        elif aspect_ratio == TwitterAspectRatio.PORTRAIT:
            # Ensure portrait orientation
            if target_width > target_height:
                filters.append(f"pad={target_height}:{target_width}:(ow-iw)/2:(oh-ih)/2:black")
        
        return filters
    
    def _get_quality_settings(
        self,
        quality: VideoQuality,
        width: int,
        height: int
    ) -> Tuple[int, Optional[int]]:
        """Get CRF and bitrate settings for quality preset."""
        pixel_count = width * height
        
        if quality == VideoQuality.HIGH:
            crf = self.specs.crf_high
            bitrate = min(int(pixel_count * 0.15 / 1000), self.specs.max_bitrate_kbps)
        elif quality == VideoQuality.MEDIUM:
            crf = self.specs.crf_medium
            bitrate = min(int(pixel_count * 0.10 / 1000), self.specs.max_bitrate_kbps)
        elif quality == VideoQuality.LOW:
            crf = self.specs.crf_low
            bitrate = min(int(pixel_count * 0.05 / 1000), self.specs.max_bitrate_kbps)
        else:  # TWITTER_OPTIMIZED
            crf = self.specs.crf_twitter
            # Calculate bitrate for target file size
            bitrate = self._calculate_target_bitrate(width, height)
        
        return crf, bitrate
    
    def _calculate_target_bitrate(self, width: int, height: int) -> int:
        """Calculate target bitrate for Twitter optimization."""
        # Estimate bitrate needed for target file size
        # This is a simplified calculation - actual file size depends on content complexity
        
        pixel_count = width * height
        base_bitrate = int(pixel_count * 0.08 / 1000)  # Base bitrate per pixel
        
        # Cap at Twitter's maximum and our target
        max_bitrate_for_size = int((self.specs.target_file_size_mb * 8 * 1024) / 140)  # For 140s max duration
        
        return min(base_bitrate, max_bitrate_for_size, self.specs.max_bitrate_kbps)
    
    def _calculate_quality_score(
        self,
        video_info: Dict[str, Any],
        file_size_mb: float
    ) -> float:
        """Calculate quality score (0-100) based on video properties."""
        score = 100.0
        
        # Penalize if over target file size
        if file_size_mb > self.specs.target_file_size_mb:
            score -= min(30, (file_size_mb - self.specs.target_file_size_mb) * 2)
        
        # Reward good resolution
        width = video_info.get('width', 0)
        height = video_info.get('height', 0)
        if height >= 1080:
            score += 10
        elif height >= 720:
            score += 5
        
        # Penalize very low bitrate
        bitrate = video_info.get('bit_rate', 0)
        if bitrate < 500000:  # Less than 500kbps
            score -= 20
        
        return max(0, min(100, score))
    
    def _check_twitter_compatibility(
        self,
        video_info: Dict[str, Any],
        file_size_mb: float
    ) -> bool:
        """Check if video meets Twitter's requirements."""
        # File size check
        if file_size_mb > self.specs.max_file_size_mb:
            return False
        
        # Format check
        format_name = video_info.get('format_name', '')
        if 'mp4' not in format_name:
            return False
        
        # Codec check
        video_codec = video_info.get('video_codec', '')
        audio_codec = video_info.get('audio_codec', '')
        
        if video_codec != 'h264' or audio_codec != 'aac':
            return False
        
        # Resolution check
        width = video_info.get('width', 0)
        height = video_info.get('height', 0)
        
        if (width < self.specs.min_width or width > self.specs.max_width or
            height < self.specs.min_height or height > self.specs.max_height):
            return False
        
        # Duration check
        duration = video_info.get('duration', 0)
        if duration > self.specs.max_duration_seconds:
            return False
        
        return True
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using ffprobe."""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            info = {
                'duration': float(probe['format']['duration']),
                'size_mb': float(probe['format']['size']) / (1024 * 1024),
                'format_name': probe['format']['format_name'],
                'bit_rate': int(probe['format'].get('bit_rate', 0))
            }
            
            if video_stream:
                info.update({
                    'width': int(video_stream['width']),
                    'height': int(video_stream['height']),
                    'fps': eval(video_stream['r_frame_rate']) if '/' in video_stream.get('r_frame_rate', '0/1') else 0,
                    'video_codec': video_stream['codec_name']
                })
            
            if audio_stream:
                info.update({
                    'audio_codec': audio_stream['codec_name'],
                    'audio_sample_rate': int(audio_stream.get('sample_rate', 0))
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info for {video_path}: {e}")
            return {}


# Convenience functions

def optimize_single_clip(
    input_path: str,
    output_path: Optional[str] = None,
    aspect_ratio: TwitterAspectRatio = TwitterAspectRatio.ORIGINAL,
    resolution: TwitterResolution = TwitterResolution.HD_720P,
    quality: VideoQuality = VideoQuality.HIGH
) -> OptimizationResult:
    """
    Optimize a single clip for Twitter - convenience function.
    
    Args:
        input_path: Path to input video
        output_path: Output file path (auto-generated if None)
        aspect_ratio: Target aspect ratio
        resolution: Target resolution (default: 720p)
        quality: Quality preset (default: HIGH)
        
    Returns:
        OptimizationResult
    """
    optimizer = TwitterOptimizer()
    
    output_filename = Path(output_path).name if output_path else None
    
    return optimizer.optimize_for_twitter(
        input_path,
        output_filename,
        aspect_ratio,
        resolution,
        quality
    )


def optimize_extracted_clips(
    clip_paths: List[str],
    aspect_ratio: TwitterAspectRatio = TwitterAspectRatio.ORIGINAL,
    quality: VideoQuality = VideoQuality.TWITTER_OPTIMIZED
) -> BatchOptimizationResult:
    """
    Optimize extracted clips for Twitter - convenience function.
    
    Args:
        clip_paths: List of extracted clip file paths
        aspect_ratio: Target aspect ratio
        quality: Quality preset
        
    Returns:
        BatchOptimizationResult
    """
    optimizer = TwitterOptimizer()
    return optimizer.optimize_batch(clip_paths, aspect_ratio, TwitterResolution.AUTO, quality)
