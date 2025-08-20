"""
Video clip extraction using ffmpeg for precise video trimming.

This module provides functionality to extract video clips from source videos
with frame-accurate timing, quality preservation, and efficient processing.
"""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import ffmpeg

from ..core.config import config
from ..analyzer.llm_analyzer import ClipRecommendation

logger = logging.getLogger(__name__)


@dataclass
class ClipExtractionResult:
    """Result of clip extraction operation."""
    
    clip_path: str
    start_time: str
    end_time: str
    duration_seconds: float
    file_size_mb: float
    success: bool
    error_message: Optional[str] = None
    extraction_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization to set defaults."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchExtractionResult:
    """Result of batch clip extraction operation."""
    
    source_video: str
    results: List[ClipExtractionResult]
    total_time: float
    success_count: int
    failure_count: int
    total_size_mb: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = len(self.results)
        return (self.success_count / total * 100) if total > 0 else 0.0


class ClipExtractor:
    """
    Video clip extractor using ffmpeg for precise video trimming.
    
    Features:
    - Frame-accurate cutting
    - Quality preservation
    - Parallel processing
    - Temporary file management
    - Multiple format support
    """
    
    def __init__(
        self,
        temp_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        cleanup_temp: bool = True,
        max_concurrent: int = 3
    ):
        """
        Initialize clip extractor.
        
        Args:
            temp_dir: Directory for temporary files
            output_dir: Directory for output clips
            cleanup_temp: Whether to cleanup temporary files
            max_concurrent: Maximum concurrent extractions
        """
        self.temp_dir = temp_dir or config.TEMP_DIR
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.cleanup_temp = cleanup_temp
        self.max_concurrent = max_concurrent
        
        # Ensure directories exist
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"ClipExtractor initialized with temp_dir={self.temp_dir}, output_dir={self.output_dir}")
    
    def extract_clip(
        self,
        source_video: str,
        start_time: str,
        end_time: str,
        output_filename: Optional[str] = None,
        quality_preset: str = "medium"
    ) -> ClipExtractionResult:
        """
        Extract a single clip from source video.
        
        Args:
            source_video: Path to source video file
            start_time: Start time in MM:SS or HH:MM:SS format
            end_time: End time in MM:SS or HH:MM:SS format
            output_filename: Custom output filename (auto-generated if None)
            quality_preset: FFmpeg quality preset (ultrafast, fast, medium, slow, veryslow)
            
        Returns:
            ClipExtractionResult with extraction details
        """
        start_extraction = time.time()
        temp_path = None
        start_seconds = 0.0
        end_seconds = 0.0
        
        try:
            # Validate inputs
            if not os.path.exists(source_video):
                return ClipExtractionResult(
                    clip_path="",
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=0.0,
                    file_size_mb=0.0,
                    success=False,
                    error_message=f"Source video not found: {source_video}"
                )
            
            # Convert time strings to seconds for validation
            start_seconds = self._time_to_seconds(start_time)
            end_seconds = self._time_to_seconds(end_time)
            duration = end_seconds - start_seconds
            
            if duration <= 0:
                return ClipExtractionResult(
                    clip_path="",
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=0.0,
                    file_size_mb=0.0,
                    success=False,
                    error_message=f"Invalid time range: start={start_time}, end={end_time}"
                )
            
            # Generate output filename if not provided
            if output_filename is None:
                source_name = Path(source_video).stem
                timestamp = int(time.time())
                output_filename = f"{source_name}_clip_{start_time.replace(':', '')}-{end_time.replace(':', '')}_{timestamp}.mp4"
            
            output_path = self.output_dir / output_filename
            
            # Use temporary file during processing to avoid partial files
            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                dir=self.temp_dir,
                delete=False
            ) as temp_file:
                temp_path = temp_file.name
            
            logger.info(f"Extracting clip: {start_time}-{end_time} from {source_video}")
            
            # Build ffmpeg command for precise extraction
            input_stream = ffmpeg.input(source_video, ss=start_seconds)
            output_stream = ffmpeg.output(
                input_stream,
                temp_path,
                t=duration,  # Duration instead of end time for precision
                c="copy",  # Copy streams without re-encoding when possible
                avoid_negative_ts="make_zero",  # Handle timestamp issues
                preset=quality_preset
            )
            
            # Run ffmpeg extraction
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            # Verify extraction was successful
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                return ClipExtractionResult(
                    clip_path="",
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    file_size_mb=0.0,
                    success=False,
                    error_message="FFmpeg extraction failed - no output file generated"
                )
            
            # Move to final location
            os.rename(temp_path, output_path)
            
            # Get file size
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            extraction_time = time.time() - start_extraction
            
            # Get metadata from extracted clip
            metadata = self._get_video_metadata(str(output_path))
            
            logger.info(f"Successfully extracted clip: {output_path} ({file_size_mb:.1f}MB, {extraction_time:.1f}s)")
            
            return ClipExtractionResult(
                clip_path=str(output_path),
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                file_size_mb=file_size_mb,
                success=True,
                extraction_time=extraction_time,
                metadata=metadata
            )
            
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg error during extraction: {str(e)}"
            logger.error(error_msg)
            
            # Cleanup temp file if it exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            
            return ClipExtractionResult(
                clip_path="",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=end_seconds - start_seconds,
                file_size_mb=0.0,
                success=False,
                error_message=error_msg,
                extraction_time=time.time() - start_extraction
            )
            
        except Exception as e:
            error_msg = f"Unexpected error during extraction: {str(e)}"
            logger.error(error_msg)
            
            return ClipExtractionResult(
                clip_path="",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=0.0,
                file_size_mb=0.0,
                success=False,
                error_message=error_msg,
                extraction_time=time.time() - start_extraction
            )
    
    def extract_clips_from_recommendations(
        self,
        source_video: str,
        recommendations: List[ClipRecommendation],
        parallel: bool = True
    ) -> BatchExtractionResult:
        """
        Extract multiple clips from recommendations.
        
        Args:
            source_video: Path to source video file
            recommendations: List of clip recommendations
            parallel: Whether to process clips in parallel
            
        Returns:
            BatchExtractionResult with all extraction results
        """
        start_time = time.time()
        
        logger.info(f"Starting batch extraction of {len(recommendations)} clips from {source_video}")
        
        if parallel and len(recommendations) > 1:
            results = self._extract_clips_parallel(source_video, recommendations)
        else:
            results = self._extract_clips_sequential(source_video, recommendations)
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        failure_count = len(results) - success_count
        total_size_mb = sum(r.file_size_mb for r in results if r.success)
        
        batch_result = BatchExtractionResult(
            source_video=source_video,
            results=results,
            total_time=total_time,
            success_count=success_count,
            failure_count=failure_count,
            total_size_mb=total_size_mb
        )
        
        logger.info(
            f"Batch extraction completed: {success_count}/{len(recommendations)} successful, "
            f"{total_size_mb:.1f}MB total, {total_time:.1f}s"
        )
        
        return batch_result
    
    def _extract_clips_parallel(
        self,
        source_video: str,
        recommendations: List[ClipRecommendation]
    ) -> List[ClipExtractionResult]:
        """Extract clips in parallel using ThreadPoolExecutor."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all extraction tasks
            future_to_rec = {
                executor.submit(
                    self.extract_clip,
                    source_video,
                    rec.start_time,
                    rec.end_time,
                    f"clip_{i+1:02d}_{rec.start_time.replace(':', '')}-{rec.end_time.replace(':', '')}.mp4"
                ): rec for i, rec in enumerate(recommendations)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_rec):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    rec = future_to_rec[future]
                    logger.error(f"Parallel extraction failed for {rec.start_time}-{rec.end_time}: {e}")
                    results.append(ClipExtractionResult(
                        clip_path="",
                        start_time=rec.start_time,
                        end_time=rec.end_time,
                        duration_seconds=rec.duration_seconds,
                        file_size_mb=0.0,
                        success=False,
                        error_message=f"Parallel extraction error: {str(e)}"
                    ))
        
        # Sort results by start time to maintain order
        results.sort(key=lambda x: self._time_to_seconds(x.start_time))
        return results
    
    def _extract_clips_sequential(
        self,
        source_video: str,
        recommendations: List[ClipRecommendation]
    ) -> List[ClipExtractionResult]:
        """Extract clips sequentially."""
        results = []
        
        for i, rec in enumerate(recommendations):
            output_filename = f"clip_{i+1:02d}_{rec.start_time.replace(':', '')}-{rec.end_time.replace(':', '')}.mp4"
            result = self.extract_clip(source_video, rec.start_time, rec.end_time, output_filename)
            results.append(result)
        
        return results
    
    def cleanup_temp_files(self) -> int:
        """
        Clean up temporary files.
        
        Returns:
            Number of files cleaned up
        """
        if not self.cleanup_temp:
            return 0
        
        cleanup_count = 0
        try:
            for temp_file in self.temp_dir.glob("*.mp4"):
                try:
                    temp_file.unlink()
                    cleanup_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} temporary files")
        
        return cleanup_count
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video information using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
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
                    'fps': eval(video_stream['r_frame_rate']),
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
    
    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Get basic metadata for extracted clip."""
        try:
            info = self.get_video_info(video_path)
            return {
                'duration': info.get('duration', 0.0),
                'width': info.get('width', 0),
                'height': info.get('height', 0),
                'fps': info.get('fps', 0.0),
                'video_codec': info.get('video_codec', ''),
                'audio_codec': info.get('audio_codec', '')
            }
        except Exception:
            return {}
    
    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert time string (MM:SS or HH:MM:SS) to seconds."""
        parts = time_str.split(":")
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            raise ValueError(f"Invalid time format: {time_str}. Expected MM:SS or HH:MM:SS")
    
    @staticmethod
    def _seconds_to_time(seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


# Convenience function for simple clip extraction
def extract_single_clip(
    source_video: str,
    start_time: str,
    end_time: str,
    output_path: Optional[str] = None
) -> ClipExtractionResult:
    """
    Extract a single clip - convenience function.
    
    Args:
        source_video: Path to source video
        start_time: Start time in MM:SS format
        end_time: End time in MM:SS format
        output_path: Output file path (auto-generated if None)
        
    Returns:
        ClipExtractionResult
    """
    extractor = ClipExtractor()
    
    if output_path:
        output_filename = Path(output_path).name
    else:
        output_filename = None
    
    return extractor.extract_clip(source_video, start_time, end_time, output_filename)


# Convenience function for batch extraction
def extract_clips_from_analysis(
    source_video: str,
    recommendations: List[ClipRecommendation],
    parallel: bool = True
) -> BatchExtractionResult:
    """
    Extract clips from LLM analysis recommendations - convenience function.
    
    Args:
        source_video: Path to source video
        recommendations: List of clip recommendations from LLM analysis
        parallel: Whether to process in parallel
        
    Returns:
        BatchExtractionResult
    """
    extractor = ClipExtractor()
    return extractor.extract_clips_from_recommendations(source_video, recommendations, parallel)
