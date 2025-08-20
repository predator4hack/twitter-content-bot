"""Video clip extraction and Twitter optimization."""

from .clip_extractor import (
    ClipExtractor,
    ClipExtractionResult,
    BatchExtractionResult,
    extract_single_clip,
    extract_clips_from_analysis
)

from .twitter_optimizer import (
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

__all__ = [
    "ClipExtractor",
    "ClipExtractionResult", 
    "BatchExtractionResult",
    "extract_single_clip",
    "extract_clips_from_analysis",
    "TwitterOptimizer",
    "TwitterSpecs",
    "TwitterAspectRatio",
    "TwitterResolution", 
    "VideoQuality",
    "OptimizationResult",
    "BatchOptimizationResult",
    "optimize_single_clip",
    "optimize_extracted_clips"
]
