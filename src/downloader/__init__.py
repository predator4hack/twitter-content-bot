"""
YouTube video downloading and metadata extraction.

This package provides comprehensive functionality for downloading YouTube videos,
extracting thumbnails, and handling video metadata with proper error handling
and validation.
"""

from .youtube_downloader import (
    YouTubeDownloader,
    YouTubeURLValidator,
    YouTubeDownloadError,
    download_youtube_video,
)
from .thumbnail_extractor import (
    ThumbnailExtractor,
    extract_youtube_thumbnail,
)

__all__ = [
    'YouTubeDownloader',
    'YouTubeURLValidator', 
    'YouTubeDownloadError',
    'download_youtube_video',
    'ThumbnailExtractor',
    'extract_youtube_thumbnail',
]
