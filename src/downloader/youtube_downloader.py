"""
YouTube Video Downloader Module

This module provides functionality for downloading YouTube videos using yt-dlp,
with proper URL validation, metadata extraction, and error handling.
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import yt_dlp
from src.core.config import config
from src.core.logger import get_logger, LoggerMixin


class YouTubeURLValidator:
    """Validates YouTube URLs and extracts video IDs."""
    
    # YouTube URL patterns
    YOUTUBE_PATTERNS = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
    ]
    
    @classmethod
    def is_valid_youtube_url(cls, url: str) -> bool:
        """
        Check if the given URL is a valid YouTube URL.
        
        Args:
            url: URL string to validate
            
        Returns:
            True if valid YouTube URL, False otherwise
        """
        if not url or not isinstance(url, str):
            return False
            
        for pattern in cls.YOUTUBE_PATTERNS:
            if re.match(pattern, url.strip()):
                return True
        return False
    
    @classmethod
    def extract_video_id(cls, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID if found, None otherwise
        """
        if not cls.is_valid_youtube_url(url):
            return None
            
        for pattern in cls.YOUTUBE_PATTERNS:
            match = re.match(pattern, url.strip())
            if match:
                return match.group(1)
        return None
    
    @classmethod
    def normalize_url(cls, url: str) -> Optional[str]:
        """
        Normalize YouTube URL to standard format.
        
        Args:
            url: YouTube URL in any format
            
        Returns:
            Normalized URL or None if invalid
        """
        video_id = cls.extract_video_id(url)
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        return None


class YouTubeDownloader(LoggerMixin):
    """
    YouTube video downloader using yt-dlp with comprehensive features.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the YouTube downloader.
        
        Args:
            output_dir: Directory to save downloaded videos
        """
        self.output_dir = output_dir or config.TEMP_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # yt-dlp options
        self.ydl_opts = {
            'format': self._get_format_selector(),
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'writesubtitles': False,
            'writeautomaticsub': False,
            'writethumbnail': config.EXTRACT_THUMBNAILS,
            'no_warnings': False,
            'extractflat': False,
            'ignoreerrors': False,
        }
        
        self.logger.info(f"YouTube downloader initialized with output dir: {self.output_dir}")
    
    def _get_format_selector(self) -> str:
        """Get format selector based on configuration."""
        quality = config.YTDL_QUALITY
        format_type = config.YTDL_FORMAT
        
        # Map quality settings to yt-dlp format selectors
        quality_map = {
            '480p': 'best[height<=480]',
            '720p': 'best[height<=720]',
            '1080p': 'best[height<=1080]',
            'best': 'best',
        }
        
        base_format = quality_map.get(quality, 'best[height<=720]')
        
        if format_type == 'mp4':
            return f'{base_format}[ext=mp4]/best[ext=mp4]/{base_format}'
        else:
            return base_format
    
    def get_video_info(self, url: str) -> Dict:
        """
        Extract video information without downloading.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary containing video metadata
            
        Raises:
            ValueError: If URL is invalid or video is not accessible
        """
        if not YouTubeURLValidator.is_valid_youtube_url(url):
            raise ValueError(f"Invalid YouTube URL: {url}")
        
        normalized_url = YouTubeURLValidator.normalize_url(url)
        self.logger.info(f"Extracting info for: {normalized_url}")
        
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(normalized_url, download=False)
                
                return {
                    'id': info.get('id'),
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader'),
                    'upload_date': info.get('upload_date'),
                    'view_count': info.get('view_count'),
                    'like_count': info.get('like_count'),
                    'thumbnail': info.get('thumbnail'),
                    'webpage_url': info.get('webpage_url'),
                    'formats': len(info.get('formats', [])),
                    'availability': info.get('availability'),
                }
                
        except yt_dlp.DownloadError as e:
            self.logger.error(f"Failed to extract video info: {e}")
            raise ValueError(f"Could not access video: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error extracting video info: {e}")
            raise ValueError(f"Error processing video: {e}")
    
    def download_video(self, url: str, filename: Optional[str] = None) -> Tuple[Path, Dict]:
        """
        Download YouTube video and return file path and metadata.
        
        Args:
            url: YouTube video URL
            filename: Optional custom filename (without extension)
            
        Returns:
            Tuple of (video_file_path, metadata_dict)
            
        Raises:
            ValueError: If URL is invalid or download fails
        """
        if not YouTubeURLValidator.is_valid_youtube_url(url):
            raise ValueError(f"Invalid YouTube URL: {url}")
        
        normalized_url = YouTubeURLValidator.normalize_url(url)
        self.logger.info(f"Starting download for: {normalized_url}")
        
        # Get video info first
        info = self.get_video_info(url)
        
        # Check duration limit
        if info['duration'] and info['duration'] > config.MAX_VIDEO_DURATION:
            raise ValueError(
                f"Video duration ({info['duration']}s) exceeds maximum allowed "
                f"({config.MAX_VIDEO_DURATION}s)"
            )
        
        # Set custom filename if provided
        ydl_opts = self.ydl_opts.copy()
        if filename:
            # Sanitize filename
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            ydl_opts['outtmpl'] = str(self.output_dir / f'{safe_filename}.%(ext)s')
        
        start_time = time.time()
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Download the video
                ydl.download([normalized_url])
                
                # Find the downloaded file
                video_file = self._find_downloaded_file(info['title'], filename)
                
                download_time = time.time() - start_time
                self.logger.info(f"Download completed in {download_time:.2f}s: {video_file}")
                
                # Update metadata with download info
                info.update({
                    'download_time': download_time,
                    'file_path': str(video_file),
                    'file_size': video_file.stat().st_size if video_file.exists() else 0,
                })
                
                return video_file, info
                
        except yt_dlp.DownloadError as e:
            self.logger.error(f"Download failed: {e}")
            raise ValueError(f"Download failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected download error: {e}")
            raise ValueError(f"Download error: {e}")
    
    def _find_downloaded_file(self, title: str, custom_filename: Optional[str] = None) -> Path:
        """
        Find the downloaded video file.
        
        Args:
            title: Original video title
            custom_filename: Custom filename if used
            
        Returns:
            Path to the downloaded file
            
        Raises:
            FileNotFoundError: If file cannot be found
        """
        # Try different possible filenames
        possible_names = []
        
        if custom_filename:
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', custom_filename)
            possible_names.append(safe_filename)
        
        if title:
            # yt-dlp sanitizes filenames, so we need to replicate that
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
            possible_names.append(safe_title)
        
        # Common video extensions
        extensions = ['mp4', 'webm', 'mkv', 'avi', 'mov']
        
        for name in possible_names:
            for ext in extensions:
                file_path = self.output_dir / f"{name}.{ext}"
                if file_path.exists():
                    return file_path
        
        # Fallback: find the most recently created video file
        video_files = []
        for ext in extensions:
            video_files.extend(self.output_dir.glob(f"*.{ext}"))
        
        if video_files:
            # Return the most recently modified file
            latest_file = max(video_files, key=lambda x: x.stat().st_mtime)
            return latest_file
        
        raise FileNotFoundError("Could not locate downloaded video file")
    
    def validate_and_prepare_url(self, url: str) -> str:
        """
        Validate URL and return normalized version.
        
        Args:
            url: Input URL
            
        Returns:
            Normalized YouTube URL
            
        Raises:
            ValueError: If URL is invalid
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
        
        url = url.strip()
        
        if not YouTubeURLValidator.is_valid_youtube_url(url):
            raise ValueError(
                "Invalid YouTube URL. Please provide a valid YouTube video link."
            )
        
        normalized = YouTubeURLValidator.normalize_url(url)
        if not normalized:
            raise ValueError("Could not normalize YouTube URL")
        
        return normalized
    
    def cleanup_downloads(self, keep_recent: int = 5) -> List[Path]:
        """
        Clean up old downloaded files, keeping only the most recent ones.
        
        Args:
            keep_recent: Number of recent files to keep
            
        Returns:
            List of deleted file paths
        """
        if not config.CLEANUP_TEMP_FILES:
            return []
        
        video_extensions = ['mp4', 'webm', 'mkv', 'avi', 'mov']
        all_files = []
        
        for ext in video_extensions:
            all_files.extend(self.output_dir.glob(f"*.{ext}"))
            all_files.extend(self.output_dir.glob(f"*.{ext}.*"))  # Include thumbnails
        
        # Sort by modification time (newest first)
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep the most recent files
        files_to_delete = all_files[keep_recent:]
        deleted_files = []
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                deleted_files.append(file_path)
                self.logger.debug(f"Deleted old file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Could not delete {file_path}: {e}")
        
        if deleted_files:
            self.logger.info(f"Cleaned up {len(deleted_files)} old files")
        
        return deleted_files


class YouTubeDownloadError(Exception):
    """Custom exception for YouTube download errors."""
    pass


# Convenience function for quick downloads
def download_youtube_video(
    url: str, 
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None
) -> Tuple[Path, Dict]:
    """
    Convenience function to download a YouTube video.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        filename: Optional custom filename
        
    Returns:
        Tuple of (video_file_path, metadata_dict)
    """
    downloader = YouTubeDownloader(output_dir)
    return downloader.download_video(url, filename)
