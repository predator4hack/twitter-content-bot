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
from src.core.logger import get_logger, LoggerMixin


def _get_config():
    """Lazy import of config to avoid circular imports."""
    from src.core.config import config
    return config


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
        self.output_dir = output_dir or _get_config().TEMP_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # yt-dlp options with minimal configuration to avoid bot detection
        self.ydl_opts = {
            'format': self._get_format_selector(),
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'writesubtitles': False,
            'writeautomaticsub': False,
            'writethumbnail': _get_config().EXTRACT_THUMBNAILS,
            'no_warnings': False,
            'extractflat': False,
            'ignoreerrors': False,
            'retries': 3,
            'fragment_retries': 3,
            'extractor_retries': 3,
            'nocheckcertificate': True,
        }
        
        self.logger.info(f"YouTube downloader initialized with output dir: {self.output_dir}")
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent for bot detection avoidance."""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Android 13; Mobile; rv:109.0) Gecko/118.0 Firefox/118.0'
        ]
        import random
        return random.choice(user_agents)
    
    def _get_format_selector(self) -> str:
        """Get format selector based on configuration with fallbacks."""
        config = _get_config()
        quality = config.YTDL_QUALITY
        format_type = config.YTDL_FORMAT
        
        # More flexible format selectors with fallbacks
        if quality == '480p':
            # Try 480p first, then fallback to lower qualities
            formats = [
                'best[height<=480][ext=mp4]',
                'best[height<=480]',
                'worst[height>=360][ext=mp4]',
                'worst[height>=360]',
                'best[ext=mp4]',
                'best'
            ]
        elif quality == '720p':
            formats = [
                'best[height<=720][ext=mp4]',
                'best[height<=720]',
                'best[height<=480][ext=mp4]',
                'best[height<=480]',
                'best[ext=mp4]',
                'best'
            ]
        elif quality == '1080p':
            formats = [
                'best[height<=1080][ext=mp4]',
                'best[height<=1080]',
                'best[height<=720][ext=mp4]',
                'best[height<=720]',
                'best[ext=mp4]',
                'best'
            ]
        else:  # 'best' or fallback
            formats = [
                'best[ext=mp4]',
                'best[height<=720][ext=mp4]',
                'best[height<=720]',
                'best'
            ]
        
        # Join with fallback operator
        return '/'.join(formats)
    
    def get_video_info(self, url: str) -> Dict:
        """
        Extract video information with enhanced bot detection avoidance.
        
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
        if not normalized_url:
            raise ValueError(f"Could not normalize URL: {url}")
        
        self.logger.info(f"Extracting info for: {normalized_url}")
        
        # Try multiple extraction strategies
        strategies = [
            self._try_standard_extraction,
            self._try_mobile_extraction,
            self._try_android_extraction,
            self._try_minimal_extraction
        ]
        
        last_error = None
        for i, strategy in enumerate(strategies):
            try:
                if i > 0:
                    delay = min(2 ** i, 10)  # Exponential backoff, max 10s
                    self.logger.info(f"Waiting {delay}s before retry {i+1}...")
                    time.sleep(delay)
                
                self.logger.info(f"Trying extraction strategy {i+1}/{len(strategies)}...")
                result = strategy(normalized_url)
                self.logger.info(f"✅ Extraction successful with strategy {i+1}")
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Strategy {i+1} failed: {str(e)[:100]}...")
                continue
        
        # All strategies failed - provide helpful error message
        error_msg = str(last_error).lower() if last_error else ""
        if any(phrase in error_msg for phrase in ["sign in", "bot", "captcha", "cookies"]):
            helpful_msg = (
                f"Could not access video due to bot detection. This commonly happens in cloud environments. "
                f"Original error: {str(last_error)} "
                f"The proxy and multiple strategies were attempted. "
                f"To fix this, you may need to: "
                f"1) Use a different video URL, "
                f"2) Run from a different IP/location, "
                f"3) Use proxy settings, or "
                f"4) Try again later as YouTube's bot detection is temporary."
            )
            raise ValueError(helpful_msg)
        else:
            raise ValueError(f"All extraction strategies failed. Last error: {last_error}")
    
    def _try_standard_extraction(self, url: str) -> Dict:
        """Standard extraction with proxy."""
        opts = self._get_enhanced_ydl_opts()
        return self._extract_with_opts(url, opts)
    
    def _try_mobile_extraction(self, url: str) -> Dict:
        """Mobile-specific extraction."""
        opts = self._get_enhanced_ydl_opts()
        opts['http_headers']['User-Agent'] = 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15'
        opts['extractor_args']['youtube']['player_client'] = ['mweb']
        return self._extract_with_opts(url, opts)
    
    def _try_android_extraction(self, url: str) -> Dict:
        """Android app extraction."""
        opts = self._get_enhanced_ydl_opts()
        opts['http_headers']['User-Agent'] = 'com.google.android.youtube/19.09.36 (Linux; U; Android 11)'
        opts['extractor_args']['youtube']['player_client'] = ['android']
        return self._extract_with_opts(url, opts)
    
    def _try_minimal_extraction(self, url: str) -> Dict:
        """Minimal extraction as last resort."""
        opts = self._get_enhanced_ydl_opts()
        opts['extract_flat'] = False
        opts['extractor_args']['youtube']['skip'] = ['dash', 'hls', 'live_chat']
        return self._extract_with_opts(url, opts)
    
    def _get_enhanced_ydl_opts(self) -> Dict:
        """Get minimal yt-dlp options that avoid triggering bot detection."""
        config = _get_config()

        opts = {
            'quiet': True,
            'no_warnings': False,
            'retries': 3,
            'fragment_retries': 3,
            'extractor_retries': 3,
            'socket_timeout': 30,
            'nocheckcertificate': True,
        }

        return opts
    
    def _extract_with_opts(self, url: str, opts: Dict) -> Dict:
        """Extract video info with given options."""
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            if not info:
                raise ValueError("No video information could be extracted")
            
            return {
                'id': info.get('id', ''),
                'title': info.get('title', 'Unknown Title'),
                'duration': info.get('duration', 0),
                'description': info.get('description', ''),
                'uploader': info.get('uploader', 'Unknown'),
                'upload_date': info.get('upload_date', ''),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'thumbnail': info.get('thumbnail', ''),
                'webpage_url': info.get('webpage_url', url),
                'formats': len(info.get('formats', [])),
                'availability': info.get('availability', 'unknown')
            }
        
        try:
            # Enhanced options for info extraction with bot detection avoidance
            info_opts = {
                'quiet': False,
                'no_warnings': False,
                'extract_flat': False,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip,deflate',
                    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                    'Keep-Alive': '300',
                    'Connection': 'keep-alive',
                },
                'retries': 5,
                'fragment_retries': 5,
                'extractor_retries': 5,
                'retry_sleep_functions': {
                    'http': lambda n: min(4 ** n, 60),
                    'fragment': lambda n: min(2 ** n, 30),
                    'extractor': lambda n: min(2 ** n, 30),
                },
                'nocheckcertificate': True,
                'sleep_interval': 1,
                'max_sleep_interval': 5,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android', 'web'],
                        'player_skip': ['webpage'],
                    }
                },
                'proxy': os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY'),
            }
            
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(normalized_url, download=False)
                
                if not info:
                    raise ValueError("No video information could be extracted")
                
                return {
                    'id': info.get('id', ''),
                    'title': info.get('title', 'Unknown Title'),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'thumbnail': info.get('thumbnail', ''),
                    'webpage_url': info.get('webpage_url', normalized_url),
                    'formats': len(info.get('formats', [])),
                    'availability': info.get('availability', 'unknown'),
                }
                
        except yt_dlp.DownloadError as e:
            error_msg = str(e).lower()
            self.logger.warning(f"Primary extraction failed: {e}")
            
            # Check if it's a bot detection error
            if any(keyword in error_msg for keyword in ['sign in', 'bot', 'cookies', 'authentication']):
                self.logger.info("Bot detection detected, trying alternative extraction strategies...")
                
                # Try different extraction strategies
                strategies = ['basic', 'simple', 'minimal']
                for strategy in strategies:
                    try:
                        self.logger.info(f"Trying {strategy} extraction strategy...")
                        alt_opts = self._get_alternative_ydl_opts(strategy)
                        
                        with yt_dlp.YoutubeDL(alt_opts) as ydl:
                            info = ydl.extract_info(normalized_url, download=False)
                            
                            if info:
                                self.logger.info(f"Successfully extracted info using {strategy} strategy")
                                return {
                                    'id': info.get('id', ''),
                                    'title': info.get('title', 'Unknown Title'),
                                    'duration': info.get('duration', 0),
                                    'description': info.get('description', ''),
                                    'uploader': info.get('uploader', 'Unknown'),
                                    'upload_date': info.get('upload_date', ''),
                                    'view_count': info.get('view_count', 0),
                                    'like_count': info.get('like_count', 0),
                                    'thumbnail': info.get('thumbnail', ''),
                                    'webpage_url': info.get('webpage_url', normalized_url),
                                    'formats': len(info.get('formats', [])),
                                    'availability': info.get('availability', 'unknown'),
                                }
                    except Exception as alt_e:
                        self.logger.warning(f"{strategy} strategy failed: {alt_e}")
                        continue
                
                # If all strategies failed, provide helpful error message
                raise ValueError(
                    f"Could not access video due to bot detection. This commonly happens in cloud environments. "
                    f"Original error: {e}. "
                    f"To fix this, you may need to: "
                    f"1) Use a different video URL, "
                    f"2) Run from a different IP/location, "
                    f"3) Use proxy settings, or "
                    f"4) Try again later as YouTube's bot detection is temporary."
                )
            else:
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
        config = _get_config()
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
            error_msg = str(e).lower()
            self.logger.warning(f"Primary download failed: {e}")
            
            # Check if it's a bot detection error
            if any(keyword in error_msg for keyword in ['sign in', 'bot', 'cookies', 'authentication']):
                self.logger.info("Bot detection detected during download, trying alternative strategies...")
                
                # Try different extraction strategies for download
                strategies = ['basic', 'simple', 'minimal']
                for strategy in strategies:
                    try:
                        self.logger.info(f"Trying download with {strategy} strategy...")
                        alt_opts = self._get_alternative_ydl_opts(strategy)
                        
                        # Override with custom filename if provided
                        if filename:
                            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                            alt_opts['outtmpl'] = str(self.output_dir / f'{safe_filename}.%(ext)s')
                        
                        with yt_dlp.YoutubeDL(alt_opts) as ydl:
                            ydl.download([normalized_url])
                            
                            # Find the downloaded file
                            video_file = self._find_downloaded_file(info['title'], filename)
                            
                            download_time = time.time() - start_time
                            self.logger.info(f"Download completed using {strategy} strategy in {download_time:.2f}s: {video_file}")
                            
                            # Update metadata with download info
                            info.update({
                                'download_time': download_time,
                                'file_path': str(video_file),
                                'file_size': video_file.stat().st_size if video_file.exists() else 0,
                                'extraction_strategy': strategy,
                            })
                            
                            return video_file, info
                            
                    except Exception as alt_e:
                        self.logger.warning(f"Download with {strategy} strategy failed: {alt_e}")
                        continue
                
                # If all strategies failed, provide helpful error message
                raise ValueError(
                    f"Download failed due to bot detection. This commonly happens in cloud environments. "
                    f"Original error: {e}. "
                    f"To fix this, you may need to: "
                    f"1) Use a different video URL, "
                    f"2) Run from a different IP/location, "
                    f"3) Use proxy settings, or "
                    f"4) Try again later as YouTube's bot detection is temporary."
                )
            else:
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
        config = _get_config()
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
    
    def _get_alternative_ydl_opts(self, strategy: str = 'basic') -> Dict:
        """
        Get alternative yt-dlp options with minimal configurations.

        Args:
            strategy: Extraction strategy ('basic', 'simple', 'minimal')

        Returns:
            Dictionary of yt-dlp options for the specified strategy
        """
        base_opts = {
            'format': self._get_format_selector(),
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'writesubtitles': False,
            'writeautomaticsub': False,
            'writethumbnail': _get_config().EXTRACT_THUMBNAILS,
            'no_warnings': False,
            'extractflat': False,
            'ignoreerrors': False,
            'nocheckcertificate': True,
            'retries': 2,
            'fragment_retries': 2,
            'extractor_retries': 2,
        }

        # All strategies use minimal configurations to avoid bot detection
        if strategy == 'simple':
            base_opts.update({
                'socket_timeout': 20,
            })
        elif strategy == 'minimal':
            base_opts.update({
                'retries': 1,
                'fragment_retries': 1,
                'extractor_retries': 1,
            })

        return base_opts


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
