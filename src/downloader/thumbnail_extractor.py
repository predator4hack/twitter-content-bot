"""
Thumbnail Extractor Module

This module provides functionality for extracting and processing YouTube video thumbnails,
including downloading, resizing, and format conversion for optimal display.
"""

import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from PIL import Image, ImageOps
import yt_dlp

from src.core.config import config
from src.core.logger import get_logger, LoggerMixin


class ThumbnailExtractor(LoggerMixin):
    """
    Extracts and processes YouTube video thumbnails.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the thumbnail extractor.
        
        Args:
            output_dir: Directory to save thumbnails
        """
        self.output_dir = output_dir or (config.CACHE_DIR / "thumbnails")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp'}
        
        # Standard thumbnail sizes for different uses
        self.sizes = {
            'small': (160, 90),      # YouTube small thumbnail
            'medium': (320, 180),    # YouTube medium thumbnail  
            'large': (480, 360),     # YouTube large thumbnail
            'maxres': (1280, 720),   # Maximum resolution
            'twitter': (1200, 675),  # Twitter card optimal size
            'preview': (640, 360),   # UI preview size
        }
        
        self.logger.info(f"Thumbnail extractor initialized with output dir: {self.output_dir}")
    
    def get_thumbnail_urls(self, video_url: str) -> Dict[str, str]:
        """
        Get all available thumbnail URLs for a YouTube video.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Dictionary mapping quality names to thumbnail URLs
            
        Raises:
            ValueError: If video cannot be accessed
        """
        try:
            # Use robust options for thumbnail extraction
            info_opts = {
                'quiet': False,
                'no_warnings': False,
                'extract_flat': False,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                },
                'retries': 3,
                'fragment_retries': 3,
                'extractor_retries': 3,
            }
            
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                if not info:
                    raise ValueError("No video information could be extracted")
                
                thumbnails = {}
                
                # Extract thumbnails from info
                if 'thumbnails' in info and info['thumbnails']:
                    for thumb in info['thumbnails']:
                        thumb_id = thumb.get('id', 'unknown')
                        thumb_url = thumb.get('url')
                        if thumb_url:
                            thumbnails[thumb_id] = thumb_url
                
                # Add direct thumbnail URL if available
                if 'thumbnail' in info and info['thumbnail']:
                    thumbnails['default'] = info['thumbnail']
                
                # Generate standard YouTube thumbnail URLs as fallback
                video_id = info.get('id')
                if video_id:
                    base_url = f"https://img.youtube.com/vi/{video_id}"
                    standard_thumbs = {
                        'default': f"{base_url}/default.jpg",
                        'mqdefault': f"{base_url}/mqdefault.jpg",
                        'hqdefault': f"{base_url}/hqdefault.jpg",
                        'sddefault': f"{base_url}/sddefault.jpg",
                        'maxresdefault': f"{base_url}/maxresdefault.jpg",
                    }
                    # Only add if we don't already have thumbnails
                    if not thumbnails:
                        thumbnails.update(standard_thumbs)
                    else:
                        # Add as fallback options
                        for key, url in standard_thumbs.items():
                            if key not in thumbnails:
                                thumbnails[f"fallback_{key}"] = url
                
                self.logger.info(f"Found {len(thumbnails)} thumbnail URLs")
                return thumbnails
                
        except Exception as e:
            self.logger.error(f"Failed to get thumbnail URLs: {e}")
            raise ValueError(f"Could not extract thumbnail URLs: {e}")
    
    def download_thumbnail(
        self, 
        video_url: str, 
        quality: str = 'hqdefault',
        filename: Optional[str] = None
    ) -> Path:
        """
        Download a specific thumbnail for a YouTube video.
        
        Args:
            video_url: YouTube video URL
            quality: Thumbnail quality to download
            filename: Optional custom filename
            
        Returns:
            Path to downloaded thumbnail file
            
        Raises:
            ValueError: If thumbnail cannot be downloaded
        """
        thumbnail_urls = self.get_thumbnail_urls(video_url)
        
        if quality not in thumbnail_urls:
            # Fallback to available qualities
            available_qualities = list(thumbnail_urls.keys())
            if not available_qualities:
                raise ValueError("No thumbnails available for this video")
            
            # Try common quality fallbacks
            fallback_order = ['hqdefault', 'mqdefault', 'default', 'maxresdefault']
            quality = next(
                (q for q in fallback_order if q in thumbnail_urls), 
                available_qualities[0]
            )
            self.logger.warning(f"Requested quality not available, using: {quality}")
        
        thumbnail_url = thumbnail_urls[quality]
        
        # Generate filename
        if not filename:
            # Extract video ID for filename
            try:
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    if info:
                        video_id = info.get('id', 'unknown')
                        filename = f"{video_id}_{quality}"
                    else:
                        filename = f"unknown_{quality}"
            except:
                filename = f"thumbnail_{quality}"
        
        return self._download_image(thumbnail_url, filename)
    
    def _download_image(self, url: str, filename: str) -> Path:
        """
        Download an image from URL.
        
        Args:
            url: Image URL
            filename: Base filename (without extension)
            
        Returns:
            Path to downloaded image
            
        Raises:
            ValueError: If download fails
        """
        try:
            self.logger.debug(f"Downloading thumbnail from: {url}")
            
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Determine file extension from content type or URL
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'webp' in content_type:
                ext = '.webp'
            else:
                # Fallback: try to get from URL
                parsed_url = urlparse(url)
                url_ext = Path(parsed_url.path).suffix.lower()
                ext = url_ext if url_ext in self.supported_formats else '.jpg'
            
            file_path = self.output_dir / f"{filename}{ext}"
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Thumbnail downloaded: {file_path}")
            return file_path
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download thumbnail: {e}")
            raise ValueError(f"Could not download thumbnail: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error downloading thumbnail: {e}")
            raise ValueError(f"Error downloading thumbnail: {e}")
    
    def resize_thumbnail(
        self, 
        image_path: Path, 
        size: str = 'preview',
        maintain_aspect: bool = True,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Resize a thumbnail image to specified dimensions.
        
        Args:
            image_path: Path to source image
            size: Size preset name or tuple (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            output_filename: Optional output filename
            
        Returns:
            Path to resized image
            
        Raises:
            ValueError: If resize operation fails
        """
        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")
        
        # Get target dimensions
        if isinstance(size, str):
            if size not in self.sizes:
                raise ValueError(f"Unknown size preset: {size}. Available: {list(self.sizes.keys())}")
            target_size = self.sizes[size]
        else:
            target_size = size
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                if maintain_aspect:
                    # Resize maintaining aspect ratio
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    
                    # Create new image with exact dimensions (padded if necessary)
                    new_img = Image.new('RGB', target_size, (0, 0, 0))
                    
                    # Center the image
                    paste_x = (target_size[0] - img.width) // 2
                    paste_y = (target_size[1] - img.height) // 2
                    new_img.paste(img, (paste_x, paste_y))
                    
                    img = new_img
                else:
                    # Resize to exact dimensions (may distort)
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Generate output filename
                if not output_filename:
                    base_name = image_path.stem
                    size_suffix = f"_{size}" if isinstance(size, str) else f"_{target_size[0]}x{target_size[1]}"
                    output_filename = f"{base_name}{size_suffix}"
                
                output_path = self.output_dir / f"{output_filename}.jpg"
                
                # Save resized image
                img.save(output_path, 'JPEG', quality=90, optimize=True)
                
                self.logger.info(f"Thumbnail resized to {target_size}: {output_path}")
                return output_path
                
        except Exception as e:
            self.logger.error(f"Failed to resize thumbnail: {e}")
            raise ValueError(f"Could not resize thumbnail: {e}")
    
    def extract_and_process_thumbnails(
        self, 
        video_url: str,
        sizes: Optional[List[str]] = None,
        filename_prefix: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Extract and process multiple thumbnail sizes for a video.
        
        Args:
            video_url: YouTube video URL
            sizes: List of size presets to generate
            filename_prefix: Optional prefix for generated files
            
        Returns:
            Dictionary mapping size names to file paths
        """
        if sizes is None:
            sizes = ['small', 'medium', 'large', 'preview']
        
        results = {}
        
        try:
            # Download the highest quality thumbnail available
            original_thumb = self.download_thumbnail(
                video_url, 
                'maxresdefault',  # Try highest quality first
                filename_prefix
            )
            
            # Generate different sizes
            for size in sizes:
                try:
                    size_suffix = f"_{size}" if filename_prefix else size
                    output_name = f"{filename_prefix}{size_suffix}" if filename_prefix else f"thumb_{size}"
                    
                    resized_thumb = self.resize_thumbnail(
                        original_thumb,
                        size,
                        maintain_aspect=True,
                        output_filename=output_name
                    )
                    
                    results[size] = resized_thumb
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create {size} thumbnail: {e}")
            
            # Keep original as 'original'
            results['original'] = original_thumb
            
            self.logger.info(f"Generated {len(results)} thumbnail variants")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to extract and process thumbnails: {e}")
            raise ValueError(f"Thumbnail processing failed: {e}")
    
    def get_thumbnail_info(self, image_path: Path) -> Dict:
        """
        Get information about a thumbnail image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image information
        """
        if not image_path.exists():
            return {}
        
        try:
            with Image.open(image_path) as img:
                return {
                    'path': str(image_path),
                    'filename': image_path.name,
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'file_size': image_path.stat().st_size,
                    'aspect_ratio': round(img.width / img.height, 2) if img.height > 0 else 0,
                }
        except Exception as e:
            self.logger.error(f"Failed to get thumbnail info: {e}")
            return {'path': str(image_path), 'error': str(e)}
    
    def cleanup_thumbnails(self, keep_recent: int = 20) -> List[Path]:
        """
        Clean up old thumbnail files.
        
        Args:
            keep_recent: Number of recent files to keep
            
        Returns:
            List of deleted file paths
        """
        if not config.CLEANUP_TEMP_FILES:
            return []
        
        # Get all thumbnail files
        thumbnail_files = []
        for ext in self.supported_formats:
            thumbnail_files.extend(self.output_dir.glob(f"*{ext}"))
        
        # Sort by modification time (newest first)
        thumbnail_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep the most recent files
        files_to_delete = thumbnail_files[keep_recent:]
        deleted_files = []
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                deleted_files.append(file_path)
                self.logger.debug(f"Deleted old thumbnail: {file_path}")
            except Exception as e:
                self.logger.warning(f"Could not delete {file_path}: {e}")
        
        if deleted_files:
            self.logger.info(f"Cleaned up {len(deleted_files)} old thumbnails")
        
        return deleted_files


# Convenience function for quick thumbnail extraction
def extract_youtube_thumbnail(
    video_url: str,
    quality: str = 'hqdefault',
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None
) -> Path:
    """
    Convenience function to extract a YouTube thumbnail.
    
    Args:
        video_url: YouTube video URL
        quality: Thumbnail quality
        output_dir: Output directory
        filename: Optional filename
        
    Returns:
        Path to downloaded thumbnail
    """
    extractor = ThumbnailExtractor(output_dir)
    return extractor.download_thumbnail(video_url, quality, filename)
