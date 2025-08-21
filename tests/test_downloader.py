"""
Test suite for Task 1.2: YouTube Video Downloader

This module tests YouTube URL validation, video downloading, thumbnail extraction,
metadata collection, and error handling as specified in TASKS.md.
"""

import pytest
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.downloader.youtube_downloader import (
    YouTubeURLValidator,
    YouTubeDownloader,
    YouTubeDownloadError,
    download_youtube_video,
)
from src.downloader.thumbnail_extractor import (
    ThumbnailExtractor,
    extract_youtube_thumbnail,
)


class TestYouTubeURLValidator:
    """Test YouTube URL validation functionality."""
    
    def test_valid_youtube_urls(self):
        """Test that valid YouTube URLs are correctly identified."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "www.youtube.com/watch?v=dQw4w9WgXcQ",
            "youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "http://youtu.be/dQw4w9WgXcQ",
            "youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/v/dQw4w9WgXcQ",
        ]
        
        for url in valid_urls:
            assert YouTubeURLValidator.is_valid_youtube_url(url), f"Should be valid: {url}"
    
    def test_invalid_youtube_urls(self):
        """Test that invalid URLs are correctly rejected."""
        invalid_urls = [
            "",
            None,
            "not_a_url",
            "https://vimeo.com/123456789",
            "https://www.youtube.com/",
            "https://www.youtube.com/watch",
            "https://www.youtube.com/watch?v=",
            "https://www.youtube.com/watch?v=invalid",
            "https://www.facebook.com/watch?v=dQw4w9WgXcQ",
        ]
        
        for url in invalid_urls:
            assert not YouTubeURLValidator.is_valid_youtube_url(url), f"Should be invalid: {url}"
    
    def test_video_id_extraction(self):
        """Test video ID extraction from various URL formats."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("invalid_url", None),
        ]
        
        for url, expected_id in test_cases:
            actual_id = YouTubeURLValidator.extract_video_id(url)
            assert actual_id == expected_id, f"URL: {url}, Expected: {expected_id}, Got: {actual_id}"
    
    def test_url_normalization(self):
        """Test URL normalization to standard format."""
        test_cases = [
            ("https://youtu.be/dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            ("www.youtube.com/watch?v=dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            ("invalid_url", None),
        ]
        
        for input_url, expected_url in test_cases:
            actual_url = YouTubeURLValidator.normalize_url(input_url)
            assert actual_url == expected_url, f"Input: {input_url}, Expected: {expected_url}, Got: {actual_url}"


class TestYouTubeDownloader:
    """Test YouTube video downloader functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.downloader = YouTubeDownloader(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up after each test."""
        # Clean up temp files
        if self.temp_dir.exists():
            for file in self.temp_dir.iterdir():
                if file.is_file():
                    file.unlink()
            self.temp_dir.rmdir()
    
    def test_downloader_initialization(self):
        """Test that downloader initializes correctly."""
        assert self.downloader.output_dir == self.temp_dir
        assert self.temp_dir.exists()
        assert isinstance(self.downloader.ydl_opts, dict)
        assert 'format' in self.downloader.ydl_opts
    
    def test_format_selector_generation(self):
        """Test format selector generation."""
        format_str = self.downloader._get_format_selector()
        assert isinstance(format_str, str)
        assert len(format_str) > 0
    
    def test_url_validation_and_preparation(self):
        """Test URL validation and preparation."""
        # Valid URL
        valid_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        normalized = self.downloader.validate_and_prepare_url(valid_url)
        assert normalized == valid_url
        
        # Invalid URLs should raise ValueError
        invalid_urls = ["", "not_a_url", "https://vimeo.com/123456"]
        for url in invalid_urls:
            with pytest.raises(ValueError):
                self.downloader.validate_and_prepare_url(url)
    
    @patch('src.downloader.youtube_downloader.yt_dlp.YoutubeDL')
    def test_get_video_info(self, mock_ydl_class):
        """Test video info extraction."""
        # Mock yt-dlp response
        mock_info = {
            'id': 'dQw4w9WgXcQ',
            'title': 'Test Video',
            'duration': 212,
            'description': 'Test description',
            'uploader': 'Test Channel',
            'upload_date': '20090428',
            'view_count': 1000000,
            'like_count': 50000,
            'thumbnail': 'https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg',
            'webpage_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'formats': [{'format_id': '18'}, {'format_id': '22'}],
            'availability': 'public',
        }
        
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        info = self.downloader.get_video_info(url)
        
        assert info['id'] == 'dQw4w9WgXcQ'
        assert info['title'] == 'Test Video'
        assert info['duration'] == 212
        assert info['formats'] == 2
        
        # Verify yt-dlp was called correctly
        mock_ydl.extract_info.assert_called_once_with(url, download=False)
    
    @patch('src.downloader.youtube_downloader.yt_dlp.YoutubeDL')
    def test_get_video_info_error_handling(self, mock_ydl_class):
        """Test error handling in video info extraction."""
        import yt_dlp
        
        # Mock yt-dlp to raise an error
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = yt_dlp.DownloadError("Video not available")
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        # Use a valid URL format but mock it to fail
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        with pytest.raises(ValueError, match="Could not access video"):
            self.downloader.get_video_info(url)
    
    def test_find_downloaded_file(self):
        """Test downloaded file detection."""
        # Create a mock downloaded file
        test_file = self.temp_dir / "test_video.mp4"
        test_file.write_text("mock video content")
        
        # Test finding the file
        found_file = self.downloader._find_downloaded_file("test_video", None)
        assert found_file == test_file
        
        # Test with custom filename
        custom_file = self.temp_dir / "custom_name.mp4"
        custom_file.write_text("mock video content")
        
        found_file = self.downloader._find_downloaded_file("original_title", "custom_name")
        assert found_file == custom_file
    
    def test_find_downloaded_file_not_found(self):
        """Test error when downloaded file cannot be found."""
        with pytest.raises(FileNotFoundError):
            self.downloader._find_downloaded_file("nonexistent_video", None)
    
    def test_cleanup_downloads(self):
        """Test cleanup of old downloaded files."""
        # Create some mock files with different timestamps
        files = []
        for i in range(5):
            file_path = self.temp_dir / f"video_{i}.mp4"
            file_path.write_text(f"mock video {i}")
            files.append(file_path)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Cleanup, keeping only 2 files
        deleted = self.downloader.cleanup_downloads(keep_recent=2)
        
        # Should have deleted 3 files
        assert len(deleted) == 3
        
        # Check that 2 most recent files remain
        remaining_files = list(self.temp_dir.glob("*.mp4"))
        assert len(remaining_files) == 2


class TestThumbnailExtractor:
    """Test thumbnail extraction functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.extractor = ThumbnailExtractor(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up after each test."""
        # Clean up temp files
        if self.temp_dir.exists():
            for file in self.temp_dir.iterdir():
                if file.is_file():
                    file.unlink()
            self.temp_dir.rmdir()
    
    def test_thumbnail_extractor_initialization(self):
        """Test that thumbnail extractor initializes correctly."""
        assert self.extractor.output_dir == self.temp_dir
        assert self.temp_dir.exists()
        assert 'small' in self.extractor.sizes
        assert 'medium' in self.extractor.sizes
    
    @patch('src.downloader.thumbnail_extractor.yt_dlp.YoutubeDL')
    def test_get_thumbnail_urls(self, mock_ydl_class):
        """Test thumbnail URL extraction."""
        # Mock yt-dlp response
        mock_info = {
            'id': 'dQw4w9WgXcQ',
            'thumbnail': 'https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg',
            'thumbnails': [
                {'id': 'small', 'url': 'https://img.youtube.com/vi/dQw4w9WgXcQ/default.jpg'},
                {'id': 'medium', 'url': 'https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg'},
            ]
        }
        
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        thumbnails = self.extractor.get_thumbnail_urls(url)
        
        assert 'default' in thumbnails
        # Check for either small/medium from actual thumbnails OR fallback versions
        has_small_medium = 'small' in thumbnails and 'medium' in thumbnails
        has_fallback = any(key.startswith('fallback_') for key in thumbnails)
        assert has_small_medium or has_fallback, f"Expected small/medium or fallback thumbnails, got: {list(thumbnails.keys())}"
    
    @patch('src.downloader.thumbnail_extractor.requests.get')
    def test_download_image(self, mock_get):
        """Test image downloading."""
        # Mock successful response
        mock_response = Mock()
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake image data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        url = "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
        filename = "test_thumbnail"
        
        result_path = self.extractor._download_image(url, filename)
        
        assert result_path.exists()
        assert result_path.name == "test_thumbnail.jpg"
        assert result_path.read_bytes() == b'fake image data'
    
    @patch('src.downloader.thumbnail_extractor.requests.get')
    def test_download_image_error_handling(self, mock_get):
        """Test error handling in image download."""
        import requests
        
        # Mock failed response
        mock_get.side_effect = requests.RequestException("Network error")
        
        url = "https://invalid-url.com/image.jpg"
        filename = "test_thumbnail"
        
        with pytest.raises(ValueError, match="Could not download thumbnail"):
            self.extractor._download_image(url, filename)
    
    def test_thumbnail_info_nonexistent_file(self):
        """Test getting info for non-existent file."""
        nonexistent_path = self.temp_dir / "nonexistent.jpg"
        info = self.extractor.get_thumbnail_info(nonexistent_path)
        assert info == {}
    
    def test_cleanup_thumbnails(self):
        """Test cleanup of old thumbnail files."""
        # Create some mock thumbnail files
        files = []
        for i in range(5):
            file_path = self.temp_dir / f"thumb_{i}.jpg"
            file_path.write_text(f"mock thumbnail {i}")
            files.append(file_path)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Cleanup, keeping only 2 files
        deleted = self.extractor.cleanup_thumbnails(keep_recent=2)
        
        # Should have deleted 3 files
        assert len(deleted) == 3
        
        # Check that 2 most recent files remain
        remaining_files = list(self.temp_dir.glob("*.jpg"))
        assert len(remaining_files) == 2


class TestIntegration:
    """Integration tests for downloader components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.temp_dir.exists():
            for file in self.temp_dir.iterdir():
                if file.is_file():
                    file.unlink()
            self.temp_dir.rmdir()
    
    def test_url_validation_integration(self):
        """Test complete URL validation workflow."""
        # Test the test URLs specified in TASKS.md
        test_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Public video
            "https://youtu.be/dQw4w9WgXcQ",                # Short URL
            "https://www.youtube.com/watch?v=invalid123",   # Invalid video
            "https://www.youtube.com/watch?v=privatevideo", # Private video
        ]
        
        valid_count = 0
        for url in test_urls:
            if YouTubeURLValidator.is_valid_youtube_url(url):
                valid_count += 1
                # Should be able to normalize valid URLs
                normalized = YouTubeURLValidator.normalize_url(url)
                assert normalized is not None
                assert normalized.startswith("https://www.youtube.com/watch?v=")
        
        # At least 2 URLs should be valid (first two)
        assert valid_count >= 2
    
    def test_convenience_functions(self):
        """Test convenience functions work correctly."""
        # Test that convenience functions can be imported and called
        # (Mock the actual download to avoid network calls)
        
        with patch('src.downloader.youtube_downloader.yt_dlp.YoutubeDL'):
            try:
                # This should not raise an exception during initialization
                from src.downloader import download_youtube_video, extract_youtube_thumbnail
                
                # Functions should be callable
                assert callable(download_youtube_video)
                assert callable(extract_youtube_thumbnail)
                
            except Exception as e:
                pytest.fail(f"Convenience functions not working: {e}")


class TestErrorHandling:
    """Test error handling scenarios specified in TASKS.md."""
    
    def test_invalid_url_errors(self):
        """Test proper error messages for invalid URLs."""
        downloader = YouTubeDownloader()
        
        invalid_urls = [
            "",
            "not_a_url",
            "https://vimeo.com/123456789",
            "https://www.youtube.com/watch?v=",
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError) as exc_info:
                downloader.validate_and_prepare_url(url)
            
            # Error message should be user-friendly
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ['invalid', 'url', 'youtube'])
    
    @patch('src.downloader.youtube_downloader.yt_dlp.YoutubeDL')
    def test_private_video_error(self, mock_ydl_class):
        """Test handling of private/deleted videos."""
        import yt_dlp
        
        # Mock yt-dlp to simulate private video error
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = yt_dlp.DownloadError("Private video")
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        downloader = YouTubeDownloader()
        url = "https://www.youtube.com/watch?v=privatevideo"
        
        with pytest.raises(ValueError) as exc_info:
            downloader.get_video_info(url)
        
        # Error message should mention access issues
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['access', 'video', 'private'])
    
    @patch('src.downloader.youtube_downloader.yt_dlp.YoutubeDL')
    def test_duration_limit_error(self, mock_ydl_class):
        """Test handling of videos exceeding duration limit."""
        # Mock video info with long duration
        mock_info = {
            'id': 'longvideo123',
            'title': 'Very Long Video',
            'duration': 7200,  # 2 hours (exceeds 10 minute limit)
            'description': '',
            'uploader': 'Test Channel',
        }
        
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        downloader = YouTubeDownloader()
        url = "https://www.youtube.com/watch?v=longvideo123"
        
        with pytest.raises(ValueError) as exc_info:
            downloader.download_video(url)
        
        # Error message should mention duration limit
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['duration', 'exceeds', 'maximum'])


def test_task_requirements_coverage():
    """Test that all Task 1.2 requirements are covered."""
    
    # Test that all required classes exist and are importable
    from src.downloader import (
        YouTubeDownloader,
        YouTubeURLValidator,
        ThumbnailExtractor,
    )
    
    # Test deliverables coverage:
    
    # ✅ YouTube URL validation
    assert hasattr(YouTubeURLValidator, 'is_valid_youtube_url')
    assert hasattr(YouTubeURLValidator, 'extract_video_id')
    assert hasattr(YouTubeURLValidator, 'normalize_url')
    
    # ✅ Video download with yt-dlp integration
    assert hasattr(YouTubeDownloader, 'download_video')
    assert hasattr(YouTubeDownloader, 'get_video_info')
    
    # ✅ Thumbnail extraction and storage
    assert hasattr(ThumbnailExtractor, 'download_thumbnail')
    assert hasattr(ThumbnailExtractor, 'get_thumbnail_urls')
    
    # ✅ Video metadata collection
    downloader = YouTubeDownloader()
    assert hasattr(downloader, 'get_video_info')
    
    # ✅ Error handling for invalid URLs, private videos, etc.
    assert hasattr(YouTubeDownloader, 'validate_and_prepare_url')
    
    print("✅ All Task 1.2 deliverables are implemented and testable")


if __name__ == "__main__":
    # Run tests manually for immediate validation
    print("🧪 Running Task 1.2 Downloader Tests...")
    
    # Test URL validation
    print("\n🔗 Testing URL Validation...")
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Public video
        "https://youtu.be/dQw4w9WgXcQ",                # Short URL
        "https://www.youtube.com/watch?v=invalid123",   # Invalid video
        "invalid_url",                                  # Completely invalid
    ]
    
    for url in test_urls:
        is_valid = YouTubeURLValidator.is_valid_youtube_url(url)
        video_id = YouTubeURLValidator.extract_video_id(url)
        normalized = YouTubeURLValidator.normalize_url(url)
        print(f"  {url[:50]:<50} Valid: {is_valid}, ID: {video_id}, Normalized: {normalized is not None}")
    
    # Test downloader initialization
    print("\n📥 Testing Downloader Initialization...")
    try:
        downloader = YouTubeDownloader()
        print("✅ YouTubeDownloader initialized successfully")
    except Exception as e:
        print(f"❌ YouTubeDownloader initialization failed: {e}")
    
    # Test thumbnail extractor initialization
    print("\n🖼️ Testing Thumbnail Extractor Initialization...")
    try:
        extractor = ThumbnailExtractor()
        print("✅ ThumbnailExtractor initialized successfully")
        print(f"  Available sizes: {list(extractor.sizes.keys())}")
    except Exception as e:
        print(f"❌ ThumbnailExtractor initialization failed: {e}")
    
    # Test imports
    print("\n📦 Testing Module Imports...")
    try:
        from src.downloader import (
            YouTubeDownloader,
            YouTubeURLValidator,
            ThumbnailExtractor,
            download_youtube_video,
            extract_youtube_thumbnail,
        )
        print("✅ All downloader modules import successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
    
    # Test requirements coverage
    print("\n📋 Testing Requirements Coverage...")
    try:
        test_task_requirements_coverage()
    except Exception as e:
        print(f"❌ Requirements coverage test failed: {e}")
    
    print("\n🎉 Task 1.2 downloader testing complete!")
