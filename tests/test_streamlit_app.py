"""
Tests for the main Streamlit application functionality.

Tests cover the main application workflow, session state management,
and integration between components.
"""

import pytest
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.ui import streamlit_app


class TestMainAppFunctions:
    """Test main application functions."""
    
    @patch('streamlit.title')
    @patch('streamlit.markdown')  
    def test_main_function_basic_structure(self, mock_markdown, mock_title):
        """Test that main function sets up the app correctly."""
        with patch('src.ui.streamlit_app.initialize_session_state'), \
             patch('src.ui.streamlit_app.render_header'), \
             patch('src.ui.streamlit_app.render_url_input'), \
             patch('src.ui.streamlit_app.render_file_upload_section'), \
             patch('src.ui.streamlit_app.render_settings_panel'), \
             patch('src.ui.streamlit_app.render_progress_section'), \
             patch('src.ui.streamlit_app.render_results_section'), \
             patch('src.ui.streamlit_app.render_sidebar'), \
             patch('streamlit.button'), \
             patch('streamlit.columns'), \
             patch('streamlit.container'):
            
            # Should not raise any exceptions
            streamlit_app.main()
    
    def test_process_video_workflow_invalid_url(self):
        """Test video processing workflow with invalid URL."""
        url_input_result = {
            'url': 'invalid-url',
            'valid': False,
            'video_id': None,
            'error': 'Invalid URL'
        }
        
        session_state = {
            'uploaded_file': None
        }
        
        with patch.object(st, 'session_state', session_state):
            with patch('streamlit.error') as mock_error:
                result = streamlit_app.process_video_workflow(url_input_result)
                
                assert result == False
                mock_error.assert_called()
    
    def test_process_video_workflow_no_input(self):
        """Test video processing workflow with no input."""
        url_input_result = {
            'url': '',
            'valid': False,
            'video_id': None,
            'error': None
        }
        
        session_state = {
            'uploaded_file': None
        }
        
        with patch.object(st, 'session_state', session_state):
            with patch('streamlit.error') as mock_error:
                result = streamlit_app.process_video_workflow(url_input_result)
                
                assert result == False
                mock_error.assert_called()
    
    @patch('src.ui.streamlit_app.YouTubeDownloader')
    @patch('src.ui.streamlit_app.ThumbnailExtractor')
    def test_process_video_workflow_youtube_success(self, mock_thumb_class, mock_downloader_class):
        """Test successful YouTube video processing."""
        url_input_result = {
            'url': 'https://www.youtube.com/watch?v=test123',
            'valid': True,
            'video_id': 'test123',
            'error': None
        }
        
        # Mock session state
        session_state = {
            'uploaded_file': None,
            'video_quality': '720p',
            'download_video': True,
            'processing': False,
            'progress': 0
        }
        
        # Mock downloader
        mock_downloader = Mock()
        mock_downloader.get_video_info.return_value = {
            'title': 'Test Video',
            'duration': 120,
            'uploader': 'Test Channel'
        }
        mock_downloader.download_video.return_value = '/path/to/video.mp4'
        mock_downloader_class.return_value = mock_downloader
        
        # Mock thumbnail extractor
        mock_extractor = Mock()
        mock_extractor.extract_thumbnail.return_value = '/path/to/thumb.jpg'
        mock_thumb_class.return_value = mock_extractor
        
        with patch.object(st, 'session_state', session_state):
            with patch('streamlit.success'), patch('streamlit.info'), \
                 patch('streamlit.progress'), patch('time.sleep'):
                
                result = streamlit_app.process_video_workflow(url_input_result)
                
                assert result == True
                assert session_state['video_processed'] == True
                assert session_state['processing'] == False
    
    @patch('src.ui.streamlit_app.YouTubeDownloader')
    def test_process_video_workflow_youtube_failure(self, mock_downloader_class):
        """Test YouTube video processing with failure."""
        url_input_result = {
            'url': 'https://www.youtube.com/watch?v=test123',
            'valid': True,
            'video_id': 'test123',
            'error': None
        }
        
        session_state = {
            'uploaded_file': None,
            'processing': False
        }
        
        # Mock downloader that raises exception
        mock_downloader = Mock()
        mock_downloader.get_video_info.side_effect = Exception("Download failed")
        mock_downloader_class.return_value = mock_downloader
        
        with patch.object(st, 'session_state', session_state):
            with patch('streamlit.error') as mock_error:
                result = streamlit_app.process_video_workflow(url_input_result)
                
                assert result == False
                mock_error.assert_called()
                assert session_state['processing'] == False


class TestRenderFunctions:
    """Test individual render functions."""
    
    @patch('streamlit.file_uploader')
    @patch('streamlit.subheader')
    def test_render_file_upload_section_basic(self, mock_subheader, mock_file_uploader):
        """Test basic file upload rendering."""
        mock_file_uploader.return_value = None
        
        with patch.object(st, 'session_state', {}):
            streamlit_app.render_file_upload_section()
            
            mock_subheader.assert_called_with("📁 Local File Upload")
            mock_file_uploader.assert_called()
    
    @patch('streamlit.file_uploader')
    def test_render_file_upload_section_with_file(self, mock_file_uploader):
        """Test file upload with uploaded file."""
        mock_file = Mock()
        mock_file.name = "test_video.mp4"
        mock_file.size = 50 * 1024 * 1024  # 50MB
        mock_file_uploader.return_value = mock_file
        
        with patch.object(st, 'session_state', {}):
            with patch('streamlit.subheader'), patch('streamlit.success'), \
                 patch('streamlit.info') as mock_info:
                streamlit_app.render_file_upload_section()
                
                # Should show file info
                mock_info.assert_called()
    
    def test_render_results_section_no_results(self):
        """Test results section with no results."""
        session_state = {'video_processed': False}
        
        with patch.object(st, 'session_state', session_state):
            with patch('streamlit.info') as mock_info:
                streamlit_app.render_results_section()
                
                mock_info.assert_called_with("🎬 Process a video to see results here")
    
    def test_render_results_section_with_results(self):
        """Test results section with processing results."""
        session_state = {
            'video_processed': True,
            'video_info': {'title': 'Test Video', 'duration': 120},
            'thumbnail_path': '/path/to/thumb.jpg'
        }
        
        with patch.object(st, 'session_state', session_state):
            with patch('src.ui.streamlit_app.render_video_info_display'), \
                 patch('src.ui.streamlit_app.render_thumbnail_display'), \
                 patch('streamlit.subheader'), \
                 patch('streamlit.info'):
                
                # Should not raise exceptions
                streamlit_app.render_results_section()
    
    @patch('streamlit.sidebar')
    def test_render_sidebar_basic(self, mock_sidebar):
        """Test basic sidebar rendering."""
        mock_sidebar.title = Mock()
        mock_sidebar.markdown = Mock()
        mock_sidebar.selectbox = Mock(return_value='🌙 Dark')
        mock_sidebar.checkbox = Mock(return_value=False)
        mock_sidebar.divider = Mock()
        mock_sidebar.expander = Mock()
        mock_expander = mock_sidebar.expander.return_value
        mock_expander.__enter__ = Mock()
        mock_expander.__exit__ = Mock()
        
        with patch.object(st, 'session_state', {}):
            with patch('streamlit.button'), \
                 patch('src.ui.streamlit_app.handle_session_reset'):
                
                streamlit_app.render_sidebar()
                
                # Should call sidebar functions
                mock_sidebar.title.assert_called()
    
    def test_handle_session_reset(self):
        """Test session reset functionality."""
        session_state = {
            'video_processed': True,
            'processing': True,
            'error': 'Some error'
        }
        
        with patch.object(st, 'session_state', session_state):
            with patch('streamlit.success') as mock_success:
                streamlit_app.handle_session_reset()
                
                assert session_state['video_processed'] == False
                assert session_state['processing'] == False
                assert session_state['error'] is None
                mock_success.assert_called()
    
    def test_render_debug_section_enabled(self):
        """Test debug section when enabled."""
        session_state = {
            'debug_mode': True,
            'video_info': {'title': 'Test'},
            'thumbnail_path': '/path/to/thumb.jpg'
        }
        
        with patch.object(st, 'session_state', session_state):
            with patch('streamlit.subheader'), \
                 patch('streamlit.json'), \
                 patch('streamlit.code'):
                
                streamlit_app.render_debug_section()
    
    def test_render_debug_section_disabled(self):
        """Test debug section when disabled."""
        session_state = {'debug_mode': False}
        
        with patch.object(st, 'session_state', session_state):
            # Should return early and not render anything
            with patch('streamlit.subheader') as mock_subheader:
                streamlit_app.render_debug_section()
                mock_subheader.assert_not_called()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_update_progress(self):
        """Test progress update function."""
        mock_progress_bar = Mock()
        mock_status_text = Mock()
        
        streamlit_app.update_progress(
            mock_progress_bar, 
            mock_status_text, 
            50, 
            "Testing progress"
        )
        
        mock_progress_bar.progress.assert_called_with(0.5)
        mock_status_text.text.assert_called_with("Testing progress")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
