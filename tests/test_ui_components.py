"""
Comprehensive tests for Streamlit UI components and application.

Tests cover component rendering, session state management, URL validation,
file handling, error display, and user interaction workflows.
"""

import pytest
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.ui import components
from src.downloader import YouTubeURLValidator


class TestSessionStateInitialization:
    """Test session state initialization and management."""
    
    def test_initialize_session_state_sets_defaults(self):
        """Test that session state is initialized with correct defaults."""
        # Clear any existing session state
        if hasattr(st, 'session_state'):
            st.session_state.clear()
        
        # Mock streamlit session_state
        with patch.object(st, 'session_state', {}) as mock_session:
            components.initialize_session_state()
            
            # Check all expected defaults are set
            expected_defaults = {
                'url': '',
                'valid_url': False,
                'video_processed': False,
                'processing': False,
                'error': None,
                'current_step': '',
                'progress': 0,
                'clip_duration': 60,
                'content_type': 'auto',
                'number_of_clips': 3,
                'video_quality': '720p',
                'download_video': False,
                'theme': 'light',
                'settings_expanded': True,
                'debug_mode': False,
            }
            
            for key, expected_value in expected_defaults.items():
                assert key in mock_session
                assert mock_session[key] == expected_value
    
    def test_initialize_session_state_preserves_existing(self):
        """Test that existing session state values are preserved."""
        existing_state = {
            'url': 'https://www.youtube.com/watch?v=test123',
            'clip_duration': 120,
            'custom_key': 'custom_value'
        }
        
        with patch.object(st, 'session_state', existing_state) as mock_session:
            components.initialize_session_state()
            
            # Existing values should be preserved
            assert mock_session['url'] == 'https://www.youtube.com/watch?v=test123'
            assert mock_session['clip_duration'] == 120
            assert mock_session['custom_key'] == 'custom_value'
            
            # New defaults should be added
            assert mock_session['valid_url'] == False
            assert mock_session['processing'] == False


class TestHeaderRendering:
    """Test header component rendering."""
    
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.divider')
    def test_render_header_basic_structure(self, mock_divider, mock_metric, mock_columns, mock_markdown, mock_title):
        """Test that header renders with correct structure."""
        # Mock columns with context manager support
        mock_col1 = MagicMock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2 = MagicMock()
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_col3 = MagicMock()
        mock_col3.__enter__ = Mock(return_value=mock_col3)
        mock_col3.__exit__ = Mock(return_value=None)
        mock_col4 = MagicMock()
        mock_col4.__enter__ = Mock(return_value=mock_col4)
        mock_col4.__exit__ = Mock(return_value=None)
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3, mock_col4]
        
        with patch.object(st, 'session_state', {}):
            components.render_header()
            
            # Check title and description
            mock_title.assert_called_once_with("✂️ YouTube to Twitter Clipper")
            mock_markdown.assert_called_once()
            
            # Check columns and metrics
            mock_columns.assert_called_once_with(4)
            assert mock_metric.call_count == 4
            mock_divider.assert_called_once()
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_render_header_status_indicators(self, mock_metric, mock_columns):
        """Test that status indicators show correct values."""
        # Mock columns with context manager support
        mock_cols = []
        for i in range(4):
            mock_col = MagicMock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            mock_cols.append(mock_col)
        mock_columns.return_value = mock_cols
        
        with patch.object(st, 'session_state', {
            'processing': True,
            'video_processed': True,
            'generated_clips': 2,
            'error': 'Some error'
        }):
            with patch('streamlit.title'), patch('streamlit.markdown'), patch('streamlit.divider'):
                components.render_header()
                
                # Check metric calls
                metric_calls = mock_metric.call_args_list
                assert len(metric_calls) == 4
                
                # Status should show processing
                assert "🟡 Processing" in str(metric_calls[0])
                
                # Videos processed should be 1
                assert ("Videos Processed", 1) == metric_calls[1][0]
                
                # Clips generated should be 2
                assert ("Clips Generated", 2) == metric_calls[2][0]
                
                # Errors should be 1
                assert ("Errors", 1) == metric_calls[3][0]


class TestURLInputRendering:
    """Test URL input component."""
    
    @patch('streamlit.subheader')
    @patch('streamlit.text_input')
    def test_render_url_input_basic(self, mock_text_input, mock_subheader):
        """Test basic URL input rendering."""
        mock_text_input.return_value = ""
        
        with patch.object(st, 'session_state', {'url': ''}):
            with patch('streamlit.info'):
                result = components.render_url_input()
                
                mock_subheader.assert_called_once_with("🔗 YouTube URL Input")
                mock_text_input.assert_called_once()
                
                assert result['url'] == ""
                assert result['valid'] == False
                assert result['error'] is None
    
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.expander')
    def test_render_url_input_valid_url(self, mock_expander, mock_success, mock_text_input):
        """Test URL input with valid YouTube URL."""
        valid_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        mock_text_input.return_value = valid_url
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        with patch.object(st, 'session_state', {'url': valid_url}):
            with patch('streamlit.subheader'), patch('streamlit.json'):
                result = components.render_url_input()
                
                assert result['url'] == valid_url
                assert result['valid'] == True
                assert result['video_id'] == 'dQw4w9WgXcQ'
                assert result['normalized_url'] is not None
                
                mock_success.assert_called_once()
    
    @patch('streamlit.text_input')
    @patch('streamlit.error')
    @patch('streamlit.expander')
    def test_render_url_input_invalid_url(self, mock_expander, mock_error, mock_text_input):
        """Test URL input with invalid URL."""
        invalid_url = "https://example.com/not-youtube"
        mock_text_input.return_value = invalid_url
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        with patch.object(st, 'session_state', {'url': invalid_url}):
            with patch('streamlit.subheader'), patch('streamlit.markdown'):
                result = components.render_url_input()
                
                assert result['url'] == invalid_url
                assert result['valid'] == False
                assert result['error'] == "Invalid YouTube URL format"
                
                mock_error.assert_called_once()


class TestSettingsPanel:
    """Test settings panel component."""
    
    @patch('streamlit.expander')
    @patch('streamlit.subheader')
    @patch('streamlit.slider')
    @patch('streamlit.selectbox')
    @patch('streamlit.checkbox')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    def test_render_settings_panel_basic(self, mock_button, mock_columns, mock_checkbox, mock_selectbox, mock_slider, mock_subheader, mock_expander):
        """Test basic settings panel rendering."""
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        # Mock columns with context manager support  
        mock_col1 = MagicMock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2 = MagicMock()
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_columns.return_value = [mock_col1, mock_col2]
        
        # Mock return values
        mock_slider.return_value = 60
        mock_selectbox.side_effect = [3, 'auto', '720p']
        mock_checkbox.side_effect = [False, True, True]
        mock_button.return_value = False
        
        with patch.object(st, 'session_state', {}):
            components.render_settings_panel()
            
            # Check that UI elements were called
            mock_expander.assert_called()
            assert mock_subheader.call_count >= 3  # Multiple subheaders
            mock_slider.assert_called_once()
            assert mock_selectbox.call_count == 3
            assert mock_checkbox.call_count >= 3
    
    @patch('streamlit.expander')
    @patch('streamlit.slider')
    @patch('streamlit.selectbox')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    def test_render_settings_panel_updates_session_state(self, mock_button, mock_columns, mock_selectbox, mock_slider, mock_expander):
        """Test that settings panel updates session state correctly."""
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        # Mock columns with context manager support
        mock_col1 = MagicMock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2 = MagicMock()
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_columns.return_value = [mock_col1, mock_col2]
        
        mock_slider.return_value = 90
        mock_selectbox.side_effect = [5, 'educational', '1080p']
        mock_button.return_value = False
        
        session_state = {}
        with patch.object(st, 'session_state', session_state):
            with patch('streamlit.subheader'), patch('streamlit.checkbox'):
                components.render_settings_panel()
                
                assert session_state['clip_duration'] == 90
                assert session_state['number_of_clips'] == 5
                assert session_state['content_type'] == 'educational'
                assert session_state['video_quality'] == '1080p'
    
    @patch('streamlit.columns')
    @patch('streamlit.button')
    def test_render_settings_panel_quick_presets(self, mock_button, mock_columns):
        """Test quick preset buttons functionality."""
        # Mock columns with context manager support
        mock_col1 = MagicMock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2 = MagicMock()
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_columns.return_value = [mock_col1, mock_col2]
        
        # Mock button returns
        mock_button.side_effect = [True, False]  # First button clicked
        
        session_state = {}
        with patch.object(st, 'session_state', session_state):
            with patch('streamlit.expander'), patch('streamlit.subheader'), \
                 patch('streamlit.slider'), patch('streamlit.selectbox'), \
                 patch('streamlit.checkbox'), patch('streamlit.rerun'):
                components.render_settings_panel()
                
                # Check that preset values are set
                assert session_state['clip_duration'] == 30
                assert session_state['number_of_clips'] == 2
                assert session_state['video_quality'] == '480p'
                assert session_state['download_video'] == False


class TestProgressSection:
    """Test progress tracking component."""
    
    @patch('streamlit.subheader')
    @patch('streamlit.progress')
    @patch('streamlit.text')
    @patch('streamlit.expander')
    def test_render_progress_section_processing(self, mock_expander, mock_text, mock_progress, mock_subheader):
        """Test progress section during processing."""
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        with patch.object(st, 'session_state', {
            'processing': True,
            'progress': 50,
            'current_step': 'Downloading video...',
            'process_start_time': '2024-01-01 10:00:00'
        }):
            with patch('streamlit.json'):
                components.render_progress_section()
                
                mock_subheader.assert_called_with("⏳ Processing Progress")
                mock_progress.assert_called_with(50)
                mock_text.assert_called_with('Downloading video...')
    
    @patch('streamlit.subheader')
    @patch('streamlit.success')
    @patch('streamlit.expander')
    def test_render_progress_section_complete(self, mock_expander, mock_success, mock_subheader):
        """Test progress section when processing is complete."""
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        with patch.object(st, 'session_state', {
            'processing': False,
            'video_processed': True,
            'video_info': {'title': 'Test Video', 'duration': 120}
        }):
            with patch('streamlit.json'):
                components.render_progress_section()
                
                mock_subheader.assert_called_with("✅ Processing Complete")
                mock_success.assert_called_with("Video processed successfully!")
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    def test_render_progress_section_ready(self, mock_info, mock_subheader):
        """Test progress section in ready state."""
        with patch.object(st, 'session_state', {
            'processing': False,
            'video_processed': False
        }):
            components.render_progress_section()
            
            mock_subheader.assert_called_with("🎯 Ready to Process")
            mock_info.assert_called_with("Configure your settings and provide a video URL to begin processing.")


class TestErrorDisplay:
    """Test error display component."""
    
    @patch('streamlit.error')
    @patch('streamlit.expander')
    @patch('streamlit.markdown')
    @patch('streamlit.button')
    def test_render_error_display_basic(self, mock_button, mock_markdown, mock_expander, mock_error):
        """Test basic error display."""
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        mock_button.return_value = False
        
        with patch('streamlit.code'):
            components.render_error_display("Test error message", "Detailed error info")
            
            mock_error.assert_called_with("❌ **Error:** Test error message")
            mock_expander.assert_called()
            mock_markdown.assert_called()
    
    @patch('streamlit.button')
    def test_render_error_display_clear_error(self, mock_button):
        """Test clearing error functionality."""
        mock_button.return_value = True  # Button clicked
        
        session_state = {'error': 'Some error'}
        with patch.object(st, 'session_state', session_state):
            with patch('streamlit.error'), patch('streamlit.expander'), \
                 patch('streamlit.markdown'), patch('streamlit.code'), \
                 patch('streamlit.rerun') as mock_rerun:
                components.render_error_display("Test error")
                
                assert 'error' not in session_state
                mock_rerun.assert_called_once()


class TestVideoInfoDisplay:
    """Test video information display component."""
    
    @patch('streamlit.subheader')
    @patch('streamlit.metric')
    @patch('streamlit.expander')
    def test_render_video_info_display(self, mock_expander, mock_metric, mock_subheader):
        """Test video info display with complete metadata."""
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        video_info = {
            'title': 'Test Video',
            'duration': 125,  # 2:05
            'uploader': 'Test Channel',
            'upload_date': '2024-01-01',
            'view_count': 1500000,
            'like_count': 50000
        }
        
        with patch('streamlit.json'):
            components.render_video_info_display(video_info)
            
            mock_subheader.assert_called_with("📋 Video Information")
            assert mock_metric.call_count == 3  # Duration, Views, Likes
    
    def test_render_video_info_display_minimal(self):
        """Test video info display with minimal metadata."""
        video_info = {'title': 'Test Video'}
        
        with patch('streamlit.subheader'), patch('streamlit.expander'), patch('streamlit.json'):
            # Should not raise any exceptions
            components.render_video_info_display(video_info)


class TestThumbnailDisplay:
    """Test thumbnail display component."""
    
    @patch('streamlit.subheader')
    @patch('streamlit.image')
    @patch('streamlit.caption')
    def test_render_thumbnail_display_success(self, mock_caption, mock_image, mock_subheader):
        """Test successful thumbnail display."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(b'fake image data')
            thumbnail_path = tmp_file.name
        
        video_info = {'title': 'Test Video'}
        
        try:
            components.render_thumbnail_display(thumbnail_path, video_info)
            
            mock_subheader.assert_called_with("🖼️ Thumbnail")
            mock_image.assert_called_once()
            mock_caption.assert_called()
        finally:
            Path(thumbnail_path).unlink(missing_ok=True)
    
    @patch('streamlit.subheader')
    @patch('streamlit.error')
    def test_render_thumbnail_display_failure(self, mock_error, mock_subheader):
        """Test thumbnail display with missing file."""
        thumbnail_path = "/non/existent/path.jpg"
        video_info = {'title': 'Test Video'}
        
        components.render_thumbnail_display(thumbnail_path, video_info)
        
        mock_subheader.assert_called_with("🖼️ Thumbnail")
        mock_error.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_format_number(self):
        """Test number formatting function."""
        assert components.format_number(500) == "500"
        assert components.format_number(1500) == "1.5K"
        assert components.format_number(1500000) == "1.5M"
        assert components.format_number(2500000000) == "2.5B"
    
    @patch('streamlit.info')
    @patch('streamlit.markdown')
    def test_render_feature_preview(self, mock_markdown, mock_info):
        """Test feature preview rendering."""
        components.render_feature_preview(
            "Test Feature",
            "This feature will do something amazing",
            "Task 2.1"
        )
        
        mock_info.assert_called_with("🚧 **Test Feature** - Coming in Task 2.1")
        mock_markdown.assert_called_with("*This feature will do something amazing*")
    
    @patch('streamlit.subheader')
    @patch('streamlit.markdown')
    def test_render_keyboard_shortcuts(self, mock_markdown, mock_subheader):
        """Test keyboard shortcuts rendering."""
        components.render_keyboard_shortcuts()
        
        mock_subheader.assert_called_with("⌨️ Keyboard Shortcuts")
        assert mock_markdown.call_count >= 5  # Multiple shortcuts
    
    @patch('streamlit.subheader')
    @patch('streamlit.markdown')
    def test_render_tips_and_tricks(self, mock_markdown, mock_subheader):
        """Test tips and tricks rendering."""
        components.render_tips_and_tricks()
        
        mock_subheader.assert_called_with("💡 Tips & Tricks")
        assert mock_markdown.call_count >= 6  # Multiple tips


class TestDownloadButton:
    """Test download button functionality."""
    
    @patch('streamlit.download_button')
    def test_create_download_button_success(self, mock_download_button):
        """Test successful download button creation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b'test file content')
            file_path = tmp_file.name
        
        try:
            components.create_download_button(file_path, "test.txt", "text/plain")
            
            mock_download_button.assert_called_once()
            call_args = mock_download_button.call_args
            assert call_args[1]['file_name'] == "test.txt"
            assert call_args[1]['mime'] == "text/plain"
            assert call_args[1]['data'] == b'test file content'
        finally:
            Path(file_path).unlink(missing_ok=True)
    
    @patch('streamlit.error')
    def test_create_download_button_failure(self, mock_error):
        """Test download button with missing file."""
        components.create_download_button("/non/existent/file.txt", "test.txt")
        
        mock_error.assert_called_once()


class TestPerformanceMetrics:
    """Test performance metrics display."""
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_render_performance_metrics_processed(self, mock_metric, mock_columns, mock_subheader):
        """Test performance metrics when video is processed."""
        # Mock columns with context manager support
        mock_cols = []
        for i in range(3):
            mock_col = MagicMock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            mock_cols.append(mock_col)
        mock_columns.return_value = mock_cols
        
        with patch.object(st, 'session_state', {'video_processed': True}):
            components.render_performance_metrics()
            
            mock_subheader.assert_called_with("📊 Performance Metrics")
            assert mock_metric.call_count == 5  # 5 metrics
    
    def test_render_performance_metrics_not_processed(self):
        """Test performance metrics when video is not processed."""
        with patch.object(st, 'session_state', {'video_processed': False}):
            # Should return early and not render anything
            with patch('streamlit.subheader') as mock_subheader:
                components.render_performance_metrics()
                mock_subheader.assert_not_called()


class TestCustomCSS:
    """Test custom CSS injection."""
    
    @patch('streamlit.markdown')
    def test_inject_custom_css(self, mock_markdown):
        """Test custom CSS injection."""
        components.inject_custom_css()
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args
        assert 'unsafe_allow_html=True' in str(call_args)
        assert '<style>' in call_args[0][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
