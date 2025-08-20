"""
UI Components for Streamlit Application

This module provides reusable UI components for the YouTube to Twitter clipper,
including input forms, progress indicators, error displays, and result presentations.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.config import config
from src.core.logger import get_logger
from src.downloader import YouTubeURLValidator

logger = get_logger("ui_components")


def initialize_session_state():
    """Initialize Streamlit session state with default values."""
    
    # Initialize default values if not present
    defaults = {
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
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    """Render the application header with title and description."""
    
    st.title("✂️ YouTube to Twitter Clipper")
    st.markdown("""
    **AI-powered video clip extraction for Twitter sharing**
    
    Transform your YouTube videos into engaging Twitter clips with AI-powered content analysis.
    """)
    
    # Add status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "🟢 Ready" if not st.session_state.get('processing', False) else "🟡 Processing"
        st.metric("Status", status)
    
    with col2:
        video_count = 1 if st.session_state.get('video_processed', False) else 0
        st.metric("Videos Processed", video_count)
    
    with col3:
        clip_count = st.session_state.get('generated_clips', 0)
        st.metric("Clips Generated", clip_count)
    
    with col4:
        error_count = 1 if st.session_state.get('error') else 0
        st.metric("Errors", error_count, delta_color="inverse")
    
    st.divider()


def render_url_input() -> Dict[str, Any]:
    """
    Render URL input section with validation.
    
    Returns:
        Dictionary containing URL validation results
    """
    
    st.subheader("🔗 YouTube URL Input")
    
    # URL input with help text
    url = st.text_input(
        "Enter YouTube URL",
        value=st.session_state.get('url', ''),
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter any valid YouTube URL format (youtube.com, youtu.be, etc.)"
    )
    
    # Update session state
    st.session_state['url'] = url
    
    # Real-time validation
    validation_result = {
        'url': url,
        'valid': False,
        'video_id': None,
        'normalized_url': None,
        'error': None
    }
    
    if url.strip():
        try:
            # Validate URL
            is_valid = YouTubeURLValidator.is_valid_youtube_url(url)
            
            if is_valid:
                video_id = YouTubeURLValidator.extract_video_id(url)
                normalized_url = YouTubeURLValidator.normalize_url(url)
                
                validation_result.update({
                    'valid': True,
                    'video_id': video_id,
                    'normalized_url': normalized_url
                })
                
                # Display success
                st.success(f"✅ Valid YouTube URL detected")
                
                # Show URL info
                with st.expander("🔍 URL Information"):
                    st.json({
                        "Video ID": video_id,
                        "Normalized URL": normalized_url,
                        "URL Format": "Valid YouTube format"
                    })
                
            else:
                validation_result['error'] = "Invalid YouTube URL format"
                st.error("❌ Invalid YouTube URL format")
                
                # Show help
                with st.expander("💡 Supported URL Formats"):
                    st.markdown("""
                    **Supported YouTube URL formats:**
                    - `https://www.youtube.com/watch?v=VIDEO_ID`
                    - `https://youtu.be/VIDEO_ID`
                    - `https://youtube.com/watch?v=VIDEO_ID`
                    - `www.youtube.com/watch?v=VIDEO_ID`
                    
                    **Examples:**
                    - `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
                    - `https://youtu.be/dQw4w9WgXcQ`
                    """)
                
        except Exception as e:
            validation_result['error'] = str(e)
            st.error(f"❌ URL validation error: {e}")
    
    elif url == "":
        # Empty URL - show placeholder help
        st.info("👆 Enter a YouTube URL above to get started")
    
    # Update session state
    st.session_state['valid_url'] = validation_result['valid']
    
    return validation_result


def render_settings_panel():
    """Render the settings configuration panel."""
    
    # Settings expander
    with st.expander("⚙️ Processing Settings", expanded=st.session_state.get('settings_expanded', True)):
        
        # Clip settings
        st.subheader("✂️ Clip Settings")
        
        clip_duration = st.slider(
            "Maximum Clip Duration (seconds)",
            min_value=15,
            max_value=140,  # Twitter limit is 2:20
            value=st.session_state.get('clip_duration', 60),
            step=5,
            help="Maximum duration for each clip (Twitter limit: 2:20)"
        )
        st.session_state['clip_duration'] = clip_duration
        
        number_of_clips = st.selectbox(
            "Number of Clips to Generate",
            options=[1, 2, 3, 4, 5],
            index=2,  # Default to 3
            help="How many clips to extract from the video"
        )
        st.session_state['number_of_clips'] = number_of_clips
        
        content_type = st.selectbox(
            "Content Type",
            options=['auto', 'educational', 'entertainment', 'interview', 'tutorial', 'comedy', 'music'],
            index=0,
            help="Content type for better AI analysis (auto-detect recommended)"
        )
        st.session_state['content_type'] = content_type
        
        # Video quality settings
        st.subheader("📹 Video Quality")
        
        video_quality = st.selectbox(
            "Download Quality",
            options=['480p', '720p', '1080p', 'best'],
            index=1,  # Default to 720p
            help="Video quality for processing (higher quality = larger files)"
        )
        st.session_state['video_quality'] = video_quality
        
        download_video = st.checkbox(
            "Download full video",
            value=st.session_state.get('download_video', False),
            help="Download the full video file (required for clip extraction)"
        )
        st.session_state['download_video'] = download_video
        
        # Advanced settings
        st.subheader("🔧 Advanced Settings")
        
        parallel_processing = st.checkbox(
            "Enable parallel processing",
            value=config.PARALLEL_PROCESSING,
            help="Process multiple clips in parallel (faster but uses more resources)"
        )
        
        cleanup_files = st.checkbox(
            "Auto-cleanup temporary files",
            value=config.CLEANUP_TEMP_FILES,
            help="Automatically delete temporary files after processing"
        )
        
        # Update session state
        st.session_state['parallel_processing'] = parallel_processing
        st.session_state['cleanup_files'] = cleanup_files
    
    # Quick presets
    st.subheader("🎯 Quick Presets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚡ Fast & Light", use_container_width=True):
            st.session_state.update({
                'clip_duration': 30,
                'number_of_clips': 2,
                'video_quality': '480p',
                'download_video': False
            })
            st.rerun()
    
    with col2:
        if st.button("🎬 High Quality", use_container_width=True):
            st.session_state.update({
                'clip_duration': 90,
                'number_of_clips': 3,
                'video_quality': '1080p',
                'download_video': True
            })
            st.rerun()


def render_progress_section():
    """Render progress tracking section."""
    
    if st.session_state.get('processing', False):
        st.subheader("⏳ Processing Progress")
        
        # Progress bar
        progress = st.session_state.get('progress', 0)
        st.progress(progress)
        
        # Current step
        current_step = st.session_state.get('current_step', 'Processing...')
        st.text(current_step)
        
        # Processing info
        with st.expander("Processing Details"):
            st.json({
                "Progress": f"{progress}%",
                "Current Step": current_step,
                "Start Time": st.session_state.get('process_start_time', 'Unknown'),
                "Video Quality": st.session_state.get('video_quality', 'Unknown'),
                "Clip Duration": f"{st.session_state.get('clip_duration', 'Unknown')}s"
            })
    
    elif st.session_state.get('video_processed', False):
        st.subheader("✅ Processing Complete")
        st.success("Video processed successfully!")
        
        # Processing summary
        with st.expander("Processing Summary"):
            processing_info = {
                "Status": "✅ Complete",
                "Video Title": st.session_state.get('video_info', {}).get('title', 'Unknown'),
                "Duration": f"{st.session_state.get('video_info', {}).get('duration', 'Unknown')}s",
                "Quality": st.session_state.get('video_quality', 'Unknown'),
                "Clips to Generate": st.session_state.get('number_of_clips', 'Unknown')
            }
            st.json(processing_info)
    
    else:
        st.subheader("🎯 Ready to Process")
        st.info("Configure your settings and provide a video URL to begin processing.")


def render_error_display(error_message: str, details: Optional[str] = None):
    """
    Render error display with user-friendly messages.
    
    Args:
        error_message: Main error message to display
        details: Optional detailed error information
    """
    
    st.error(f"❌ **Error:** {error_message}")
    
    # Show details if available
    if details:
        with st.expander("🔍 Error Details"):
            st.code(details, language="text")
    
    # Common solutions
    with st.expander("💡 Common Solutions"):
        st.markdown("""
        **If you're having issues, try:**
        
        1. **Invalid URL Error:**
           - Check that the YouTube URL is correct
           - Try copying the URL directly from YouTube
           - Make sure the video is public and available
        
        2. **Processing Errors:**
           - Try with a shorter video
           - Check your internet connection
           - Reduce video quality in settings
        
        3. **Upload Errors:**
           - Ensure file is in supported format (MP4, MOV, AVI, MKV, WebM)
           - Check file size (recommended < 100MB)
           - Try a different video file
        
        4. **Performance Issues:**
           - Close other browser tabs
           - Reduce video quality
           - Disable parallel processing
        """)
    
    # Clear error button
    if st.button("🔄 Clear Error", key="clear_error"):
        if 'error' in st.session_state:
            del st.session_state['error']
        st.rerun()


def render_video_info_display(video_info: Dict[str, Any]):
    """
    Render video information display.
    
    Args:
        video_info: Dictionary containing video metadata
    """
    
    st.subheader("📋 Video Information")
    
    # Main video info
    info_data = {
        "Title": video_info.get('title', 'Unknown'),
        "Duration": f"{video_info.get('duration', 'Unknown')} seconds" if video_info.get('duration') else 'Unknown',
        "Uploader": video_info.get('uploader', 'Unknown'),
        "Upload Date": video_info.get('upload_date', 'Unknown'),
    }
    
    # Display as metrics for key information
    if video_info.get('duration'):
        duration_min = int(video_info['duration']) // 60
        duration_sec = int(video_info['duration']) % 60
        st.metric("Duration", f"{duration_min}:{duration_sec:02d}")
    
    if video_info.get('view_count'):
        views = format_number(video_info['view_count'])
        st.metric("Views", views)
    
    if video_info.get('like_count'):
        likes = format_number(video_info['like_count'])
        st.metric("Likes", likes)
    
    # Full info in expander
    with st.expander("📊 Detailed Information"):
        st.json(video_info)


def render_thumbnail_display(thumbnail_path: str, video_info: Dict[str, Any]):
    """
    Render thumbnail image display.
    
    Args:
        thumbnail_path: Path to thumbnail image
        video_info: Video metadata for caption
    """
    
    st.subheader("🖼️ Thumbnail")
    
    try:
        # Display thumbnail
        st.image(
            thumbnail_path,
            caption=video_info.get('title', 'Video Thumbnail'),
            use_column_width=True
        )
        
        # Thumbnail info
        thumbnail_file = Path(thumbnail_path)
        if thumbnail_file.exists():
            size_mb = thumbnail_file.stat().st_size / (1024 * 1024)
            st.caption(f"📁 {thumbnail_file.name} ({size_mb:.2f} MB)")
        
    except Exception as e:
        st.error(f"Failed to load thumbnail: {e}")


def format_number(num: int) -> str:
    """
    Format large numbers for display.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted number string
    """
    
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def render_feature_preview(feature_name: str, description: str, coming_in_task: str):
    """
    Render a preview for upcoming features.
    
    Args:
        feature_name: Name of the feature
        description: Description of what the feature will do
        coming_in_task: Which task will implement this feature
    """
    
    with st.container():
        st.info(f"🚧 **{feature_name}** - Coming in {coming_in_task}")
        st.markdown(f"*{description}*")


def render_keyboard_shortcuts():
    """Render keyboard shortcuts help."""
    
    shortcuts = {
        "Ctrl + R": "Refresh page",
        "Ctrl + Shift + R": "Hard refresh (clear cache)",
        "Ctrl + Enter": "Process video (when URL is valid)",
        "Escape": "Cancel current operation",
        "F11": "Toggle fullscreen",
    }
    
    st.subheader("⌨️ Keyboard Shortcuts")
    
    for shortcut, description in shortcuts.items():
        st.markdown(f"- **{shortcut}**: {description}")


def render_tips_and_tricks():
    """Render tips and tricks for better usage."""
    
    st.subheader("💡 Tips & Tricks")
    
    tips = [
        "Use shorter videos (< 10 minutes) for faster processing",
        "720p quality offers the best balance of quality and speed",
        "Enable parallel processing for multiple clips if you have a powerful computer",
        "Auto content type detection works well for most videos",
        "Download the full video only if you need to extract multiple clips",
        "Keep clip duration under 90 seconds for better Twitter engagement"
    ]
    
    for i, tip in enumerate(tips, 1):
        st.markdown(f"{i}. {tip}")


def create_download_button(file_path: str, file_name: str, mime_type: str = "application/octet-stream"):
    """
    Create a download button for files.
    
    Args:
        file_path: Path to the file to download
        file_name: Name for the downloaded file
        mime_type: MIME type of the file
    """
    
    try:
        with open(file_path, "rb") as file:
            file_data = file.read()
        
        st.download_button(
            label=f"📥 Download {file_name}",
            data=file_data,
            file_name=file_name,
            mime=mime_type,
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Failed to prepare download: {e}")


def render_performance_metrics():
    """Render performance metrics for the current session."""
    
    if not st.session_state.get('video_processed', False):
        return
    
    st.subheader("📊 Performance Metrics")
    
    # Mock performance data (will be real in later tasks)
    metrics = {
        "Processing Time": "3.2 seconds",
        "Video Info Extraction": "1.1 seconds", 
        "Thumbnail Extraction": "2.1 seconds",
        "Memory Usage": "45 MB",
        "Success Rate": "100%"
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (metric, value) in enumerate(metrics.items()):
        col = [col1, col2, col3][i % 3]
        with col:
            st.metric(metric, value)


# Custom CSS for better styling
def inject_custom_css():
    """Inject custom CSS for better UI styling."""
    
    st.markdown("""
    <style>
    .stProgress .st-bo {
        background-color: #f0f2f6;
    }
    
    .stProgress .st-bp {
        background-color: #1f77b4;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    
    .success-message {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
