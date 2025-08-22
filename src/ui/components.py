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
from datetime import datetime

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
        'extraction_results': [],
        'optimization_results': [],
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
                # with st.expander("🔍 URL Information"):
                #     st.json({
                #         "Video ID": video_id,
                #         "Normalized URL": normalized_url,
                #         "URL Format": "Valid YouTube format"
                #     })
                
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
    Render thumbnail image display with enhanced gallery functionality.
    
    Args:
        thumbnail_path: Path to thumbnail image
        video_info: Video metadata for caption
    """
    
    # st.subheader("🖼️ Thumbnail Gallery")
    
    try:
        # Check if thumbnail exists
        if not Path(thumbnail_path).exists():
            st.warning("Thumbnail not found. Generating placeholder...")
            # Create a placeholder thumbnail
            create_placeholder_thumbnail()
            return
        
        # Display main thumbnail
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(
                thumbnail_path,
                caption=video_info.get('title', 'Video Thumbnail'),
                use_container_width=True
            )
        
        with col2:
            # Thumbnail info and controls
            thumbnail_file = Path(thumbnail_path)
            size_mb = thumbnail_file.stat().st_size / (1024 * 1024)
            
            st.write("**Thumbnail Details:**")
            st.write(f"📁 **File:** {thumbnail_file.name}")
            st.write(f"📏 **Size:** {size_mb:.2f} MB")
            st.write(f"📅 **Created:** {thumbnail_file.stat().st_mtime}")
            
            # Thumbnail actions
            st.write("**Actions:**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📥 Download", key="download_thumb", use_container_width=True):
                    create_download_button(thumbnail_path, thumbnail_file.name, "image/jpeg")
            
            with col_b:
                if st.button("🔄 Refresh", key="refresh_thumb", use_container_width=True):
                    st.rerun()
        
        # Check for additional thumbnails in the same directory
        # thumbnail_dir = Path(thumbnail_path).parent
        # if thumbnail_dir.exists():
        #     thumbnail_files = list(thumbnail_dir.glob("*.jpg")) + list(thumbnail_dir.glob("*.png"))
            
        #     if len(thumbnail_files) > 1:
        #         st.subheader("📚 Additional Thumbnails")
                
        #         # Create thumbnail grid
        #         cols = st.columns(min(4, len(thumbnail_files)))
        #         for i, thumb_file in enumerate(thumbnail_files):
        #             with cols[i % len(cols)]:
        #                 try:
        #                     st.image(
        #                         str(thumb_file),
        #                         caption=thumb_file.stem,
        #                         use_column_width=True
        #                     )
                            
        #                     # Quick actions for each thumbnail
        #                     if st.button(f"📥 {thumb_file.stem}", key=f"download_thumb_{i}", use_container_width=True):
        #                         create_download_button(str(thumb_file), thumb_file.name, "image/jpeg")
        #                 except Exception as e:
        #                     st.error(f"Failed to load {thumb_file.name}")
        
    except Exception as e:
        st.error(f"Failed to load thumbnail: {e}")
        st.info("🖼️ Thumbnail display would appear here")


def create_placeholder_thumbnail():
    """Create a placeholder thumbnail when none exists."""
    
    st.markdown("""
    <div style="
        width: 100%;
        height: 200px;
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        color: #6c757d;
        margin: 10px 0;
    ">
        🖼️ Thumbnail Placeholder<br>
        <small>No thumbnail available</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 Thumbnails are automatically generated when processing YouTube videos.")


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


def render_video_preview(video_path: str, width: int = 400) -> None:
    """
    Render video preview player with custom controls.
    
    Args:
        video_path: Path to the video file
        width: Width of the video player
    """
    if video_path and Path(video_path).exists():
        try:
            # Get video file info
            file_size = Path(video_path).stat().st_size / (1024 * 1024)  # MB
            file_name = Path(video_path).name
            
            # Video info header
            st.write(f"**📹 {file_name}** ({file_size:.1f} MB)")
            
            # Enhanced video player with custom controls
            st.video(video_path, start_time=0, format="video/mp4")
            
            # Video controls and info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Restart Video", key=f"restart_{file_name}", use_container_width=True):
                    st.rerun()
            
            with col2:
                if st.button("📥 Download", key=f"download_{file_name}", use_container_width=True):
                    create_download_button(video_path, file_name, "video/mp4")
            
            with col3:
                if st.button("🔍 Video Info", key=f"info_{file_name}", use_container_width=True):
                    # Show video metadata
                    try:
                        import subprocess
                        result = subprocess.run([
                            'ffprobe', '-v', 'quiet', '-print_format', 'json',
                            '-show_format', '-show_streams', video_path
                        ], capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            import json
                            info = json.loads(result.stdout)
                            
                            with st.expander("Video Information", expanded=True):
                                if 'format' in info:
                                    format_info = info['format']
                                    st.write(f"**Duration:** {float(format_info.get('duration', 0)):.1f}s")
                                    st.write(f"**Size:** {float(format_info.get('size', 0)) / (1024*1024):.1f} MB")
                                    st.write(f"**Format:** {format_info.get('format_name', 'Unknown')}")
                                
                                if 'streams' in info:
                                    video_streams = [s for s in info['streams'] if s['codec_type'] == 'video']
                                    if video_streams:
                                        video = video_streams[0]
                                        st.write(f"**Resolution:** {video.get('width', 'Unknown')}x{video.get('height', 'Unknown')}")
                                        st.write(f"**Codec:** {video.get('codec_name', 'Unknown')}")
                                        st.write(f"**Bitrate:** {int(video.get('bit_rate', 0)) / 1000:.0f} kbps")
                    except Exception as e:
                        st.error(f"Could not read video info: {e}")
                        
        except Exception as e:
            st.error(f"Failed to load video: {str(e)}")
            st.info("📹 Video preview would appear here when video file is available")
    else:
        # Show placeholder when video file is not available
        st.info("📹 Video preview would appear here")
        
        # Add a demo placeholder box
        st.markdown(f"""
        <div style="
            width: {width}px;
            height: {int(width * 9/16)}px;
            background: linear-gradient(45deg, #f0f0f0, #e0e0e0);
            border: 2px dashed #ccc;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            color: #666;
            margin: 10px 0;
        ">
            🎥 Video Preview<br>
            <small style="font-size: 12px;">({width}x{int(width * 9/16)})</small>
        </div>
        """, unsafe_allow_html=True)


def render_clip_results_gallery(
    extraction_results: Optional[Any] = None,
    optimization_results: Optional[List[Any]] = None
) -> None:
    """
    Render a gallery of extracted and optimized clips with thumbnails and download buttons.
    
    Args:
        extraction_results: Results from clip extraction
        optimization_results: Results from Twitter optimization
    """
    if not extraction_results or not extraction_results.results:
        st.info("🎬 No clips extracted yet. Process a video to see results here.")
        return
    
    st.subheader("🎬 Extracted Clips Gallery")
    
    # Create columns for clip display
    clips = extraction_results.results
    optimized_clips = optimization_results or []
    
    # Add gallery controls
    col_controls1, col_controls2, col_controls3 = st.columns([1, 1, 1])
    
    with col_controls1:
        view_mode = st.selectbox(
            "View Mode",
            ["Grid", "List", "Comparison"],
            key="gallery_view_mode"
        )
    
    with col_controls2:
        sort_by = st.selectbox(
            "Sort By",
            ["Time", "Duration", "Size", "Quality"],
            key="gallery_sort_by"
        )
    
    with col_controls3:
        if st.button("🔄 Refresh Gallery", use_container_width=True):
            st.rerun()
    
    st.divider()
    
    if view_mode == "Grid":
        render_clip_grid_view(clips, optimized_clips)
    elif view_mode == "List":
        render_clip_list_view(clips, optimized_clips)
    else:
        render_clip_comparison_view(clips, optimized_clips)


def render_clip_grid_view(clips, optimized_clips):
    """Render clips in a grid layout."""
    
    # Calculate grid columns based on number of clips
    num_clips = len([c for c in clips if c.success])
    cols_per_row = min(3, max(1, num_clips))
    
    for i in range(0, num_clips, cols_per_row):
        row_clips = clips[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        
        for j, clip_result in enumerate(row_clips):
            if not clip_result.success:
                continue
                
            with cols[j]:
                render_clip_card(clip_result, optimized_clips, i + j)


def render_clip_list_view(clips, optimized_clips):
    """Render clips in a list layout."""
    
    for i, clip_result in enumerate(clips):
        if not clip_result.success:
            continue
            
        render_clip_expanded(clip_result, optimized_clips, i)


def render_clip_card(clip_result, optimized_clips, index):
    """Render a single clip as a card."""
    
    with st.container():
        st.markdown(f"""
        <div style="
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 10px;
            margin: 5px 0;
            background: white;
        ">
        """, unsafe_allow_html=True)
        
        # Clip header
        st.write(f"**🎬 Clip {index + 1}**")
        st.write(f"⏱️ {clip_result.start_time} - {clip_result.end_time}")
        
        # Video preview (smaller for grid)
        video_path = Path(clip_result.clip_path)
        if video_path.exists():
            st.video(str(video_path.resolve()), start_time=0)
        else:
            # Try alternative paths
            alt_paths = [
                Path("outputs/clips") / video_path.name,
                Path("outputs/optimized") / video_path.name,
                video_path.name  # Just the filename
            ]
            
            video_found = False
            for alt_path in alt_paths:
                if alt_path.exists():
                    st.video(str(alt_path.resolve()), start_time=0)
                    video_found = True
                    break
            
            if not video_found:
                st.warning(f"⚠️ Video not found: {clip_result.clip_path}")
                st.caption("The video file may have been moved or deleted.")
        
        # Quick metadata
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"📏 {clip_result.duration_seconds:.1f}s")
        with col2:
            st.write(f"💾 {clip_result.file_size_mb:.1f} MB")
        
        # Quick actions
        if st.button(f"📥 Download", key=f"quick_download_{index}", use_container_width=True):
            create_download_button(
                clip_result.clip_path,
                f"clip_{index + 1}.mp4",
                "video/mp4"
            )
        
        # Show optimization status
        optimized_result = optimized_clips[index] if index < len(optimized_clips) and optimized_clips[index].success else None
        if optimized_result:
            st.success("✅ Twitter Optimized")
            if st.button(f"🐦 Download Optimized", key=f"quick_optimized_{index}", use_container_width=True):
                create_download_button(
                    optimized_result.optimized_path,
                    f"clip_{index + 1}_twitter.mp4",
                    "video/mp4"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)


def render_clip_expanded(clip_result, optimized_clips, index):
    """Render a single clip in expanded view."""
    
    # Create expandable section for each clip
    with st.expander(f"🎬 Clip {index+1}: {clip_result.start_time} - {clip_result.end_time}", expanded=True):
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Video preview
            st.write("**Original Clip**")
            video_path = Path(clip_result.clip_path)
            if video_path.exists():
                # Enhanced video preview with controls
                render_video_preview(str(video_path.resolve()), width=300)
            else:
                # Try alternative paths
                alt_paths = [
                    Path("outputs/clips") / video_path.name,
                    Path("outputs/optimized") / video_path.name,
                    video_path.name  # Just the filename
                ]
                
                video_found = False
                for alt_path in alt_paths:
                    if alt_path.exists():
                        render_video_preview(str(alt_path.resolve()), width=300)
                        video_found = True
                        break
                
                if not video_found:
                    # Show placeholder for missing video
                    st.markdown("""
                <div style="
                    width: 100%;
                    height: 200px;
                    background: linear-gradient(45deg, #f8f9fa, #e9ecef);
                    border: 2px dashed #dee2e6;
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 16px;
                    color: #6c757d;
                    margin: 10px 0;
                ">
                    🎬 Original Clip Preview<br>
                    <small>Video would play here</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Clip metadata
            st.write(f"**Duration:** {clip_result.duration_seconds:.1f}s")
            st.write(f"**Size:** {clip_result.file_size_mb:.1f} MB")
            
            # Generate thumbnail if video exists
            if Path(clip_result.clip_path).exists():
                thumbnail_path = generate_clip_thumbnail(clip_result.clip_path, index)
                if thumbnail_path:
                    st.write("**Thumbnail:**")
                    st.image(thumbnail_path, use_container_width=True)
        
        with col2:
            # Optimized version if available
            optimized_result = optimized_clips[index] if index < len(optimized_clips) and optimized_clips[index].success else None
            
            if optimized_result:
                st.write("**Twitter Optimized**")
                if Path(optimized_result.optimized_path).exists():
                    render_video_preview(optimized_result.optimized_path, width=300)
                else:
                    # Show placeholder for missing optimized video
                    st.markdown("""
                    <div style="
                        width: 100%;
                        height: 200px;
                        background: linear-gradient(45deg, #e3f2fd, #bbdefb);
                        border: 2px dashed #2196f3;
                        border-radius: 8px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 16px;
                        color: #1976d2;
                        margin: 10px 0;
                    ">
                        🐦 Twitter Optimized<br>
                        <small>Optimized video would play here</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Optimization metrics
                st.write(f"**Size:** {optimized_result.optimized_size_mb:.1f} MB")
                st.write(f"**Compression:** {optimized_result.compression_ratio:.1f}x")
                st.write(f"**Quality Score:** {optimized_result.quality_score:.0f}/100")
                
                # Twitter compatibility indicator
                if optimized_result.twitter_compatible:
                    st.success("✅ Twitter Compatible")
                else:
                    st.warning("⚠️ May not be Twitter compatible")
            else:
                st.info("Twitter optimization not available")
        
        with col3:
            # Download buttons
            st.write("**Downloads**")
            
            # Original clip download
            if Path(clip_result.clip_path).exists():
                create_download_button(
                    clip_result.clip_path,
                    f"clip_{index+1}_original.mp4",
                    "video/mp4"
                )
            
            # Optimized clip download
            if optimized_result and Path(optimized_result.optimized_path).exists():
                create_download_button(
                    optimized_result.optimized_path,
                    f"clip_{index+1}_twitter.mp4",
                    "video/mp4"
                )
            
            # Additional actions
            st.write("**Actions:**")
            
            if st.button("🔄 Re-process", key=f"reprocess_{index}", use_container_width=True):
                st.info("Feature coming soon: Re-process clip with different settings")
            
            if st.button("📊 Analytics", key=f"analytics_{index}", use_container_width=True):
                st.info("Feature coming soon: Detailed clip analytics")


def generate_clip_thumbnail(video_path: str, index: int) -> Optional[str]:
    """Generate a thumbnail for a video clip."""
    
    try:
        import subprocess
        from pathlib import Path
        
        # Create thumbnails directory if it doesn't exist
        thumb_dir = Path("cache/thumbnails")
        thumb_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate thumbnail filename
        thumb_path = thumb_dir / f"clip_{index + 1}_thumb.jpg"
        
        # Use ffmpeg to extract thumbnail at 1 second mark
        result = subprocess.run([
            'ffmpeg', '-i', video_path, '-ss', '00:00:01', '-vframes', '1',
            '-q:v', '2', str(thumb_path), '-y'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and thumb_path.exists():
            return str(thumb_path)
        else:
            return None
            
    except Exception as e:
        # Silently fail - thumbnails are optional
        return None


def render_llm_reasoning_display(analysis_results: Optional[Dict[str, Any]] = None) -> None:
    """
    Display LLM reasoning and analysis results with enhanced visualization.
    
    Args:
        analysis_results: Results from LLM content analysis
    """
    if not analysis_results:
        st.info("🤖 No content analysis available yet.")
        return
    
    st.subheader("🤖 AI Content Analysis & Strategy")
    
    # Overall analysis summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'content_type' in analysis_results:
            content_type = analysis_results['content_type'].title()
            st.metric("Content Type", content_type)
            
            # Content type icon
            type_icons = {
                'educational': '📚',
                'entertainment': '🎭',
                'interview': '🎤',
                'tutorial': '🔧',
                'news': '📰',
                'podcast': '🎧',
                'music': '🎵',
                'gaming': '🎮'
            }
            icon = type_icons.get(analysis_results['content_type'].lower(), '📹')
            st.write(f"{icon} {content_type}")
    
    with col2:
        if 'strategy' in analysis_results:
            strategy = analysis_results['strategy'].replace('_', ' ').title()
            st.metric("Recommended Strategy", strategy)
            
            # Strategy description
            strategy_descriptions = {
                'thought_leadership': 'Share insights and expertise to establish authority',
                'viral_content': 'Create highly shareable, engaging content',
                'educational': 'Focus on teaching and knowledge sharing',
                'entertainment': 'Prioritize fun and engaging moments',
                'conversation_starter': 'Ask questions to encourage discussion',
                'trending_topics': 'Connect to current events and trends'
            }
            desc = strategy_descriptions.get(analysis_results['strategy'], 'Optimized for engagement')
            st.caption(desc)
    
    with col3:
        if 'recommendations' in analysis_results:
            total_clips = len(analysis_results['recommendations'])
            avg_confidence = sum(r.get('confidence', 0) for r in analysis_results['recommendations']) / total_clips if total_clips > 0 else 0
            st.metric("Total Clips", total_clips)
            st.metric("Avg Confidence", f"{avg_confidence:.0f}%")
    
    st.divider()
    
    # Clip recommendations with enhanced reasoning
    if 'recommendations' in analysis_results:
        st.subheader("🎯 Clip Recommendations & Reasoning")
        
        # Add recommendation filters
        col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])
        
        with col_filter1:
            min_confidence = st.slider(
                "Min Confidence %",
                min_value=0,
                max_value=100,
                value=50,
                key="confidence_filter"
            )
        
        with col_filter2:
            hook_filter = st.selectbox(
                "Hook Strength",
                ["All", "High", "Medium", "Low"],
                key="hook_filter"
            )
        
        with col_filter3:
            if st.button("🔍 Apply Filters", use_container_width=True):
                st.rerun()
        
        # Filter recommendations
        filtered_recommendations = []
        for rec in analysis_results['recommendations']:
            if rec.get('confidence', 0) >= min_confidence:
                if hook_filter == "All" or rec.get('hook_strength', '').lower() == hook_filter.lower():
                    filtered_recommendations.append(rec)
        
        if not filtered_recommendations:
            st.warning(f"No recommendations match the current filters. Try adjusting the criteria.")
            return
        
        # Display filtered recommendations
        for i, rec in enumerate(filtered_recommendations):
            with st.expander(f"🎬 Recommendation {i+1}: {rec.get('start_time', 'N/A')} - {rec.get('end_time', 'N/A')}", expanded=True):
                
                # Recommendation header with metrics
                col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
                
                with col_metrics1:
                    # Confidence indicator with color coding
                    confidence = rec.get('confidence', 0)
                    if confidence >= 80:
                        st.success(f"**Confidence:** {confidence}%")
                        st.write("🔥 **High Confidence**")
                    elif confidence >= 60:
                        st.warning(f"**Confidence:** {confidence}%")
                        st.write("⚡ **Medium Confidence**")
                    else:
                        st.error(f"**Confidence:** {confidence}%")
                        st.write("💡 **Low Confidence**")
                
                with col_metrics2:
                    # Hook strength with emoji
                    hook_strength = rec.get('hook_strength', 'unknown')
                    hook_emoji = {"high": "🔥", "medium": "⚡", "low": "💡"}.get(hook_strength, "❓")
                    st.write(f"**Hook Strength:**")
                    st.write(f"{hook_emoji} {hook_strength.title()}")
                    
                    # Hook strength explanation
                    hook_explanations = {
                        'high': 'Strong opening that grabs attention',
                        'medium': 'Good content with moderate engagement potential',
                        'low': 'Basic content, may need enhancement'
                    }
                    st.caption(hook_explanations.get(hook_strength, 'Unknown strength level'))
                
                with col_metrics3:
                    # Duration and timing
                    start_time = rec.get('start_time', '00:00:00')
                    end_time = rec.get('end_time', '00:00:00')
                    st.write(f"**Timing:**")
                    st.write(f"⏱️ {start_time} - {end_time}")
                    
                    # Calculate duration if possible
                    try:
                        from datetime import datetime
                        start_dt = datetime.strptime(start_time, '%H:%M:%S')
                        end_dt = datetime.strptime(end_time, '%H:%M:%S')
                        duration = (end_dt - start_dt).total_seconds()
                        st.write(f"📏 {duration:.0f}s")
                    except:
                        st.write("📏 Duration unknown")
                
                with col_metrics4:
                    # Keywords and topics
                    if 'keywords' in rec and rec['keywords']:
                        st.write(f"**Key Topics:**")
                        for keyword in rec['keywords'][:3]:  # Show first 3 keywords
                            st.write(f"🏷️ {keyword}")
                        if len(rec['keywords']) > 3:
                            st.caption(f"+{len(rec['keywords']) - 3} more topics")
                    else:
                        st.write("**Key Topics:**")
                        st.write("📝 No topics identified")
                
                # Detailed reasoning section
                st.markdown("---")
                st.write("**🤖 AI Reasoning & Analysis:**")
                
                if 'reasoning' in rec:
                    # Enhanced reasoning display
                    reasoning = rec['reasoning']
                    
                    # Split reasoning into paragraphs for better readability
                    paragraphs = reasoning.split('. ')
                    for para in paragraphs:
                        if para.strip():
                            st.write(f"💭 {para.strip()}")
                    
                    # Add reasoning insights
                    st.info("💡 **Why this segment works:** This clip has been identified as having high potential for Twitter engagement based on the content analysis.")
                
                # Content strategy recommendations
                st.markdown("---")
                st.write("**📋 Content Strategy Recommendations:**")
                
                col_strategy1, col_strategy2 = st.columns(2)
                
                with col_strategy1:
                    st.write("**🎯 Best Use Cases:**")
                    use_cases = []
                    
                    if confidence >= 80:
                        use_cases.append("Primary content for high engagement")
                        use_cases.append("Featured clip in thread")
                    elif confidence >= 60:
                        use_cases.append("Secondary content piece")
                        use_cases.append("Supporting material")
                    else:
                        use_cases.append("Background content")
                        use_cases.append("Reference material")
                    
                    for use_case in use_cases:
                        st.write(f"✅ {use_case}")
                
                with col_strategy2:
                    st.write("**🚀 Optimization Tips:**")
                    
                    if hook_strength == 'high':
                        st.write("🔥 Lead with this clip")
                        st.write("📱 Perfect for mobile viewing")
                    elif hook_strength == 'medium':
                        st.write("⚡ Add engaging caption")
                        st.write("🏷️ Use relevant hashtags")
                    else:
                        st.write("💡 Consider enhancing intro")
                        st.write("📝 Add context in caption")
                
                # Action buttons
                st.markdown("---")
                col_action1, col_action2, col_action3 = st.columns(3)
                
                with col_action1:
                    if st.button(f"✂️ Extract This Clip", key=f"extract_clip_{i}", use_container_width=True):
                        extract_single_clip_from_recommendation(rec, i)
                
                with col_action2:
                    if st.button(f"🔍 Detailed Analysis", key=f"detailed_analysis_{i}", use_container_width=True):
                        st.info("📊 Detailed analysis feature coming soon: Deep dive into content metrics, sentiment analysis, and engagement predictions.")
                
                with col_action3:
                    if st.button(f"📝 Generate Caption", key=f"generate_caption_{i}", use_container_width=True):
                        st.info("✍️ Caption generation feature coming soon: AI-powered Twitter captions optimized for engagement.")
    
    # Additional analysis tools
    st.divider()
    st.subheader("🔧 Advanced Analysis Tools")
    
    col_tools1, col_tools2, col_tools3 = st.columns(3)
    
    with col_tools1:
        if st.button("🔄 Re-analyze Content", use_container_width=True):
            st.info("🔄 Re-analysis feature coming soon: Re-run AI analysis with different parameters and strategies.")
    
    with col_tools2:
        if st.button("📊 Sentiment Analysis", use_container_width=True):
            st.info("😊 Sentiment analysis feature coming soon: Deep emotional tone analysis and mood detection.")
    
    with col_tools3:
        if st.button("📈 Engagement Prediction", use_container_width=True):
            st.info("📈 Engagement prediction feature coming soon: AI-powered engagement forecasting based on content analysis.")


def render_processing_progress(current_step: str, progress: int, status_text: str = "") -> None:
    """
    Render enhanced progress tracking with step indicators.
    
    Args:
        current_step: Current processing step
        progress: Progress percentage (0-100)
        status_text: Additional status information
    """
    st.subheader("⚙️ Processing Status")
    
    # Processing steps
    steps = [
        ("download", "📥 Download Video"),
        ("transcribe", "🎙️ Transcribe Audio"),
        ("analyze", "🤖 AI Analysis"),
        ("extract", "✂️ Extract Clips"),
        ("optimize", "🐦 Twitter Optimization"),
        ("complete", "✅ Complete")
    ]
    
    # Create step indicators
    cols = st.columns(len(steps))
    for i, (step_key, step_name) in enumerate(steps):
        with cols[i]:
            if current_step == step_key:
                st.success(f"**{step_name}**")
                st.progress(progress / 100.0)
            elif steps.index((current_step, next(name for key, name in steps if key == current_step))) > i:
                st.success(f"✅ {step_name}")
            else:
                st.info(f"⏳ {step_name}")
    
    # Overall progress bar
    st.progress(progress / 100.0)
    
    if status_text:
        st.write(f"**Status:** {status_text}")


def render_clip_comparison_view(clips_data: List[Dict[str, Any]]) -> None:
    """
    Render enhanced side-by-side comparison of clips with advanced controls.
    
    Args:
        clips_data: List of clip data dictionaries
    """
    if not clips_data:
        return
    
    st.subheader("🔍 Enhanced Clip Comparison")
    
    # Comparison controls
    col_control1, col_control2, col_control3 = st.columns([1, 1, 1])
    
    with col_control1:
        comparison_mode = st.selectbox(
            "Comparison Mode",
            ["Side by Side", "Grid View", "Timeline View"],
            key="comparison_mode"
        )
    
    with col_control2:
        sort_comparison = st.selectbox(
            "Sort By",
            ["Time", "Duration", "Size", "Quality"],
            key="sort_comparison"
        )
    
    with col_control3:
        if st.button("🔄 Refresh Comparison", use_container_width=True):
            st.rerun()
    
    st.divider()
    
    if comparison_mode == "Side by Side":
        render_side_by_side_comparison(clips_data)
    elif comparison_mode == "Grid View":
        render_grid_comparison(clips_data)
    else:
        render_timeline_comparison(clips_data)


def render_side_by_side_comparison(clips_data: List[Dict[str, Any]]) -> None:
    """Render side-by-side comparison of two selected clips."""
    
    if len(clips_data) < 2:
        st.warning("Need at least 2 clips for comparison.")
        return
    
    # Allow user to select clips to compare
    clip_options = [f"Clip {i+1}: {clip.get('start_time', 'N/A')} - {clip.get('end_time', 'N/A')}" 
                   for i, clip in enumerate(clips_data)]
    
    col_select1, col_select2 = st.columns(2)
    
    with col_select1:
        selected_clip_1 = st.selectbox("Select first clip:", clip_options, key="compare_clip_1")
        clip_1_idx = clip_options.index(selected_clip_1)
        clip_1_data = clips_data[clip_1_idx]
    
    with col_select2:
        selected_clip_2 = st.selectbox("Select second clip:", clip_options, key="compare_clip_2")
        clip_2_idx = clip_options.index(selected_clip_2)
        clip_2_data = clips_data[clip_2_idx]
    
    # Ensure different clips are selected
    if clip_1_idx == clip_2_idx:
        st.warning("Please select different clips for comparison.")
        return
    
    st.divider()
    
    # Side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🎬 {selected_clip_1}")
        
        # Video preview
        if 'video_path' in clip_1_data and Path(clip_1_data['video_path']).exists():
            st.video(clip_1_data['video_path'])
        else:
            st.info("Video not available")
        
        # Clip metrics
        render_clip_metrics(clip_1_data, "Clip 1")
        
        # Download button
        if 'video_path' in clip_1_data and Path(clip_1_data['video_path']).exists():
            create_download_button(
                clip_1_data['video_path'],
                f"clip_{clip_1_idx + 1}_comparison.mp4",
                "video/mp4"
            )
    
    with col2:
        st.subheader(f"🎬 {selected_clip_2}")
        
        # Video preview
        if 'video_path' in clip_2_data and Path(clip_2_data['video_path']).exists():
            st.video(clip_2_data['video_path'])
        else:
            st.info("Video not available")
        
        # Clip metrics
        render_clip_metrics(clip_2_data, "Clip 2")
        
        # Download button
        if 'video_path' in clip_2_data and Path(clip_2_data['video_path']).exists():
            create_download_button(
                clip_2_data['video_path'],
                f"clip_{clip_2_idx + 1}_comparison.mp4",
                "video/mp4"
            )
    
    # Comparison analysis
    st.divider()
    render_comparison_analysis(clip_1_data, clip_2_data)


def render_grid_comparison(clips_data: List[Dict[str, Any]]) -> None:
    """Render grid comparison of multiple clips."""
    
    # Calculate grid layout
    num_clips = len(clips_data)
    cols_per_row = min(4, max(2, num_clips))
    
    st.write(f"**Grid Comparison View ({num_clips} clips)**")
    
    for i in range(0, num_clips, cols_per_row):
        row_clips = clips_data[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        
        for j, clip_data in enumerate(row_clips):
            with cols[j]:
                render_clip_comparison_card(clip_data, i + j)


def render_timeline_comparison(clips_data: List[Dict[str, Any]]) -> None:
    """Render timeline-based comparison of clips."""
    
    st.write("**Timeline Comparison View**")
    
    # Sort clips by start time if possible
    try:
        sorted_clips = sorted(clips_data, key=lambda x: x.get('start_time', '00:00:00'))
    except:
        sorted_clips = clips_data
    
    # Create timeline visualization
    for i, clip_data in enumerate(sorted_clips):
        with st.container():
            col_timeline, col_content = st.columns([1, 4])
            
            with col_timeline:
                st.write(f"**{i+1}**")
                st.write(f"⏱️ {clip_data.get('start_time', 'N/A')}")
                st.write(f"📏 {clip_data.get('duration', 'N/A')}")
            
            with col_content:
                # Clip preview (smaller for timeline)
                if 'video_path' in clip_data and Path(clip_data['video_path']).exists():
                    st.video(clip_data['video_path'], start_time=0)
                else:
                    st.info("Video not available")
                
                # Quick metrics
                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                with col_metrics1:
                    st.write(f"💾 {clip_data.get('size_mb', 'N/A')} MB")
                with col_metrics2:
                    st.write(f"🎯 {clip_data.get('quality', 'N/A')}")
                with col_metrics3:
                    if st.button(f"📥 Download", key=f"timeline_download_{i}", use_container_width=True):
                        create_download_button(
                            clip_data['video_path'],
                            f"clip_{i+1}_timeline.mp4",
                            "video/mp4"
                        )
        
        if i < len(sorted_clips) - 1:
            st.divider()


def render_clip_comparison_card(clip_data: Dict[str, Any], index: int) -> None:
    """Render a single clip card for comparison."""
    
    with st.container():
        st.markdown(f"""
        <div style="
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 10px;
            margin: 5px 0;
            background: white;
        ">
        """, unsafe_allow_html=True)
        
        # Clip header
        st.write(f"**🎬 Clip {index + 1}**")
        st.write(f"⏱️ {clip_data.get('start_time', 'N/A')} - {clip_data.get('end_time', 'N/A')}")
        
        # Video preview (smaller for grid)
        if 'video_path' in clip_data and Path(clip_data['video_path']).exists():
            st.video(clip_data['video_path'], start_time=0)
        else:
            st.info("Video not available")
        
        # Quick metrics
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"📏 {clip_data.get('duration', 'N/A')}")
        with col2:
            st.write(f"💾 {clip_data.get('size_mb', 'N/A')} MB")
        
        # Quick actions
        if st.button(f"📥 Download", key=f"grid_download_{index}", use_container_width=True):
            create_download_button(
                clip_data['video_path'],
                f"clip_{index + 1}.mp4",
                "video/mp4"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)


def render_clip_metrics(clip_data: Dict[str, Any], clip_label: str) -> None:
    """Render detailed metrics for a clip."""
    
    st.write(f"**{clip_label} Metrics:**")
    
    # Basic metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"⏱️ **Duration:** {clip_data.get('duration', 'N/A')}")
        st.write(f"💾 **Size:** {clip_data.get('size_mb', 'N/A')} MB")
    
    with col2:
        st.write(f"🎯 **Quality:** {clip_data.get('quality', 'N/A')}")
        st.write(f"📅 **Created:** {clip_data.get('created', 'N/A')}")
    
    # Additional metrics if available
    if 'bitrate' in clip_data:
        st.write(f"📊 **Bitrate:** {clip_data['bitrate']} kbps")
    
    if 'resolution' in clip_data:
        st.write(f"🖼️ **Resolution:** {clip_data['resolution']}")
    
    if 'fps' in clip_data:
        st.write(f"🎬 **FPS:** {clip_data['fps']}")


def render_comparison_analysis(clip_1_data: Dict[str, Any], clip_2_data: Dict[str, Any]) -> None:
    """Render analysis comparing two clips."""
    
    st.subheader("📊 Comparison Analysis")
    
    col_analysis1, col_analysis2 = st.columns(2)
    
    with col_analysis1:
        st.write("**📈 Performance Comparison:**")
        
        # Duration comparison
        try:
            duration_1 = float(clip_1_data.get('duration', '0').replace('s', ''))
            duration_2 = float(clip_2_data.get('duration', '0').replace('s', ''))
            
            if duration_1 > duration_2:
                st.write(f"⏱️ Clip 1 is {duration_1 - duration_2:.1f}s longer")
            elif duration_2 > duration_1:
                st.write(f"⏱️ Clip 2 is {duration_2 - duration_1:.1f}s longer")
            else:
                st.write("⏱️ Both clips have the same duration")
        except:
            st.write("⏱️ Duration comparison not available")
        
        # Size comparison
        try:
            size_1 = float(clip_1_data.get('size_mb', 0))
            size_2 = float(clip_2_data.get('size_mb', 0))
            
            if size_1 > size_2:
                st.write(f"💾 Clip 1 is {size_1 - size_2:.1f} MB larger")
            elif size_2 > size_1:
                st.write(f"💾 Clip 2 is {size_2 - size_1:.1f} MB larger")
            else:
                st.write("💾 Both clips have the same size")
        except:
            st.write("💾 Size comparison not available")
    
    with col_analysis2:
        st.write("**🎯 Recommendations:**")
        
        # Content strategy recommendations
        st.write("💡 **Best Use Cases:**")
        
        # Simple recommendations based on metrics
        try:
            size_1 = float(clip_1_data.get('size_mb', 0))
            size_2 = float(clip_2_data.get('size_mb', 0))
            
            if size_1 < size_2:
                st.write("✅ Clip 1 is better for mobile sharing")
            else:
                st.write("✅ Clip 2 is better for mobile sharing")
            
            duration_1 = float(clip_1_data.get('duration', '0').replace('s', ''))
            duration_2 = float(clip_2_data.get('duration', '0').replace('s', ''))
            
            if duration_1 < 60 and duration_2 < 60:
                st.write("✅ Both clips are Twitter-optimized length")
            elif duration_1 < 60:
                st.write("✅ Clip 1 is Twitter-optimized length")
            elif duration_2 < 60:
                st.write("✅ Clip 2 is Twitter-optimized length")
            else:
                st.write("⚠️ Both clips may be too long for Twitter")
                
        except:
            st.write("📝 Analysis not available")
    
    # Action buttons
    st.divider()
    col_action1, col_action2, col_action3 = st.columns(3)
    
    with col_action1:
        if st.button("📊 Detailed Metrics", key="detailed_metrics", use_container_width=True):
            st.info("📊 Detailed metrics feature coming soon: Comprehensive performance analysis and benchmarking.")
    
    with col_action2:
        if st.button("🎬 Side-by-Side Playback", key="side_by_side_playback", use_container_width=True):
            st.info("🎬 Side-by-side playback feature coming soon: Synchronized video comparison with custom controls.")
    
    with col_action3:
        if st.button("📋 Export Comparison", key="export_comparison", use_container_width=True):
            st.info("📋 Export comparison feature coming soon: Save comparison results and analysis as a report.")


def render_batch_download_section(
    extraction_results: Optional[Any] = None,
    optimization_results: Optional[List[Any]] = None
) -> None:
    """
    Render batch download options for all clips with enhanced functionality.
    
    Args:
        extraction_results: Results from clip extraction
        optimization_results: Results from Twitter optimization
    """
    if not extraction_results or not extraction_results.results:
        return
    
    st.subheader("📦 Batch Downloads & Management")
    
    # Download statistics
    total_clips = len([r for r in extraction_results.results if r.success])
    optimized_clips = len([r for r in optimization_results if r.success]) if optimization_results else 0
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("Total Clips", total_clips)
    
    with col_stats2:
        st.metric("Optimized Clips", optimized_clips)
    
    with col_stats3:
        total_size_original = sum(r.file_size_mb for r in extraction_results.results if r.success)
        st.metric("Original Size", f"{total_size_original:.1f} MB")
    
    with col_stats4:
        if optimization_results:
            total_size_optimized = sum(r.optimized_size_mb for r in optimization_results if r.success)
            compression_ratio = total_size_original / total_size_optimized if total_size_optimized > 0 else 1
            st.metric("Optimized Size", f"{total_size_optimized:.1f} MB")
            st.caption(f"Compression: {compression_ratio:.1f}x")
    
    st.divider()
    
    # Batch download options
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        st.write("**📥 Download Options**")
        
        # Original clips batch download
        if st.button("📥 Download All Original Clips", use_container_width=True, key="batch_original"):
            download_all_original_clips(extraction_results)
        
        # Optimized clips batch download
        if optimization_results and optimized_clips > 0:
            if st.button("🐦 Download All Twitter-Ready Clips", use_container_width=True, key="batch_optimized"):
                download_all_optimized_clips(optimization_results)
        
        # Selective download
        if st.button("🎯 Selective Download", use_container_width=True, key="selective_download"):
            st.session_state['show_selective_download'] = True
    
    with col_download2:
        st.write("**📊 Export Options**")
        
        # Export metadata
        if st.button("📋 Export Clip Metadata", use_container_width=True, key="export_metadata"):
            export_clip_metadata(extraction_results, optimization_results)
        
        # Export summary report
        if st.button("📊 Export Summary Report", use_container_width=True, key="export_report"):
            export_summary_report(extraction_results, optimization_results)
    
    # Selective download interface
    if st.session_state.get('show_selective_download', False):
        render_selective_download_interface(extraction_results, optimization_results)
    
    # Download progress tracking
    if st.session_state.get('batch_download_progress', False):
        render_download_progress()


def download_all_original_clips(extraction_results):
    """Download all original clips as a batch."""
    
    try:
        successful_clips = [r for r in extraction_results.results if r.success]
        
        if not successful_clips:
            st.warning("No successful clips to download.")
            return
        
        # Create a zip file with all clips
        import zipfile
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                for i, clip in enumerate(successful_clips):
                    if Path(clip.clip_path).exists():
                        zipf.write(clip.clip_path, f"clip_{i+1}_original.mp4")
            
            # Read the zip file and create download button
            with open(tmp_file.name, 'rb') as f:
                zip_data = f.read()
            
            st.download_button(
                label=f"📥 Download All Original Clips ({len(successful_clips)} files)",
                data=zip_data,
                file_name="all_original_clips.zip",
                mime="application/zip",
                use_container_width=True
            )
            
            # Clean up temp file
            Path(tmp_file.name).unlink()
            
            st.success(f"✅ Ready to download {len(successful_clips)} original clips!")
            
    except Exception as e:
        st.error(f"Failed to prepare batch download: {e}")


def download_all_optimized_clips(optimization_results):
    """Download all optimized clips as a batch."""
    
    try:
        successful_optimized = [r for r in optimization_results if r.success]
        
        if not successful_optimized:
            st.warning("No optimized clips to download.")
            return
        
        # Create a zip file with all optimized clips
        import zipfile
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                for i, clip in enumerate(successful_optimized):
                    if Path(clip.optimized_path).exists():
                        zipf.write(clip.optimized_path, f"clip_{i+1}_twitter_optimized.mp4")
            
            # Read the zip file and create download button
            with open(tmp_file.name, 'rb') as f:
                zip_data = f.read()
            
            st.download_button(
                label=f"🐦 Download All Twitter-Ready Clips ({len(successful_optimized)} files)",
                data=zip_data,
                file_name="all_twitter_optimized_clips.zip",
                mime="application/zip",
                use_container_width=True
            )
            
            # Clean up temp file
            Path(tmp_file.name).unlink()
            
            st.success(f"✅ Ready to download {len(successful_optimized)} Twitter-optimized clips!")
            
    except Exception as e:
        st.error(f"Failed to prepare batch download: {e}")


def render_selective_download_interface(extraction_results, optimization_results):
    """Render interface for selective clip downloads."""
    
    st.subheader("🎯 Selective Download")
    
    successful_clips = [r for r in extraction_results.results if r.success]
    
    # Clip selection checkboxes
    st.write("**Select clips to download:**")
    
    selected_clips = []
    for i, clip in enumerate(successful_clips):
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            is_selected = st.checkbox(f"Clip {i+1}", key=f"select_clip_{i}")
            if is_selected:
                selected_clips.append(i)
        
        with col2:
            st.write(f"{clip.start_time} - {clip.end_time} ({clip.duration_seconds:.1f}s)")
        
        with col3:
            st.write(f"{clip.file_size_mb:.1f} MB")
    
    if selected_clips:
        st.write(f"**Selected {len(selected_clips)} clips**")
        
        col_select1, col_select2 = st.columns(2)
        
        with col_select1:
            if st.button("📥 Download Selected Original", use_container_width=True):
                download_selected_clips(selected_clips, successful_clips, "original")
        
        with col_select2:
            if optimization_results:
                if st.button("🐦 Download Selected Optimized", use_container_width=True):
                    download_selected_clips(selected_clips, optimization_results, "optimized")
    
    # Close selective download
    if st.button("❌ Close Selection", use_container_width=True):
        st.session_state['show_selective_download'] = False
        st.rerun()


def download_selected_clips(selected_indices, clip_results, clip_type):
    """Download selected clips of specified type."""
    
    try:
        selected_clips = [clip_results[i] for i in selected_indices if i < len(clip_results)]
        
        if not selected_clips:
            st.warning("No clips selected for download.")
            return
        
        # Create zip file with selected clips
        import zipfile
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                for i, clip in enumerate(selected_clips):
                    if clip_type == "original" and Path(clip.clip_path).exists():
                        zipf.write(clip.clip_path, f"selected_clip_{i+1}_original.mp4")
                    elif clip_type == "optimized" and Path(clip.optimized_path).exists():
                        zipf.write(clip.optimized_path, f"selected_clip_{i+1}_twitter.mp4")
            
            # Read the zip file and create download button
            with open(tmp_file.name, 'rb') as f:
                zip_data = f.read()
            
            file_name = f"selected_{clip_type}_clips.zip"
            label = f"📥 Download Selected {clip_type.title()} Clips ({len(selected_clips)} files)"
            
            st.download_button(
                label=label,
                data=zip_data,
                file_name=file_name,
                mime="application/zip",
                use_container_width=True
            )
            
            # Clean up temp file
            Path(tmp_file.name).unlink()
            
            st.success(f"✅ Ready to download {len(selected_clips)} selected {clip_type} clips!")
            
    except Exception as e:
        st.error(f"Failed to prepare selective download: {e}")


def export_clip_metadata(extraction_results, optimization_results):
    """Export clip metadata as JSON."""
    
    try:
        import json
        
        metadata = {
            "extraction_results": [],
            "optimization_results": [],
            "export_timestamp": str(datetime.now()),
            "total_clips": len([r for r in extraction_results.results if r.success])
        }
        
        # Extract metadata from extraction results
        for result in extraction_results.results:
            if result.success:
                clip_meta = {
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "duration_seconds": result.duration_seconds,
                    "file_size_mb": result.file_size_mb,
                    "clip_path": str(result.clip_path),
                    "success": result.success
                }
                metadata["extraction_results"].append(clip_meta)
        
        # Extract metadata from optimization results
        if optimization_results:
            for result in optimization_results:
                if result.success:
                    opt_meta = {
                        "original_path": str(result.original_path),
                        "optimized_path": str(result.optimized_path),
                        "original_size_mb": result.original_size_mb,
                        "optimized_size_mb": result.optimized_size_mb,
                        "compression_ratio": result.compression_ratio,
                        "quality_score": result.quality_score,
                        "twitter_compatible": result.twitter_compatible,
                        "success": result.success
                    }
                    metadata["optimization_results"].append(opt_meta)
        
        # Create download button for metadata
        json_data = json.dumps(metadata, indent=2, default=str)
        
        st.download_button(
            label="📋 Download Clip Metadata (JSON)",
            data=json_data,
            file_name="clip_metadata.json",
            mime="application/json",
            use_container_width=True
        )
        
        st.success("✅ Metadata export ready!")
        
    except Exception as e:
        st.error(f"Failed to export metadata: {e}")


def export_summary_report(extraction_results, optimization_results):
    """Export a summary report as text."""
    
    try:
        successful_clips = [r for r in extraction_results.results if r.success]
        successful_optimized = [r for r in optimization_results if r.success] if optimization_results else []
        
        report_lines = [
            "YouTube to Twitter Clip Extraction - Summary Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EXTRACTION SUMMARY:",
            f"- Total clips extracted: {len(successful_clips)}",
            f"- Total original size: {sum(r.file_size_mb for r in successful_clips):.1f} MB",
            f"- Average clip duration: {sum(r.duration_seconds for r in successful_clips) / len(successful_clips):.1f}s",
            "",
            "OPTIMIZATION SUMMARY:",
            f"- Clips optimized: {len(successful_optimized)}",
            f"- Total optimized size: {sum(r.optimized_size_mb for r in successful_optimized):.1f} MB",
            f"- Average compression ratio: {sum(r.compression_ratio for r in successful_optimized) / len(successful_optimized):.1f}x",
            f"- Average quality score: {sum(r.quality_score for r in successful_optimized) / len(successful_optimized):.0f}/100",
            "",
            "CLIP DETAILS:",
        ]
        
        for i, clip in enumerate(successful_clips):
            report_lines.append(f"Clip {i+1}: {clip.start_time} - {clip.end_time} ({clip.duration_seconds:.1f}s, {clip.file_size_mb:.1f} MB)")
        
        if successful_optimized:
            report_lines.append("")
            report_lines.append("OPTIMIZATION DETAILS:")
            for i, opt in enumerate(successful_optimized):
                report_lines.append(f"Clip {i+1}: {opt.compression_ratio:.1f}x compression, {opt.quality_score:.0f}/100 quality")
        
        report_text = "\n".join(report_lines)
        
        st.download_button(
            label="📊 Download Summary Report (TXT)",
            data=report_text,
            file_name="clip_extraction_summary.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        st.success("✅ Summary report ready!")
        
    except Exception as e:
        st.error(f"Failed to export summary report: {e}")


def render_download_progress():
    """Render download progress tracking."""
    
    st.subheader("📥 Download Progress")
    
    progress = st.session_state.get('download_progress', 0)
    status = st.session_state.get('download_status', 'Preparing...')
    
    st.progress(progress / 100.0)
    st.write(f"**Status:** {status}")
    
    if progress >= 100:
        st.success("✅ Download complete!")
        st.session_state['batch_download_progress'] = False


def render_video_analytics_panel(
    video_info: Optional[Dict[str, Any]] = None,
    extraction_results: Optional[Any] = None,
    optimization_results: Optional[List[Any]] = None
) -> None:
    """
    Render analytics panel with video processing statistics.
    
    Args:
        video_info: Original video information
        extraction_results: Clip extraction results
        optimization_results: Twitter optimization results
    """
    st.subheader("📊 Processing Analytics")
    
    if not video_info:
        st.info("No analytics available yet.")
        return
    
    # Video information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Duration", f"{video_info.get('duration', 0):.0f}s")
    
    with col2:
        st.metric("Original Size", f"{video_info.get('size_mb', 0):.1f} MB")
    
    with col3:
        clips_count = len(extraction_results.results) if extraction_results else 0
        st.metric("Clips Extracted", clips_count)
    
    with col4:
        if optimization_results:
            optimized_count = sum(1 for r in optimization_results if r.success)
            st.metric("Twitter Ready", optimized_count)
    
    # Processing statistics
    if extraction_results:
        st.write("**Extraction Statistics:**")
        total_extracted_duration = sum(r.duration_seconds for r in extraction_results.results if r.success)
        total_extracted_size = sum(r.file_size_mb for r in extraction_results.results if r.success)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Clips Duration", f"{total_extracted_duration:.0f}s")
        with col2:
            st.metric("Total Clips Size", f"{total_extracted_size:.1f} MB")
    
    # Optimization statistics
    if optimization_results:
        successful_optimizations = [r for r in optimization_results if r.success]
        if successful_optimizations:
            avg_compression = sum(r.compression_ratio for r in successful_optimizations) / len(successful_optimizations)
            avg_quality = sum(r.quality_score for r in successful_optimizations) / len(successful_optimizations)
            total_optimized_size = sum(r.optimized_size_mb for r in successful_optimizations)
            
            st.write("**Optimization Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Compression", f"{avg_compression:.1f}x")
            with col2:
                st.metric("Avg Quality Score", f"{avg_quality:.0f}/100")
            with col3:
                st.metric("Total Optimized Size", f"{total_optimized_size:.1f} MB")


def extract_single_clip_from_recommendation(rec: Dict[str, Any], index: int) -> None:
    """
    Extract a single clip based on LLM recommendation.
    
    Args:
        rec: LLM recommendation dictionary
        index: Index of the recommendation
    """
    try:
        # Get video path from session state
        video_path = st.session_state.get('video_path') or st.session_state.get('local_video_path')
        
        # If no video path, try to download the video
        if not video_path:
            youtube_url = st.session_state.get('url')
            if not youtube_url or not st.session_state.get('valid_url'):
                st.error("❌ No valid YouTube URL available. Please process a video first.")
                return
            
            # Download video for clip extraction
            with st.spinner("📥 Downloading video for clip extraction..."):
                try:
                    from src.downloader import YouTubeDownloader
                    downloader = YouTubeDownloader()
                    downloaded_video_path, _ = downloader.download_video(youtube_url)
                    st.session_state['video_path'] = str(downloaded_video_path)
                    video_path = str(downloaded_video_path)
                    st.success("✅ Video downloaded successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to download video: {str(e)}")
                    return
        
        # Validate video file exists
        if not Path(video_path).exists():
            st.error(f"❌ Video file not found: {video_path}")
            return
        
        # Extract timing information
        start_time = rec.get('start_time', '00:00:00')
        end_time = rec.get('end_time', '00:01:00')
        
        # Convert recommendation to ClipRecommendation object
        try:
            from src.analyzer.llm_analyzer import ClipRecommendation, HookStrength
            clip_recommendation = ClipRecommendation(
                start_time=start_time,
                end_time=end_time,
                reasoning=rec.get('reasoning', 'AI recommended clip'),
                confidence=rec.get('confidence', 80),
                hook_strength=HookStrength(rec.get('hook_strength', 'medium')),
                keywords=rec.get('keywords', []),
                sentiment=rec.get('sentiment', 'positive')
            )
        except Exception as e:
            st.error(f"❌ Error creating clip recommendation: {e}")
            return
        
        # Show progress
        progress_container = st.container()
        with progress_container:
            with st.spinner(f"🎬 Extracting clip {index + 1}..."):
                try:
                    # Ensure output directories exist
                    clips_dir = Path("outputs/clips")
                    optimized_dir = Path("outputs/optimized")
                    clips_dir.mkdir(parents=True, exist_ok=True)
                    optimized_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Verify directories were created
                    if not clips_dir.exists():
                        st.error(f"❌ Failed to create clips directory: {clips_dir}")
                        return
                    if not optimized_dir.exists():
                        st.error(f"❌ Failed to create optimized directory: {optimized_dir}")
                        return
                    
                    # Initialize clip extractor (import locally to avoid circular imports)
                    from src.clipper.clip_extractor import ClipExtractor
                    clip_extractor = ClipExtractor(
                        output_dir=clips_dir,
                        cleanup_temp=False
                    )
                    
                    # Show extraction info
                    st.info(f"⏰ Extracting clip from {start_time} to {end_time}")
                    
                    # Extract the clip
                    extraction_result = clip_extractor.extract_clips_from_recommendations(
                        source_video=str(video_path),
                        recommendations=[clip_recommendation]
                    )
                    
                    if extraction_result.results and extraction_result.results[0].success:
                        extracted_clip = extraction_result.results[0]
                        
                        # Optimize for Twitter
                        with st.spinner("🐦 Optimizing for Twitter..."):
                            from src.clipper.twitter_optimizer import TwitterOptimizer
                            twitter_optimizer = TwitterOptimizer(
                                output_dir=optimized_dir
                            )
                            
                            from src.clipper.twitter_optimizer import VideoQuality
                            optimization_result = twitter_optimizer.optimize_for_twitter(
                                input_path=extracted_clip.clip_path,
                                quality=VideoQuality.HIGH
                            )
                        
                        # Store results in session state
                        if 'extraction_results' not in st.session_state or st.session_state['extraction_results'] is None:
                            st.session_state['extraction_results'] = []
                        
                        if 'optimization_results' not in st.session_state or st.session_state['optimization_results'] is None:
                            st.session_state['optimization_results'] = []
                        
                        # Add to results
                        st.session_state['extraction_results'].append(extracted_clip)
                        
                        if optimization_result.success:
                            st.session_state['optimization_results'].append(optimization_result)
                        
                        # Show success message
                        st.success(f"✅ Clip {index + 1} extracted successfully!")
                        st.info(f"📁 Original: {extracted_clip.clip_path}")
                        if optimization_result.success:
                            st.info(f"🐦 Optimized: {optimization_result.optimized_path}")
                        
                        # Update metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Duration", f"{extracted_clip.duration_seconds:.1f}s")
                        with col2:
                            st.metric("Size", f"{extracted_clip.file_size_mb:.1f} MB")
                        with col3:
                            if optimization_result.success:
                                st.metric("Compression", f"{optimization_result.compression_ratio:.1f}x")
                        
                        # Auto-refresh to show results
                        st.rerun()
                        
                    else:
                        st.error("❌ Clip extraction failed. Please check the video and timing parameters.")
                        if extraction_result.results:
                            st.error(f"Error: {extraction_result.results[0].error_message}")
                
                except Exception as e:
                    error_msg = str(e)
                    if "ffmpeg" in error_msg.lower():
                        st.error("❌ FFmpeg not found. Please install FFmpeg to extract clips:")
                        st.code("# On macOS:\nbrew install ffmpeg\n\n# On Ubuntu/Debian:\nsudo apt update\nsudo apt install ffmpeg", language="bash")
                    else:
                        st.error(f"❌ Extraction failed: {error_msg}")
                    logger.error(f"Clip extraction error: {e}")
    
    except Exception as e:
        st.error(f"❌ Unexpected error during clip extraction: {str(e)}")
        logger.error(f"Unexpected clip extraction error: {e}")


def initialize_extraction_results():
    """Initialize extraction results in session state if not present."""
    if 'extraction_results' not in st.session_state:
        st.session_state['extraction_results'] = []
    if 'optimization_results' not in st.session_state:
        st.session_state['optimization_results'] = []
