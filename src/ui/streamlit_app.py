"""
Main Streamlit Application for YouTube to Twitter Clip Extraction

This module provides the main user interface for the YouTube to Twitter clipper,
including URL input, settings configuration, progress tracking, and result display.
"""

import streamlit as st
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.config import config
from src.core.logger import get_logger
from src.downloader import YouTubeDownloader, YouTubeURLValidator, ThumbnailExtractor
from src.ui.components import (
    render_header,
    render_url_input,
    render_settings_panel,
    render_progress_section,
    render_error_display,
    render_video_info_display,
    render_thumbnail_display,
    initialize_session_state,
)


# Configure Streamlit page
st.set_page_config(
    page_title="YouTube to Twitter Clipper",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/youtube-clipper',
        'Report a bug': 'https://github.com/your-repo/youtube-clipper/issues',
        'About': "YouTube to Twitter Clip Extraction Tool - AI-powered video clip extraction"
    }
)

# Initialize logger
logger = get_logger("streamlit_app")


def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Main application layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎥 Video Input")
        
        # URL input section
        url_input_result = render_url_input()
        
        # File upload section
        render_file_upload_section()
        
        # Process button and results
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            process_video_workflow(url_input_result)
    
    with col2:
        st.header("⚙️ Settings")
        
        # Settings panel
        render_settings_panel()
        
        # Progress section
        render_progress_section()
    
    # Results section (full width)
    if st.session_state.get('video_processed', False):
        st.header("📊 Results")
        render_results_section()


def render_file_upload_section():
    """Render file upload section for local videos."""
    
    st.subheader("📁 Or Upload Local Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
        help="Upload a local video file to extract clips from"
    )
    
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
        
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "File type": uploaded_file.type
        }
        
        st.json(file_details)
        
        # Save uploaded file
        if st.button("📥 Save Uploaded File", key="save_upload"):
            try:
                # Save to temp directory
                temp_path = config.TEMP_DIR / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state['local_video_path'] = str(temp_path)
                st.success(f"✅ File saved: {temp_path.name}")
                
            except Exception as e:
                st.error(f"❌ Failed to save file: {e}")


def process_video_workflow(url_input_result: Dict[str, Any]):
    """Process the video workflow with progress tracking."""
    
    if not url_input_result.get('valid', False) and not st.session_state.get('local_video_path'):
        st.error("Please provide a valid YouTube URL or upload a local video file.")
        return
    
    # Initialize progress tracking
    st.session_state['processing'] = True
    st.session_state['current_step'] = 'Initializing...'
    st.session_state['progress'] = 0
    
    # Create progress containers
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize downloader
            update_progress(progress_bar, status_text, 10, "🔧 Initializing downloader...")
            downloader = YouTubeDownloader()
            
            video_info = None
            video_path = None
            
            if url_input_result.get('valid', False):
                # Process YouTube URL
                url = url_input_result['url']
                
                # Step 2: Extract video information
                update_progress(progress_bar, status_text, 30, "📋 Extracting video information...")
                video_info = downloader.get_video_info(url)
                st.session_state['video_info'] = video_info
                
                # Step 3: Extract thumbnail
                update_progress(progress_bar, status_text, 50, "🖼️ Extracting thumbnail...")
                extractor = ThumbnailExtractor()
                thumbnail_path = extractor.download_thumbnail(url, 'hqdefault', 'current_video')
                st.session_state['thumbnail_path'] = str(thumbnail_path)
                
                # Step 4: Download video (optional for now)
                if st.session_state.get('download_video', False):
                    update_progress(progress_bar, status_text, 70, "📥 Downloading video...")
                    video_path, _ = downloader.download_video(url)
                    st.session_state['video_path'] = str(video_path)
                
            else:
                # Process local video file
                video_path = st.session_state.get('local_video_path')
                update_progress(progress_bar, status_text, 50, "📁 Processing local video...")
                
                # Create mock video info for local files
                video_info = {
                    'title': Path(video_path).stem,
                    'duration': 'Unknown',
                    'uploader': 'Local File',
                    'file_path': video_path,
                }
                st.session_state['video_info'] = video_info
            
            # Step 5: Complete processing
            update_progress(progress_bar, status_text, 100, "✅ Processing complete!")
            
            # Mark as processed
            st.session_state['video_processed'] = True
            st.session_state['processing'] = False
            
            # Display success message
            st.success("🎉 Video processed successfully!")
            
            # Trigger rerun to show results
            st.rerun()
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            st.session_state['processing'] = False
            st.session_state['error'] = str(e)
            
            # Display error with details
            render_error_display(
                f"Processing failed: {e}",
                details=traceback.format_exc() if st.session_state.get('debug_mode', False) else None
            )


def update_progress(progress_bar, status_text, progress: int, message: str):
    """Update progress bar and status text."""
    progress_bar.progress(progress)
    status_text.text(message)
    st.session_state['progress'] = progress
    st.session_state['current_step'] = message
    time.sleep(0.1)  # Small delay for visual feedback


def render_results_section():
    """Render the results section with video info and options."""
    
    video_info = st.session_state.get('video_info', {})
    
    if not video_info:
        st.warning("No video information available.")
        return
    
    # Create columns for results layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Video information display
        render_video_info_display(video_info)
        
        # Action buttons
        st.subheader("🎬 Next Steps")
        
        if st.button("🎤 Transcribe Audio", use_container_width=True):
            st.info("🚧 Transcription feature coming in Task 2.1!")
        
        if st.button("🤖 Analyze Content", use_container_width=True):
            st.info("🚧 AI Analysis feature coming in Task 2.2!")
        
        if st.button("✂️ Extract Clips", use_container_width=True):
            st.info("🚧 Clip extraction feature coming in Task 3.1!")
    
    with col2:
        # Thumbnail display
        thumbnail_path = st.session_state.get('thumbnail_path')
        if thumbnail_path and Path(thumbnail_path).exists():
            render_thumbnail_display(thumbnail_path, video_info)
        
        # Video file info
        video_path = st.session_state.get('video_path') or st.session_state.get('local_video_path')
        if video_path and Path(video_path).exists():
            st.subheader("📁 Video File")
            file_info = {
                "Path": Path(video_path).name,
                "Size": f"{Path(video_path).stat().st_size / (1024*1024):.2f} MB",
                "Status": "✅ Available"
            }
            st.json(file_info)


def handle_session_reset():
    """Handle session reset functionality."""
    
    if st.sidebar.button("🔄 Reset Session", use_container_width=True):
        # Clear all session state except theme
        keys_to_keep = ['theme']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        
        st.sidebar.success("Session reset successfully!")
        st.rerun()


def render_debug_section():
    """Render debug section in sidebar."""
    
    if st.sidebar.checkbox("🐛 Debug Mode", value=False):
        st.session_state['debug_mode'] = True
        
        st.sidebar.subheader("Debug Information")
        
        # Session state
        with st.sidebar.expander("Session State"):
            st.json(dict(st.session_state))
        
        # Configuration
        with st.sidebar.expander("Configuration"):
            config_info = {
                "LLM Provider": config.DEFAULT_LLM_PROVIDER,
                "Whisper Model": config.WHISPER_MODEL,
                "Max Video Duration": f"{config.MAX_VIDEO_DURATION}s",
                "Output Dir": str(config.OUTPUT_DIR),
                "Temp Dir": str(config.TEMP_DIR),
            }
            st.json(config_info)
        
        # System info
        with st.sidebar.expander("System Info"):
            import platform
            system_info = {
                "Python Version": platform.python_version(),
                "Platform": platform.system(),
                "Streamlit Version": st.__version__,
            }
            st.json(system_info)
    else:
        st.session_state['debug_mode'] = False


def render_sidebar():
    """Render sidebar with additional options."""
    
    with st.sidebar:
        st.title("🛠️ Tools")
        
        # Session management
        st.subheader("Session")
        handle_session_reset()
        
        # App information
        st.subheader("📊 App Status")
        
        # Show current configuration
        status_info = {
            "Processing": "🟢 Ready" if not st.session_state.get('processing', False) else "🟡 Processing...",
            "Video Loaded": "✅ Yes" if st.session_state.get('video_processed', False) else "❌ No",
            "Errors": "❌ Yes" if st.session_state.get('error') else "✅ None",
        }
        
        for key, value in status_info.items():
            st.text(f"{key}: {value}")
        
        # Debug section
        render_debug_section()
        
        # Help section
        st.subheader("❓ Help")
        with st.expander("How to Use"):
            st.markdown("""
            1. **Enter YouTube URL** or upload a local video file
            2. **Configure settings** in the settings panel
            3. **Click Process Video** to begin extraction
            4. **Review results** and proceed with next steps
            
            **Supported formats:**
            - YouTube URLs (any format)
            - Local videos: MP4, MOV, AVI, MKV, WebM
            """)
        
        with st.expander("Keyboard Shortcuts"):
            st.markdown("""
            - `Ctrl+R`: Refresh page
            - `Ctrl+Shift+R`: Hard refresh
            - `F11`: Toggle fullscreen
            """)


if __name__ == "__main__":
    # Render sidebar
    render_sidebar()
    
    # Run main application
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please refresh the page.")
        
        if st.session_state.get('debug_mode', False):
            st.exception(e)
