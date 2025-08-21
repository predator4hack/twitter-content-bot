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
    render_video_preview,
    render_clip_results_gallery,
    render_llm_reasoning_display,
    render_processing_progress,
    render_clip_comparison_view,
    render_batch_download_section,
    render_video_analytics_panel,
    create_download_button,
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
    
    # Main application layout with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Input & Processing", "🎬 Clip Results", "🤖 AI Analysis", "📊 Analytics"])
    
    with tab1:
        # Main processing interface
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
            
            # Enhanced progress section
            if st.session_state.get('processing', False):
                render_processing_progress(
                    st.session_state.get('current_step', ''),
                    st.session_state.get('progress', 0),
                    st.session_state.get('status_text', '')
                )
            else:
                render_progress_section()
        
        # Basic results section for processed video
        if st.session_state.get('video_processed', False):
            st.header("📊 Video Information")
            render_basic_results_section()
    
    with tab2:
        # Enhanced clip results with preview and downloads
        render_enhanced_clip_results()
    
    with tab3:
        # LLM analysis and reasoning display
        render_ai_analysis_section()
    
    with tab4:
        # Analytics and statistics
        render_analytics_section()


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
                    'title': Path(video_path).stem if video_path else 'Unknown',
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


def render_basic_results_section():
    """Render basic video information and processing status."""
    
    video_info = st.session_state.get('video_info', {})
    
    if not video_info:
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Video information display
        render_video_info_display(video_info)
        
        # Action buttons
        st.subheader("🎬 Next Steps")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🤖 Analyze Content", use_container_width=True, key="analyze_content_basic"):
                st.info("🚧 AI Analysis feature - Demo available in AI Analysis tab!")
        
        with col_b:
            if st.button("✂️ Extract Clips", use_container_width=True, key="extract_clips_basic"):
                st.info("🚧 Clip extraction feature - Demo available in Clip Results tab!")
    
    with col2:
        # Thumbnail display
        thumbnail_path = st.session_state.get('thumbnail_path')
        if thumbnail_path and Path(thumbnail_path).exists():
            render_thumbnail_display(thumbnail_path, video_info)
        
        # Video preview if available
        video_path = st.session_state.get('video_path') or st.session_state.get('local_video_path')
        if video_path and Path(video_path).exists():
            st.subheader("📹 Video Preview")
            render_video_preview(video_path, width=300)


def render_enhanced_clip_results():
    """Render enhanced clip results with previews and downloads."""
    
    st.header("🎬 Extracted Clips & Results")
    
    # Check if we have any processing results
    extraction_results = st.session_state.get('extraction_results')
    optimization_results = st.session_state.get('optimization_results')
    
    if not extraction_results:
        # Demo mode with sample clips
        st.info("No clips extracted yet. Here's a preview of what clip results would look like:")
        
        # Sample demonstration
        with st.expander("Demo: Sample Clip Results", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write("**Original Clip**")
                # Show video placeholder
                st.markdown("""
                <div style="
                    width: 100%;
                    height: 180px;
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
                st.write("**Duration:** 45.2s")
                st.write("**Size:** 12.3 MB")
            
            with col2:
                st.write("**Twitter Optimized**")
                # Show optimized video placeholder
                st.markdown("""
                <div style="
                    width: 100%;
                    height: 180px;
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
                st.write("**Size:** 8.1 MB")
                st.write("**Compression:** 1.5x")
                st.write("**Quality Score:** 87/100")
                st.success("✅ Twitter Compatible")
            
            with col3:
                st.write("**Downloads**")
                st.button("📥 Original", disabled=True)
                st.button("🐦 Twitter Ready", disabled=True)
        
        # Instructions for getting actual results
        st.markdown("""
        **To see actual clip results:**
        1. Process a video in the Input & Processing tab
        2. Run content analysis (coming in Phase 2)
        3. Extract clips based on AI recommendations
        4. Twitter optimization will be applied automatically
        """)
        
        return
    
    # Render actual results
    render_clip_results_gallery(extraction_results, optimization_results)
    
    # Batch download options
    render_batch_download_section(extraction_results, optimization_results)
    
    # Clip comparison tool
    if extraction_results.results and len(extraction_results.results) >= 2:
        st.markdown("---")
        clips_data = [
            {
                'start_time': result.start_time,
                'end_time': result.end_time,
                'video_path': result.clip_path,
                'duration': f"{result.duration_seconds:.1f}s",
                'size_mb': result.file_size_mb
            }
            for result in extraction_results.results if result.success
        ]
        render_clip_comparison_view(clips_data)


def render_ai_analysis_section():
    """Render AI analysis results and reasoning."""
    
    st.header("🤖 AI Content Analysis & Recommendations")
    
    # Check if we have analysis results
    analysis_results = st.session_state.get('analysis_results')
    
    if not analysis_results:
        # Demo mode with sample analysis
        st.info("No AI analysis available yet. Here's a preview of what AI analysis results would look like:")
        
        # Sample analysis demonstration
        sample_analysis = {
            'content_type': 'educational',
            'strategy': 'thought_leadership',
            'recommendations': [
                {
                    'start_time': '00:02:15',
                    'end_time': '00:03:00',
                    'reasoning': 'This segment contains a clear, actionable explanation that would resonate well with Twitter audiences. The speaker provides a concrete example that could spark engagement and discussion.',
                    'confidence': 87,
                    'hook_strength': 'high',
                    'keywords': ['productivity', 'tips', 'workflow']
                },
                {
                    'start_time': '00:05:30',
                    'end_time': '00:06:15',
                    'reasoning': 'Strong hook with an unexpected insight that challenges conventional thinking. The emotional appeal and practical value make this highly shareable content.',
                    'confidence': 92,
                    'hook_strength': 'high',
                    'keywords': ['innovation', 'mindset', 'success']
                }
            ]
        }
        
        render_llm_reasoning_display(sample_analysis)
        
        # Instructions for getting actual analysis
        st.markdown("""
        **To get actual AI analysis:**
        1. Process a video with transcription enabled
        2. Our AI will analyze the content for viral potential
        3. Recommendations will appear here with detailed reasoning
        4. Use the recommendations to guide clip extraction
        """)
        
        return
    
    # Render actual analysis results
    render_llm_reasoning_display(analysis_results)
    
    # Additional analysis tools
    st.markdown("---")
    st.subheader("🔧 Analysis Tools")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Re-analyze Content", use_container_width=True):
            st.info("Feature coming soon: Re-run AI analysis with different parameters")
    
    with col2:
        if st.button("📊 Detailed Sentiment Analysis", use_container_width=True):
            st.info("Feature coming soon: Deep sentiment and keyword analysis")


def render_analytics_section():
    """Render processing analytics and performance metrics."""
    
    st.header("📊 Processing Analytics & Performance")
    
    # Check if we have processing data
    video_info = st.session_state.get('video_info')
    extraction_results = st.session_state.get('extraction_results')
    optimization_results = st.session_state.get('optimization_results')
    
    if not video_info:
        st.info("No analytics available yet. Process a video to see detailed performance metrics.")
        
        # Demo analytics
        st.subheader("📈 Sample Analytics Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Processing Time", "2m 34s", delta="-23s")
        with col2:
            st.metric("Success Rate", "100%", delta="✅")
        with col3:
            st.metric("Total Clips", "3", delta="+1")
        with col4:
            st.metric("Avg Compression", "2.1x", delta="+0.3x")
        
        # Sample charts
        st.subheader("📈 Performance Charts")
        
        import pandas as pd
        import numpy as np
        
        # Sample processing timeline
        timeline_data = pd.DataFrame({
            'Step': ['Download', 'Transcribe', 'Analyze', 'Extract', 'Optimize'],
            'Duration (s)': [15, 45, 8, 22, 34],
            'Status': ['✅', '✅', '✅', '✅', '✅']
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Processing Timeline**")
            st.bar_chart(timeline_data.set_index('Step')['Duration (s)'])
        
        with col2:
            # Sample compression ratios
            compression_data = pd.DataFrame({
                'Clip': ['Clip 1', 'Clip 2', 'Clip 3'],
                'Original (MB)': [15.2, 18.7, 12.1],
                'Optimized (MB)': [7.8, 8.9, 6.2]
            })
            st.write("**Compression Results**")
            st.bar_chart(compression_data.set_index('Clip'))
        
        return
    
    # Render actual analytics
    render_video_analytics_panel(video_info, extraction_results, optimization_results)
    
    # Processing performance metrics
    if st.session_state.get('processing_metrics'):
        st.markdown("---")
        st.subheader("⚡ Performance Metrics")
        
        metrics = st.session_state['processing_metrics']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Processing Time", f"{metrics.get('total_time', 0):.1f}s")
        with col2:
            st.metric("Avg Time per Clip", f"{metrics.get('avg_clip_time', 0):.1f}s")
        with col3:
            st.metric("Processing Efficiency", f"{metrics.get('efficiency', 100):.0f}%")


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


def handle_session_reset():
    """Handle session reset functionality."""
    
    if st.button("🔄 Reset Session", use_container_width=True):
        # Clear all session state except theme
        keys_to_keep = ['theme']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        
        st.success("✅ Session reset successfully!")
        st.rerun()


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
