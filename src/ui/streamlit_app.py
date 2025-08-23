"""
Main Streamlit Application for YouTube to Twitter Clip Extraction

This module provides the main user interface for the YouTube to Twitter clipper,
including URL input, settings configuration, progress tracking, and result display.
"""

import streamlit as st
import sys
import time
import asyncio
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
    render_progress_section,
    render_error_display,
    initialize_session_state,
    render_processing_progress,
    render_video_preview,
    render_thumbnail_display,
    create_download_button,
    extract_single_clip_from_recommendation,
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
    
    # Ensure output directories exist
    Path("outputs/clips").mkdir(parents=True, exist_ok=True)
    Path("outputs/optimized").mkdir(parents=True, exist_ok=True)
    
    # Render header
    st.title("✂️ YouTube to Twitter Clipper")
    # render_header()
    
    # Single page layout - Video input and processing
    # st.header("🎥 Video Input")
    
    # URL input section
    url_input_result = render_url_input()
    
    # File upload section
    # render_file_upload_section()
    
    # Process button
    if st.button("🚀 Process Video", type="primary", use_container_width=True):
        process_video_workflow(url_input_result)
    
    # Enhanced progress section
    if st.session_state.get('processing', False):
        render_processing_progress(
            st.session_state.get('current_step', ''),
            st.session_state.get('progress', 0),
            st.session_state.get('status_text', '')
        )
    
    # Show video info and thumbnail if processed
    if st.session_state.get('video_processed', False):
        video_info = st.session_state.get('video_info', {})
        if video_info:
            st.success(f"✅ Video processed: {video_info.get('title', 'Unknown title')}")
            
            # Show thumbnail preview with download
            thumbnail_path = st.session_state.get('thumbnail_path')
            if thumbnail_path and Path(thumbnail_path).exists():
                st.subheader("🖼️ Thumbnail Preview")
                render_thumbnail_display(thumbnail_path, video_info)
    
    # AI Analysis and Recommendations Section
    if st.session_state.get('video_processed', False):
        st.divider()
        render_ai_recommendations_section()


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
                
                # Step 4: Download video (required for clip extraction)
                update_progress(progress_bar, status_text, 70, "📥 Downloading video for clip extraction...")
                video_path, _ = downloader.download_video(url)
                st.session_state['video_path'] = str(video_path)
                
            else:
                # Process local video file
                video_path = st.session_state.get('local_video_path')
                if video_path:
                    st.session_state['video_path'] = str(video_path)
                update_progress(progress_bar, status_text, 50, "📁 Processing local video...")
                
                # Create mock video info for local files
                video_info = {
                    'title': Path(video_path).stem if video_path else 'Unknown',
                    'duration': 'Unknown',
                    'uploader': 'Local File',
                    'file_path': video_path,
                }
                st.session_state['video_info'] = video_info
            
            # Step 5: Transcribe video with Whisper
            update_progress(progress_bar, status_text, 75, "🎙️ Transcribing audio with Whisper...")
            transcription_result = transcribe_video_with_whisper(str(video_path))
            st.session_state['transcription_result'] = transcription_result
            
            # Step 6: Analyze content with LLM
            update_progress(progress_bar, status_text, 85, "🤖 Analyzing content with AI...")
            analysis_results = analyze_content_with_llm(transcription_result)
            st.session_state['analysis_results'] = analysis_results
            
            # Step 7: Extract clips based on LLM recommendations
            update_progress(progress_bar, status_text, 95, "✂️ Extracting recommended clips...")
            extract_clips_from_analysis(str(video_path), analysis_results)
            
            # Step 8: Complete processing
            update_progress(progress_bar, status_text, 100, "✅ Processing complete!")
            
            # Mark as processed
            st.session_state['video_processed'] = True
            st.session_state['processing'] = False
            
            # Display success message
            st.success("🎉 Video processing complete! Clips extracted and optimized.")
            
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


def transcribe_video_with_whisper(video_path: str):
    """Transcribe video using Whisper."""
    try:
        print(f"🔧 DEBUG: Starting Whisper transcription for: {video_path}")
        
        from src.transcription.whisper_transcriber import WhisperTranscriber
        
        # Initialize Whisper transcriber
        transcriber = WhisperTranscriber(
            model_size="base",  # Use base model for good speed/accuracy balance
            device="auto"
        )
        
        # Transcribe the video
        transcription_result = transcriber.transcribe_file(video_path)
        
        print(f"🔧 DEBUG: Transcription completed - {len(transcription_result.segments)} segments")
        print(f"🔧 DEBUG: Total duration: {transcription_result.duration:.1f}s")
        
        return transcription_result
        
    except Exception as e:
        print(f"❌ DEBUG: Transcription failed: {e}")
        logger.error(f"Whisper transcription failed: {e}")
        st.error(f"❌ Transcription failed: {str(e)}")
        raise


def analyze_content_with_llm(transcription_result):
    """Analyze transcription with LLM to get clip recommendations."""
    try:
        print(f"🔧 DEBUG: Starting LLM analysis")
        
        from src.analyzer.llm_analyzer import LLMAnalyzerFactory
        
        # Create LLM analyzer - try different providers for SSL issues
        from src.analyzer.llm_analyzer import LLMAnalyzerFactory
        
        # Get available providers and prefer Groq if available (fewer SSL issues)
        available_providers = LLMAnalyzerFactory.get_available_providers()
        print(f"🔧 DEBUG: Available LLM providers: {available_providers}")
        
        if "groq" in available_providers:
            print(f"🔧 DEBUG: Using Groq provider (more reliable)")
            analyzer = LLMAnalyzerFactory.create_analyzer("groq")
        elif "gemini" in available_providers:
            print(f"🔧 DEBUG: Using Gemini provider")
            analyzer = LLMAnalyzerFactory.create_analyzer("gemini")
        else:
            raise ValueError("No LLM providers available - check API keys")
        
        # Analyze transcript for 2 clips
        analysis_result = asyncio.run(analyzer.analyze_transcript(
            transcription_result,
            max_clips=2,
            target_duration=50  # ~50 second clips
        ))
        
        print(f"🔧 DEBUG: LLM analysis completed")
        print(f"🔧 DEBUG: Content type: {analysis_result.content_type}")
        print(f"🔧 DEBUG: Number of recommendations: {len(analysis_result.recommendations)}")
        
        # Convert to format expected by UI
        analysis_data = {
            'content_type': analysis_result.content_type.value,
            'strategy': 'ai_recommended',
            'recommendations': []
        }
        
        for rec in analysis_result.recommendations:
            analysis_data['recommendations'].append({
                'start_time': rec.start_time,
                'end_time': rec.end_time,
                'reasoning': rec.reasoning,
                'confidence': rec.confidence,
                'hook_strength': rec.hook_strength.value,
                'keywords': rec.keywords,
                'sentiment': rec.sentiment
            })
        
        print(f"🔧 DEBUG: Analysis data prepared for UI")
        return analysis_data
        
    except Exception as e:
        print(f"❌ DEBUG: LLM analysis failed: {e}")
        logger.error(f"LLM analysis failed: {e}")
        st.error(f"❌ AI analysis failed: {str(e)}")
        raise


def extract_clips_from_analysis(video_path: str, analysis_results: Dict[str, Any]):
    """Extract clips based on LLM analysis results."""
    try:
        print(f"🔧 DEBUG: Starting clip extraction from analysis")
        print(f"🔧 DEBUG: Video path: {video_path}")
        print(f"🔧 DEBUG: Number of recommendations: {len(analysis_results.get('recommendations', []))}")
        
        # Initialize session state for results
        if 'extraction_results' not in st.session_state:
            st.session_state['extraction_results'] = []
        if 'optimization_results' not in st.session_state:
            st.session_state['optimization_results'] = []
        
        # Clear previous results
        st.session_state['extraction_results'] = []
        st.session_state['optimization_results'] = []
        
        # Extract each recommended clip
        recommendations = analysis_results.get('recommendations', [])
        for i, rec in enumerate(recommendations):
            try:
                print(f"🔧 DEBUG: Extracting clip {i+1}: {rec['start_time']} to {rec['end_time']}")
                
                # Use the existing extraction function but don't trigger rerun
                extract_single_clip_from_recommendation(rec, i)
                
                print(f"✅ DEBUG: Clip {i+1} extracted successfully")
                
            except Exception as clip_error:
                print(f"❌ DEBUG: Failed to extract clip {i+1}: {clip_error}")
                logger.error(f"Failed to extract clip {i+1}: {clip_error}")
                continue
        
        final_count = len(st.session_state.get('extraction_results', []))
        print(f"🔧 DEBUG: Extraction completed - {final_count} clips in session state")
        
    except Exception as e:
        print(f"❌ DEBUG: Clip extraction from analysis failed: {e}")
        logger.error(f"Clip extraction from analysis failed: {e}")
        st.error(f"❌ Clip extraction failed: {str(e)}")
        raise


# Removed old demo functions - now using proper pipeline


# Removed - simplified to show just success message


def render_ai_recommendations_section():
    """Render AI analysis and clip recommendations in a clean single-page format."""
    
    st.header("🤖 AI Recommendations")
    
    # Check if we have analysis results from the pipeline
    analysis_results = st.session_state.get('analysis_results')
    
    if not analysis_results:
        st.info("🤖 Process a video to see AI-powered clip recommendations with reasoning.")
        return
    
    # Show analysis summary
    st.subheader("📊 Content Analysis Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Content Type", analysis_results.get('content_type', 'Unknown').title())
    
    with col2:
        recommendations = analysis_results.get('recommendations', [])
        st.metric("Clips Recommended", len(recommendations))
    
    with col3:
        if recommendations:
            avg_confidence = sum(r.get('confidence', 0) for r in recommendations) / len(recommendations)
            st.metric("Avg Confidence", f"{avg_confidence:.0f}%")
    
    # Show transcription info if available
    transcription_result = st.session_state.get('transcription_result')
    if transcription_result:
        with st.expander("📝 Transcription Info"):
            st.write(f"**Duration:** {transcription_result.duration:.1f} seconds")
            st.write(f"**Language:** {transcription_result.language}")
            st.write(f"**Segments:** {len(transcription_result.segments)}")
    
    # Render the recommendations
    render_clean_recommendations(analysis_results)


def render_clean_recommendations(analysis_results):
    """Render recommendations in a clean, focused format."""
    
    if not analysis_results or 'recommendations' not in analysis_results:
        return
    
    recommendations = analysis_results['recommendations'][:2]  # Show only top 2
    
    for i, rec in enumerate(recommendations):
        st.subheader(f"🎬 Recommendation {i+1}")
        
        # Create two columns for preview and info
        col_preview, col_info = st.columns([1, 1])
        
        with col_preview:
            # st.write("**🔍 Preview**")
            
            # Check if we have actual extracted clips for this recommendation
            extraction_results = st.session_state.get('extraction_results')
            optimization_results = st.session_state.get('optimization_results')
            
            has_actual_clip = (extraction_results and 
                             isinstance(extraction_results, list) and 
                             i < len(extraction_results) and 
                             extraction_results[i] is not None and
                             extraction_results[i].success)
            
            if has_actual_clip:
                # Show actual video preview
                clip_result = extraction_results[i]
                if Path(clip_result.clip_path).exists():
                    render_video_preview(clip_result.clip_path, width=300)
                    
                    # Show clip info
                    st.write(f"**Duration:** {clip_result.duration_seconds:.1f}s")
                    st.write(f"**Size:** {clip_result.file_size_mb:.1f} MB")
                else:
                    # Fallback to placeholder if file doesn't exist
                    render_video_placeholder(rec)
            else:
                # Show placeholder
                render_video_placeholder(rec)
            
            # Download buttons
            if has_actual_clip:
                clip_result = extraction_results[i]
                
                # Original clip download
                if Path(clip_result.clip_path).exists():
                    with open(clip_result.clip_path, "rb") as file:
                        file_data = file.read()
                    st.download_button(
                        label=f"📎 Download Original",
                        data=file_data,
                        file_name=f"clip_{i+1}_original.mp4",
                        mime="video/mp4",
                        key=f"download_original_{i}",
                        use_container_width=True
                    )
                
                # Optimized clip download if available
                has_optimized = (optimization_results and 
                               isinstance(optimization_results, list) and
                               i < len(optimization_results) and 
                               optimization_results[i] is not None and
                               optimization_results[i].success and
                               Path(optimization_results[i].optimized_path).exists())
                
                if has_optimized and optimization_results[i] is not None:
                    with open(optimization_results[i].optimized_path, "rb") as file:
                        file_data = file.read()
                    st.download_button(
                        label=f"🐦 Download Twitter-Ready",
                        data=file_data,
                        file_name=f"clip_{i+1}_twitter.mp4",
                        mime="video/mp4",
                        key=f"download_optimized_{i}",
                        use_container_width=True
                    )
            else:
                # Show message that clips are being processed
                st.info("🎬 Clips will be ready after video processing")
        
        with col_info:
            st.write("**🧠 Analysis & Reasoning**")
            
            # Confidence and metrics
            confidence = rec.get('confidence', 0)
            hook_strength = rec.get('hook_strength', 'medium')
            
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                if confidence >= 80:
                    st.success(f"**Confidence:** {confidence}%")
                elif confidence >= 60:
                    st.warning(f"**Confidence:** {confidence}%")
                else:
                    st.error(f"**Confidence:** {confidence}%")
            
            with col_metrics2:
                hook_emoji = {"high": "🔥", "medium": "⚡", "low": "💡"}.get(hook_strength, "❓")
                st.write(f"**Hook:** {hook_emoji} {hook_strength.title()}")
            
            # Reasoning
            st.write("**Why this works:**")
            reasoning = rec.get('reasoning', 'No reasoning provided')
            st.write(f"💭 {reasoning}")
            
            # Keywords
            if 'keywords' in rec and rec['keywords']:
                st.write("**Key topics:**")
                keywords_str = ", ".join([f"🏷️ {kw}" for kw in rec['keywords'][:3]])
                st.write(keywords_str)
        
        if i < len(recommendations) - 1:
            st.divider()
    
    # Final debug summary
    print(f"\n🔧 DEBUG: render_clean_recommendations completed")
    print(f"🔧 DEBUG: Rendered {len(recommendations)} recommendation UI elements")
    extraction_results = st.session_state.get('extraction_results', [])
    print(f"🔧 DEBUG: Available clips in session state: {len(extraction_results)}")
    for i, clip in enumerate(extraction_results):
        if clip:
            print(f"🔧 DEBUG: Clip {i+1}: {clip.clip_path} (exists: {Path(clip.clip_path).exists()})")


def render_video_placeholder(rec):
    """Render a placeholder for video preview."""
    st.markdown(f"""
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
        🎬 Clip Preview<br>
        <small>{rec.get('start_time', 'N/A')} - {rec.get('end_time', 'N/A')}</small>
    </div>
    """, unsafe_allow_html=True)


# Removed - functionality moved to render_ai_recommendations_section which handles everything in one place


# Removed - functionality moved to render_ai_recommendations_section


# Removed - analytics functionality simplified


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
    """Render simplified sidebar with only essential controls."""
    
    with st.sidebar:
        st.title("🛠️ Controls")
        
        # Session reset
        handle_session_reset()
        
        st.divider()
        
        # Debug section
        render_debug_section()


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
