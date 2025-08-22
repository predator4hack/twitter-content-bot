#!/usr/bin/env python3
"""
Example script demonstrating the end-to-end YouTube to Twitter clip pipeline.

This script shows how to use the TwitterClipPipeline to process a YouTube video
and generate optimized Twitter clips with progress tracking.
"""

import asyncio
import sys
from pathlib import Path

from src.core.pipeline import (
    TwitterClipPipeline, 
    PipelineProgress, 
    PipelineStage,
    process_youtube_video
)


def progress_callback(progress: PipelineProgress):
    """
    Progress callback function to display pipeline status.
    
    Args:
        progress: PipelineProgress object with current status
    """
    stage_names = {
        PipelineStage.VALIDATION: "🔍 Validating URL",
        PipelineStage.DOWNLOAD: "⬇️  Downloading video",
        PipelineStage.THUMBNAIL: "🖼️  Extracting thumbnail", 
        PipelineStage.TRANSCRIPTION: "🎤 Transcribing audio",
        PipelineStage.ANALYSIS: "🤖 Analyzing content",
        PipelineStage.EXTRACTION: "✂️  Extracting clips",
        PipelineStage.OPTIMIZATION: "🚀 Optimizing for Twitter",
        PipelineStage.CLEANUP: "🧹 Cleaning up",
        PipelineStage.COMPLETED: "✅ Completed successfully",
        PipelineStage.FAILED: "❌ Failed"
    }
    
    stage_name = stage_names.get(progress.current_stage, progress.current_stage.value)
    
    print(f"\r{stage_name} ({progress.progress_percentage:.1f}%) - "
          f"Elapsed: {progress.elapsed_time:.1f}s", end="", flush=True)
    
    if progress.current_stage in [PipelineStage.COMPLETED, PipelineStage.FAILED]:
        print()  # New line at completion
        
        if progress.error_message:
            print(f"Error: {progress.error_message}")


async def process_video_example():
    """Example of processing a video using the pipeline."""
    
    # Example YouTube URL (replace with actual URL)
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print("🎬 YouTube to Twitter Clip Pipeline Example")
    print("=" * 50)
    print(f"Processing: {youtube_url}")
    print()
    
    # Method 1: Using the convenience function
    print("Method 1: Using convenience function")
    print("-" * 30)
    
    try:
        result = await process_youtube_video(
            youtube_url=youtube_url,
            output_dir=Path("example_outputs"),
            num_clips=2,
            max_clip_duration=120,
            llm_provider="gemini",  # or "groq"
            progress_callback=progress_callback
        )
        
        if result.success:
            print(f"\n🎉 Pipeline completed successfully in {result.execution_time:.2f}s")
            print(f"📁 Output directory: {Path('example_outputs').absolute()}")
            print(f"🎥 Original video: {result.video_path}")
            print(f"🖼️  Thumbnail: {result.thumbnail_path}")
            print(f"📝 Transcription: {len(result.transcription_result.segments)} segments")
            print(f"🎯 Recommendations: {len(result.clip_recommendations)} clips")
            print(f"✂️  Extracted clips: {len(result.extracted_clips)}")
            print(f"🚀 Optimized clips: {len(result.optimized_clips)}")
            
            # Display clip details
            print("\n📋 Generated Clips:")
            for i, clip in enumerate(result.optimized_clips, 1):
                print(f"  {i}. {Path(clip['optimized_path']).name}")
                print(f"     Size: {clip['optimized_size_mb']:.2f}MB "
                      f"(Compression: {clip['compression_ratio']:.2f})")
                print(f"     Quality: {clip['quality_score']}/100")
        else:
            print(f"\n❌ Pipeline failed: {result.error_message}")
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")


async def custom_pipeline_example():
    """Example of using the pipeline with custom configuration."""
    
    print("\n\nMethod 2: Using custom pipeline configuration")
    print("-" * 45)
    
    # Create custom pipeline
    pipeline = TwitterClipPipeline(
        output_dir=Path("custom_outputs"),
        max_retries=2,
        retry_delay=2.0,
        llm_provider="groq",  # Use Groq for faster processing
        whisper_model="small",  # Better quality than base
        cleanup_temp_files=True
    )
    
    # Set progress callback
    pipeline.set_progress_callback(progress_callback)
    
    # Process video with custom settings
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        result = await pipeline.process_video(
            youtube_url=youtube_url,
            num_clips=3,  # More clips
            max_clip_duration=60,  # Shorter clips
            twitter_strategy="viral"  # Viral strategy
        )
        
        if result.success:
            print(f"\n🎉 Custom pipeline completed in {result.execution_time:.2f}s")
            print(f"📁 Custom output: {Path('custom_outputs').absolute()}")
            
            # Show recommendations with reasoning
            print("\n🤖 LLM Analysis Results:")
            for i, rec in enumerate(result.clip_recommendations, 1):
                print(f"  {i}. {rec.start_time} - {rec.end_time}")
                print(f"     Confidence: {rec.confidence}%")
                print(f"     Hook: {rec.hook_strength.value}")
                print(f"     Reasoning: {rec.reasoning}")
                print(f"     Twitter text: {rec.twitter_text}")
                print()
        else:
            print(f"\n❌ Custom pipeline failed: {result.error_message}")
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")


def print_usage():
    """Print usage instructions."""
    print("Usage: python example_pipeline.py [youtube_url]")
    print("\nExamples:")
    print("  python example_pipeline.py")
    print("  python example_pipeline.py 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'")
    print("\nNote: This example requires:")
    print("  - Valid API keys in .env file (GEMINI_API_KEY and/or GROQ_API_KEY)")
    print("  - All dependencies installed (pip install -e .)")
    print("  - Internet connection for downloading videos")


async def main():
    """Main function to run the pipeline examples."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print_usage()
            return
            
        # Use provided URL
        youtube_url = sys.argv[1]
        
        print(f"🎬 Processing provided URL: {youtube_url}")
        print("=" * 60)
        
        try:
            result = await process_youtube_video(
                youtube_url=youtube_url,
                output_dir=Path("outputs"),
                num_clips=2,
                progress_callback=progress_callback
            )
            
            if result.success:
                print(f"\n✅ Successfully processed: {youtube_url}")
                print(f"📁 Check 'outputs' directory for results")
            else:
                print(f"\n❌ Failed to process: {result.error_message}")
                
        except Exception as e:
            print(f"\n💥 Error: {e}")
    else:
        # Run examples with demo URL
        print("🚀 Running pipeline examples with demo configuration")
        print("=" * 60)
        print("Note: Replace the demo URL with actual YouTube videos for real testing")
        print()
        
        await process_video_example()
        await custom_pipeline_example()
        
        print("\n" + "=" * 60)
        print("🎯 Examples completed!")
        print("\nTo test with your own video:")
        print("  python example_pipeline.py 'https://www.youtube.com/watch?v=YOUR_VIDEO_ID'")


if __name__ == "__main__":
    # Set up event loop policy for Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)