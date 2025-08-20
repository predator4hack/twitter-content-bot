"""
Example: Integrating Whisper Transcription into the YouTube Twitter Clipper

This example shows how to use the WhisperTranscriber in the main application
to add transcription capabilities to video clips.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transcription import WhisperTranscriber, TranscriptionResult


class VideoProcessor:
    """
    Example integration of transcription into video processing workflow.
    """
    
    def __init__(self, transcriber_model: str = "base"):
        """Initialize with a Whisper transcriber."""
        self.transcriber = WhisperTranscriber(model_size=transcriber_model)
        
        # Verify transcriber is working
        if not self.transcriber.is_healthy():
            raise RuntimeError("Transcriber initialization failed")
        
        print(f"✅ Video processor initialized with Whisper model: {transcriber_model}")
        
    def process_video_with_transcription(self, video_path: str, language: str = None) -> dict:
        """
        Process a video file and generate transcription.
        
        Args:
            video_path: Path to the video file
            language: Language code for transcription (auto-detect if None)
            
        Returns:
            Dictionary with video processing results including transcription
        """
        results = {
            'video_path': video_path,
            'transcription': None,
            'clips': [],
            'metadata': {}
        }
        
        print(f"🎬 Processing video: {video_path}")
        
        # Step 1: Transcribe the video
        try:
            print("🎤 Transcribing audio...")
            transcription = self.transcriber.transcribe_file(video_path, language=language)
            results['transcription'] = transcription
            
            print(f"✅ Transcription completed:")
            print(f"   Language: {transcription.language}")
            print(f"   Duration: {transcription.duration:.2f}s")
            print(f"   Word count: {transcription.word_count}")
            print(f"   Segments: {len(transcription.segments)}")
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return results
        
        # Step 2: Identify interesting segments for clipping
        interesting_segments = self.find_interesting_segments(transcription)
        results['clips'] = interesting_segments
        
        # Step 3: Add metadata
        results['metadata'] = {
            'original_duration': transcription.duration,
            'total_segments': len(transcription.segments),
            'suggested_clips': len(interesting_segments),
            'language': transcription.language,
            'confidence': transcription.confidence
        }
        
        return results
    
    def find_interesting_segments(self, transcription: TranscriptionResult) -> list:
        """
        Analyze transcription to find segments suitable for Twitter clips.
        
        This is a simple example - in practice you might use more sophisticated
        analysis like sentiment analysis, keyword detection, etc.
        """
        clips = []
        
        # Example criteria for interesting segments:
        # 1. Segments with certain keywords
        # 2. Segments of appropriate length (10-30 seconds for Twitter)
        # 3. Complete sentences or thoughts
        
        keywords = [
            'important', 'amazing', 'incredible', 'breakthrough', 'never',
            'always', 'exactly', 'perfect', 'absolutely', 'definitely',
            'question', 'answer', 'secret', 'truth', 'reveal'
        ]
        
        for i, segment in enumerate(transcription.segments):
            segment_duration = segment.end_time - segment.start_time
            segment_text = segment.text.lower()
            
            # Check if segment is good length for Twitter (10-30 seconds)
            if 10 <= segment_duration <= 30:
                # Check for interesting keywords
                has_keywords = any(keyword in segment_text for keyword in keywords)
                
                # Check if it's a complete thought (ends with punctuation)
                complete_thought = segment.text.strip().endswith(('.', '!', '?'))
                
                if has_keywords or complete_thought:
                    clips.append({
                        'segment_index': i,
                        'start_time': segment.start_time,
                        'end_time': segment.end_time,
                        'duration': segment_duration,
                        'text': segment.text,
                        'confidence': segment.confidence,
                        'reason': 'keywords' if has_keywords else 'complete_thought'
                    })
        
        # Sort by confidence or other criteria
        clips.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"🎯 Found {len(clips)} interesting segments for clipping")
        return clips[:5]  # Return top 5 candidates
    
    def get_transcription_summary(self, transcription: TranscriptionResult) -> str:
        """Generate a summary of the transcription for social media."""
        if not transcription.text:
            return "No transcription available"
        
        # Simple summary - take first few sentences
        sentences = transcription.text.split('. ')
        summary = '. '.join(sentences[:2])
        
        if len(summary) > 200:
            summary = summary[:197] + "..."
        
        return summary
    
    def export_transcript(self, transcription: TranscriptionResult, output_path: str):
        """Export transcription to a text file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcription Results\n")
            f.write(f"Language: {transcription.language}\n")
            f.write(f"Duration: {transcription.duration:.2f} seconds\n")
            f.write(f"Confidence: {transcription.confidence:.3f}\n")
            f.write(f"Word count: {transcription.word_count}\n")
            f.write(f"\n{'='*50}\n")
            f.write(f"FULL TRANSCRIPT\n")
            f.write(f"{'='*50}\n\n")
            f.write(transcription.text)
            f.write(f"\n\n{'='*50}\n")
            f.write(f"SEGMENTS\n")
            f.write(f"{'='*50}\n\n")
            
            for i, segment in enumerate(transcription.segments, 1):
                f.write(f"{i:3d}. [{segment.start_time:6.2f}-{segment.end_time:6.2f}] "
                       f"(conf: {segment.confidence:.3f}) {segment.text}\n")
        
        print(f"📄 Transcript exported to: {output_path}")


def example_usage():
    """Demonstrate how to use the video processor with transcription."""
    # Initialize the processor
    processor = VideoProcessor(transcriber_model="tiny")  # Use tiny for faster processing
    
    print("\n" + "="*60)
    print("WHISPER TRANSCRIPTION INTEGRATION EXAMPLE")
    print("="*60)
    
    # Example video file path (replace with actual path)
    video_path = "sample_video.mp4"
    
    print(f"\nExample workflow for video: {video_path}")
    print("-" * 40)
    
    print("1. 🎬 Load video file")
    print("2. 🎤 Extract and transcribe audio using Whisper")
    print("3. 📝 Analyze transcription for interesting segments")
    print("4. ✂️  Identify clips suitable for Twitter (10-30 seconds)")
    print("5. 📊 Generate metadata and summaries")
    print("6. 💾 Export results")
    
    print(f"\nSupported languages: {len(processor.transcriber.get_supported_languages())}")
    print("Model loaded and ready for transcription!")
    
    # Example of how you would process a real video:
    """
    if os.path.exists(video_path):
        results = processor.process_video_with_transcription(video_path)
        
        # Export transcript
        transcript_path = video_path.replace('.mp4', '_transcript.txt')
        processor.export_transcript(results['transcription'], transcript_path)
        
        # Show suggested clips
        for i, clip in enumerate(results['clips'], 1):
            print(f"Clip {i}: {clip['start_time']:.1f}-{clip['end_time']:.1f}s - {clip['text'][:50]}...")
    """
    
    print("\n✅ Integration example completed!")
    print("\nTo use with real videos:")
    print("1. Place your video file in the project directory")
    print("2. Update the video_path variable")
    print("3. Run this script")


if __name__ == '__main__':
    try:
        example_usage()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
