#!/usr/bin/env python3
"""
Demo script to test Twitter thread generation functionality.

This script demonstrates the end-to-end thread generation feature without
requiring a full video processing pipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analyzer.thread_generator import TwitterThreadGenerator
from src.analyzer.llm_analyzer import ContentType
from src.transcription.base import TranscriptionResult, TranscriptionSegment


def create_mock_transcript():
    """Create a mock transcript for testing."""
    segments = [
        TranscriptionSegment(
            text="Welcome to this video about artificial intelligence and machine learning.",
            start_time=0.0,
            end_time=5.0,
            confidence=0.95
        ),
        TranscriptionSegment(
            text="Today we're going to explore how neural networks actually work.",
            start_time=5.0,
            end_time=10.0,
            confidence=0.92
        ),
        TranscriptionSegment(
            text="Neural networks are inspired by how the human brain processes information.",
            start_time=10.0,
            end_time=15.0,
            confidence=0.88
        ),
        TranscriptionSegment(
            text="They consist of layers of interconnected nodes called neurons.",
            start_time=15.0,
            end_time=20.0,
            confidence=0.90
        ),
        TranscriptionSegment(
            text="Each neuron receives inputs, processes them, and produces an output.",
            start_time=20.0,
            end_time=25.0,
            confidence=0.93
        ),
        TranscriptionSegment(
            text="The magic happens when you combine many layers together in deep learning.",
            start_time=25.0,
            end_time=30.0,
            confidence=0.87
        ),
        TranscriptionSegment(
            text="This allows the network to learn complex patterns and relationships.",
            start_time=30.0,
            end_time=35.0,
            confidence=0.91
        ),
        TranscriptionSegment(
            text="That's the basic concept behind neural networks and deep learning.",
            start_time=35.0,
            end_time=40.0,
            confidence=0.94
        )
    ]
    
    full_text = " ".join(segment.text for segment in segments)
    
    return TranscriptionResult(
        text=full_text,
        segments=segments,
        language="en",
        confidence=0.91,
        metadata={"duration": 40.0}
    )


async def demo_thread_generation():
    """Demo the thread generation functionality."""
    print("🧵 Twitter Thread Generation Demo")
    print("=" * 50)
    
    # Create mock transcript
    print("📝 Creating mock transcript...")
    transcript = create_mock_transcript()
    print(f"✅ Created transcript with {len(transcript.segments)} segments")
    print(f"   Duration: {transcript.duration:.1f} seconds")
    print(f"   Language: {transcript.language}")
    
    # Test the thread generator (mock mode for demo)
    print("\n🤖 Testing thread generation...")
    
    try:
        # Create generator with mock mode
        generator = TwitterThreadGenerator()
        
        # Test fallback thread generation first
        print("🔧 Testing fallback thread creation...")
        fallback_data = generator._create_fallback_thread()
        fallback_thread = generator._create_twitter_thread(
            fallback_data, 
            transcript,
            "https://youtube.com/watch?v=demo123"
        )
        
        print(f"✅ Fallback thread created with {fallback_thread.total_tweets} tweets")
        
        # Display the fallback thread
        print("\n📋 Fallback Thread Preview:")
        print("-" * 30)
        for tweet in fallback_thread.tweets:
            print(f"Tweet {tweet.tweet_number}: {tweet.content}")
            print(f"  Characters: {tweet.character_count}/280")
            if tweet.hashtags:
                print(f"  Hashtags: {', '.join(tweet.hashtags)}")
            print()
        
        print(f"Content Type: {fallback_thread.content_type.value}")
        print(f"Reading Time: {fallback_thread.estimated_reading_time} seconds")
        print(f"Reasoning: {fallback_thread.reasoning}")
        
        # Test validation
        print("\n🔍 Testing thread validation...")
        validation_issues = generator.validate_thread(fallback_thread)
        if validation_issues:
            print("⚠️ Validation issues found:")
            for issue in validation_issues:
                print(f"  - {issue}")
        else:
            print("✅ Thread passes all validation checks")
        
        print("\n🎉 Demo completed successfully!")
        print("\nTo see this in action with real LLM generation:")
        print("1. Set up your API keys in .env file")
        print("2. Run the Streamlit app: streamlit run src/ui/streamlit_app.py")
        print("3. Process a YouTube video")
        print("4. Use the '🧵 Twitter Thread Generator' section")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def test_data_structures():
    """Test the basic data structures."""
    print("\n🧪 Testing Data Structures")
    print("-" * 30)
    
    from src.analyzer.llm_analyzer import ThreadTweet, TwitterThread, ContentType
    
    # Test ThreadTweet
    tweet = ThreadTweet(
        content="🧵 Test tweet about AI",
        tweet_number=1,
        character_count=0,  # Will be calculated
        hashtags=["#AI", "#test"]
    )
    
    print(f"✅ ThreadTweet: '{tweet.content}' ({tweet.character_count} chars)")
    
    # Test TwitterThread
    tweets = [
        ThreadTweet("🧵 Hook tweet", 1, 0),
        ThreadTweet("2/ Middle tweet", 2, 0),
        ThreadTweet("3/ Final tweet", 3, 0)
    ]
    
    thread = TwitterThread(
        tweets=tweets,
        hook_tweet=tweets[0],
        total_tweets=3,
        estimated_reading_time=45,
        reasoning="Test thread",
        content_type=ContentType.EDUCATIONAL
    )
    
    print(f"✅ TwitterThread: {thread.total_tweets} tweets, {thread.content_type.value} type")
    print("✅ All data structures working correctly")


if __name__ == "__main__":
    print("Starting Twitter Thread Generation Demo...")
    
    # Test basic data structures first
    test_data_structures()
    
    # Run the main demo
    asyncio.run(demo_thread_generation())