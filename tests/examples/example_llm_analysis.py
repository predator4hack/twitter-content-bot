"""
Example integration script demonstrating LLM content analysis.

This script shows how to use the LLM analyzer to analyze video transcripts
and get recommendations for Twitter clips.
"""

import asyncio
import json
from pathlib import Path

from src.analyzer import (
    analyze_content,
    ContentStrategy,
    LLMAnalyzerFactory,
    ContentType,
    TwitterStrategy
)
from src.transcription.base import TranscriptionResult, TranscriptionSegment
from src.core.config import config


def create_sample_transcript() -> TranscriptionResult:
    """Create a sample transcript for demonstration."""
    segments = [
        TranscriptionSegment(
            text="Hey everyone, welcome back to my channel! Today I have something absolutely incredible to share with you.",
            start_time=0.0,
            end_time=6.0,
            confidence=0.95
        ),
        TranscriptionSegment(
            text="This Python trick that I discovered will completely change how you think about programming.",
            start_time=6.0,
            end_time=12.0,
            confidence=0.92
        ),
        TranscriptionSegment(
            text="Most developers don't know about this, but it can reduce your code by 50% and make it twice as fast.",
            start_time=12.0,
            end_time=19.0,
            confidence=0.94
        ),
        TranscriptionSegment(
            text="Let me show you exactly how it works. First, you need to understand the basic concept.",
            start_time=60.0,
            end_time=66.0,
            confidence=0.90
        ),
        TranscriptionSegment(
            text="When you apply this technique to list comprehensions, the performance improvement is dramatic.",
            start_time=66.0,
            end_time=72.0,
            confidence=0.93
        ),
        TranscriptionSegment(
            text="I tested this on a million-item dataset and the results were shocking - 10x faster execution!",
            start_time=120.0,
            end_time=127.0,
            confidence=0.96
        ),
        TranscriptionSegment(
            text="Before I show you the code, make sure to subscribe and hit the bell icon for more programming secrets.",
            start_time=127.0,
            end_time=134.0,
            confidence=0.88
        )
    ]
    
    full_text = " ".join(segment.text for segment in segments)
    
    return TranscriptionResult(
        text=full_text,
        segments=segments,
        language="en",
        confidence=0.93,
        metadata={
            "title": "Secret Python Trick That Will Blow Your Mind",
            "duration": 134.0,
            "channel": "CodeMaster Pro"
        }
    )


async def run_analysis_demo():
    """Run a complete analysis demonstration."""
    print("🎬 YouTube to Twitter Clip Analysis Demo")
    print("=" * 50)
    
    # Create sample transcript
    transcript = create_sample_transcript()
    print(f"📝 Sample transcript created:")
    print(f"   Duration: {transcript.duration:.1f} seconds")
    print(f"   Language: {transcript.language}")
    print(f"   Segments: {len(transcript.segments)}")
    print(f"   Confidence: {transcript.confidence:.2%}")
    print()
    
    # Check available providers
    available_providers = LLMAnalyzerFactory.get_available_providers()
    print(f"🔗 Available LLM providers: {available_providers}")
    
    if not available_providers:
        print("❌ No LLM providers available (API keys not configured)")
        print("   Please set GOOGLE_API_KEY or GROQ_API_KEY in your environment")
        return
    
    # Use first available provider
    provider = available_providers[0]
    print(f"   Using provider: {provider}")
    print()
    
    try:
        # Analyze the transcript
        print("🤖 Analyzing transcript with LLM...")
        analysis = await analyze_content(
            transcript,
            provider=provider,
            max_clips=3,
            target_duration=60
        )
        
        print(f"✅ Analysis completed in {analysis.analysis_time:.2f}s")
        print()
        
        # Display results
        print("📊 Analysis Results:")
        print(f"   Content Type: {analysis.content_type.value}")
        print(f"   Summary: {analysis.summary}")
        print(f"   Provider: {analysis.provider}")
        print(f"   Recommendations: {len(analysis.recommendations)}")
        print()
        
        # Show recommendations
        print("🎯 Clip Recommendations:")
        for i, rec in enumerate(analysis.recommendations, 1):
            print(f"   {i}. [{rec.start_time} - {rec.end_time}] (Confidence: {rec.confidence}%)")
            print(f"      Hook Strength: {rec.hook_strength.value}")
            print(f"      Reasoning: {rec.reasoning}")
            print(f"      Keywords: {', '.join(rec.keywords)}")
            print(f"      Sentiment: {rec.sentiment}")
            print()
        
        # Apply content strategy optimization
        print("🎯 Content Strategy Analysis:")
        strategy = ContentStrategy.detect_strategy(analysis)
        print(f"   Detected Strategy: {strategy.value}")
        
        optimized_clips = ContentStrategy.optimize_recommendations(
            analysis,
            strategy=strategy,
            max_clips=2,
            target_duration=60
        )
        
        print(f"   Optimized to {len(optimized_clips)} clips")
        print()
        
        # Generate Twitter content
        print("🐦 Twitter Content Generation:")
        for i, clip in enumerate(optimized_clips, 1):
            twitter_text = ContentStrategy.generate_twitter_text(clip, max_length=280)
            print(f"   Clip {i}:")
            print(f"   Time: {clip.start_time} - {clip.end_time}")
            print(f"   Tweet: {twitter_text}")
            print(f"   Length: {len(twitter_text)} characters")
            print()
        
        # Competition analysis
        competition = ContentStrategy.analyze_competition(optimized_clips)
        print("📈 Competition Analysis:")
        print(f"   Total Duration: {competition['total_duration']:.1f}s")
        print(f"   Average Confidence: {competition['avg_confidence']:.1f}%")
        print(f"   Hook Distribution: {competition['hook_distribution']}")
        
        if competition['recommendations']:
            print("   Recommendations:")
            for rec in competition['recommendations']:
                print(f"   - {rec}")
        else:
            print("   ✅ No improvement recommendations - clips look good!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print(f"   Error type: {type(e).__name__}")


async def run_provider_comparison():
    """Compare different LLM providers if multiple are available."""
    print("\n🔄 Provider Comparison")
    print("=" * 30)
    
    available_providers = LLMAnalyzerFactory.get_available_providers()
    
    if len(available_providers) < 2:
        print("ℹ️  Only one provider available, skipping comparison")
        return
    
    transcript = create_sample_transcript()
    results = {}
    
    for provider in available_providers:
        try:
            print(f"   Testing {provider}...")
            analysis = await analyze_content(
                transcript,
                provider=provider,
                max_clips=2,
                target_duration=60
            )
            results[provider] = {
                "time": analysis.analysis_time,
                "clips": len(analysis.recommendations),
                "avg_confidence": sum(r.confidence for r in analysis.recommendations) / len(analysis.recommendations) if analysis.recommendations else 0,
                "content_type": analysis.content_type.value
            }
        except Exception as e:
            results[provider] = {"error": str(e)}
    
    print("\n📊 Comparison Results:")
    for provider, result in results.items():
        print(f"   {provider.upper()}:")
        if "error" in result:
            print(f"     ❌ Error: {result['error']}")
        else:
            print(f"     ⏱️  Time: {result['time']:.2f}s")
            print(f"     🎯 Clips: {result['clips']}")
            print(f"     📊 Avg Confidence: {result['avg_confidence']:.1f}%")
            print(f"     🏷️  Content Type: {result['content_type']}")


def main():
    """Main function to run the demo."""
    print("Checking configuration...")
    
    # Check config
    validation = config.validate_config()
    print(f"Config validation: {validation}")
    
    if not validation["api_keys"]:
        print("\n⚠️  No API keys configured!")
        print("To use this demo, please set one or both of:")
        print("   export GOOGLE_API_KEY='your-gemini-key'")
        print("   export GROQ_API_KEY='your-groq-key'")
        print("\nYou can also create a .env file with these keys.")
        return
    
    # Run async demos
    asyncio.run(run_analysis_demo())
    asyncio.run(run_provider_comparison())


if __name__ == "__main__":
    main()
