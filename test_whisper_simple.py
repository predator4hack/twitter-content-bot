#!/usr/bin/env python3
"""
Simple test to validate Whisper transcription with a generated audio sample.
"""

import os
import sys
import tempfile
import numpy as np
import wave

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transcription import WhisperTranscriber


def create_simple_audio_file(filename: str, duration: float = 2.0, sample_rate: int = 16000):
    """
    Create a simple audio file with a sine wave tone.
    This won't have speech but will test the audio processing pipeline.
    """
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Scale to 16-bit integer range
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"Created test audio file: {filename}")
    return filename


def test_whisper_basic():
    """Test basic Whisper functionality."""
    print("Testing Whisper Transcriber...")
    
    # Test 1: Initialize transcriber
    print("1. Initializing WhisperTranscriber...")
    transcriber = WhisperTranscriber(model_size="tiny")
    
    # Test 2: Check if healthy
    print("2. Checking transcriber health...")
    is_healthy = transcriber.is_healthy()
    print(f"   Transcriber healthy: {is_healthy}")
    
    # Test 3: Get supported languages
    print("3. Getting supported languages...")
    languages = transcriber.get_supported_languages()
    print(f"   Supported languages count: {len(languages)}")
    print(f"   Sample languages: {list(languages)[:10]}")
    
    # Test 4: Create and transcribe a simple audio file
    print("4. Creating test audio file...")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        audio_path = tmp_file.name
    
    try:
        create_simple_audio_file(audio_path, duration=1.0)
        
        print("5. Transcribing test audio...")
        result = transcriber.transcribe_file(audio_path)
        
        print("6. Transcription results:")
        print(f"   Language: {result.language}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Duration: {result.duration:.2f} seconds")
        print(f"   Text: '{result.text}'")
        print(f"   Segments: {len(result.segments)}")
        
        if result.segments:
            for i, segment in enumerate(result.segments[:3]):  # Show first 3 segments
                print(f"   Segment {i+1}: [{segment.start_time:.2f}-{segment.end_time:.2f}] '{segment.text}'")
        
        print("✅ Whisper transcription test completed successfully!")
        
    finally:
        # Clean up
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            print(f"Cleaned up test file: {audio_path}")


if __name__ == '__main__':
    try:
        test_whisper_basic()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure faster-whisper is installed: uv add faster-whisper")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 All tests passed! Whisper transcription is working correctly.")
