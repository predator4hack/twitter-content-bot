#!/usr/bin/env python3
"""
Command-line interface for testing Whisper transcription functionality.

This script allows you to test the Whisper transcriber with audio/video files
or generate a test audio file to validate the transcription system.
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transcription import WhisperTranscriber, TranscriptionResult


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_test_audio(output_path: str, duration: int = 5) -> str:
    """
    Create a simple test audio file using system text-to-speech.
    
    Args:
        output_path: Path where to save the audio file
        duration: Duration in seconds (approximate)
        
    Returns:
        Path to the created audio file
    """
    import subprocess
    
    # Text to convert to speech
    test_text = "Hello world. This is a test of the Whisper speech recognition system. The quick brown fox jumps over the lazy dog."
    
    try:
        # Try using espeak (Linux text-to-speech)
        cmd = [
            'espeak',
            '-w', output_path,  # Write to WAV file
            '-s', '150',        # Speed (words per minute)
            test_text
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Created test audio file: {output_path}")
        return output_path
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If espeak is not available, try festival
        try:
            # Create a temporary text file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_text)
                text_file = f.name
            
            cmd = [
                'text2wave',
                text_file,
                '-o', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            os.unlink(text_file)  # Clean up temp file
            print(f"Created test audio file: {output_path}")
            return output_path
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: No text-to-speech system found (espeak or festival).")
            print("You'll need to provide your own audio file for testing.")
            return None


def transcribe_file(file_path: str, model_size: str = "base", language: str = None) -> TranscriptionResult:
    """
    Transcribe an audio or video file using Whisper.
    
    Args:
        file_path: Path to the file to transcribe
        model_size: Whisper model size to use
        language: Language code (auto-detect if None)
        
    Returns:
        TranscriptionResult object
    """
    print(f"Initializing Whisper transcriber (model: {model_size})...")
    transcriber = WhisperTranscriber(model_size=model_size)
    
    print("Checking transcriber health...")
    if not transcriber.is_healthy():
        raise RuntimeError("Transcriber is not healthy")
    
    print(f"Transcribing file: {file_path}")
    result = transcriber.transcribe_file(file_path, language=language)
    
    return result


def print_transcription_result(result: TranscriptionResult):
    """Print the transcription result in a formatted way."""
    print("\n" + "="*60)
    print("TRANSCRIPTION RESULT")
    print("="*60)
    print(f"Language: {result.language}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Duration: {result.duration:.2f} seconds")
    print(f"Word count: {result.word_count}")
    print("-"*60)
    print("FULL TEXT:")
    print(result.text)
    print("-"*60)
    
    if result.segments:
        print("SEGMENTS:")
        for i, segment in enumerate(result.segments, 1):
            print(f"{i:2d}. [{segment.start_time:6.2f}-{segment.end_time:6.2f}] {segment.text}")
    
    print("-"*60)
    print("METADATA:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")
    print("="*60)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Test Whisper transcription functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create and transcribe a test audio file
  python test_whisper_cli.py --create-test
  
  # Transcribe an existing audio file
  python test_whisper_cli.py --file audio.wav
  
  # Transcribe with specific model and language
  python test_whisper_cli.py --file video.mp4 --model small --language en
  
  # Enable debug logging
  python test_whisper_cli.py --file audio.wav --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to audio or video file to transcribe'
    )
    
    parser.add_argument(
        '--create-test', '-t',
        action='store_true',
        help='Create a test audio file and transcribe it'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='base',
        choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large'],
        help='Whisper model size to use (default: base)'
    )
    
    parser.add_argument(
        '--language', '-l',
        type=str,
        help='Language code (e.g., en, es, fr). Auto-detect if not specified.'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for test audio (default: temp file)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        if args.create_test:
            # Create test audio file
            if args.output:
                audio_path = args.output
            else:
                audio_path = tempfile.mktemp(suffix='.wav')
            
            created_audio = create_test_audio(audio_path)
            if created_audio:
                print(f"Transcribing test audio: {created_audio}")
                result = transcribe_file(created_audio, args.model, args.language)
                print_transcription_result(result)
                
                # Clean up temp file if we created one
                if not args.output:
                    os.unlink(created_audio)
                    print(f"Cleaned up temporary file: {created_audio}")
            else:
                print("Could not create test audio file. Please install espeak or festival.")
                return 1
                
        elif args.file:
            # Transcribe provided file
            if not os.path.exists(args.file):
                print(f"Error: File not found: {args.file}")
                return 1
            
            result = transcribe_file(args.file, args.model, args.language)
            print_transcription_result(result)
            
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Transcription failed")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
