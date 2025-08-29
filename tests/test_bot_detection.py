#!/usr/bin/env python3
"""
Test script for YouTube bot detection handling.

This script tests the enhanced bot detection avoidance features
in the YouTube downloader.
"""

import sys
import os
from pathlib import Path

def main():
    """Main test function."""
    print("YouTube Bot Detection Test")
    print("=" * 40)
    print("Testing enhanced bot detection avoidance...")
    
    # Set environment for testing
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("RATE_LIMIT_DELAY", "0.5")  # Faster for testing
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        # Import the downloader class directly to avoid circular imports
        from src.downloader.youtube_downloader import YouTubeDownloader
        
        test_video_info_extraction(YouTubeDownloader)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   This might be due to missing dependencies or circular imports.")
        print("   Try running: pip install -r requirements.txt")
        return 1
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print("If you see bot detection errors, check:")
    print("1. docs/BOT_DETECTION_SOLUTIONS.md")
    print("2. Your network configuration")
    print("3. Try running from a different IP/location")
    print(f"{'='*60}")
    return 0


def test_video_info_extraction(downloader_class):
    """Test video info extraction with bot detection handling."""
    
    # Test videos (use well-known public videos)
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll - very stable
        "https://youtu.be/dQw4w9WgXcQ",                # Short format
    ]
    
    try:
        downloader = downloader_class()
    except Exception as e:
        print(f"❌ Failed to create downloader: {e}")
        return
    
    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"Testing URL: {url}")
        print(f"{'='*60}")
        
        try:
            info = downloader.get_video_info(url)
            print(f"✅ SUCCESS: Video info extracted")
            print(f"   Title: {info['title']}")
            print(f"   Duration: {info['duration']}s")
            print(f"   Uploader: {info['uploader']}")
            print(f"   View Count: {info['view_count']:,}")
            
            # Check if alternative strategy was used
            if 'extraction_strategy' in info:
                print(f"   🔄 Used alternative strategy: {info['extraction_strategy']}")
            else:
                print(f"   ✅ Used primary extraction method")
                
        except Exception as e:
            print(f"❌ FAILED: {e}")
            
            # Check if it's a bot detection error
            if "bot detection" in str(e).lower():
                print(f"   🤖 Bot detection encountered - this is expected in some environments")
                print(f"   💡 Recommendation: Try deploying with proxy settings or different region")
            elif "sign in" in str(e).lower():
                print(f"   🔐 Authentication required - this video may have restrictions")
            else:
                print(f"   ⚠️  Unexpected error type")


if __name__ == "__main__":
    sys.exit(main())
