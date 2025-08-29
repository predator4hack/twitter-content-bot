#!/usr/bin/env python3
"""
Simple bot detection test without circular imports.
"""

import os
import sys
from pathlib import Path

# Set environment variables first
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("RATE_LIMIT_DELAY", "1.0")
os.environ.setdefault("TEMP_DIR", "./temp")
os.environ.setdefault("OUTPUT_DIR", "./output")
os.environ.setdefault("CACHE_DIR", "./cache")
os.environ.setdefault("WHISPER_MODEL", "small")
os.environ.setdefault("WHISPER_DEVICE", "cpu")
os.environ.setdefault("MAX_VIDEO_DURATION", "600")
os.environ.setdefault("YTDL_QUALITY", "720p")
os.environ.setdefault("YTDL_FORMAT", "mp4")
os.environ.setdefault("EXTRACT_THUMBNAILS", "false")
os.environ.setdefault("CLEANUP_TEMP_FILES", "false")

def test_simple_youtube_access():
    """Test simple YouTube access with yt-dlp directly."""
    import yt_dlp
    
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print(f"Testing YouTube access with URL: {test_url}")
    print("-" * 60)
    
    # Basic options similar to our implementation
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Accept-Encoding': 'gzip,deflate',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
            'Keep-Alive': '300',
            'Connection': 'keep-alive',
        },
        'retries': 5,
        'fragment_retries': 5,
        'extractor_retries': 5,
        'nocheckcertificate': True,
        'sleep_interval': 1,
        'max_sleep_interval': 5,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
                'player_skip': ['webpage'],
            }
        },
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("🔍 Extracting video info...")
            info = ydl.extract_info(test_url, download=False)
            
            if info:
                print("✅ SUCCESS: Video info extracted with primary method")
                print(f"   Title: {info.get('title', 'Unknown')}")
                print(f"   Duration: {info.get('duration', 0)}s")
                print(f"   Uploader: {info.get('uploader', 'Unknown')}")
                print(f"   View Count: {info.get('view_count', 0):,}")
                return True
            else:
                print("❌ No video information extracted")
                return False
                
    except Exception as e:
        error_msg = str(e).lower()
        print(f"❌ Primary extraction failed: {e}")
        
        if any(keyword in error_msg for keyword in ['sign in', 'bot', 'cookies', 'authentication']):
            print("🤖 Bot detection detected! Trying alternative strategies...")
            
            # Try Android client
            android_opts = ydl_opts.copy()
            android_opts['http_headers'] = {
                'User-Agent': 'com.google.android.youtube/17.31.35 (Linux; U; Android 11; SM-G981B) gzip',
            }
            android_opts['extractor_args'] = {
                'youtube': {
                    'player_client': ['android'],
                }
            }
            
            try:
                with yt_dlp.YoutubeDL(android_opts) as ydl:
                    print("📱 Trying Android client strategy...")
                    info = ydl.extract_info(test_url, download=False)
                    if info:
                        print("✅ SUCCESS: Video info extracted with Android strategy")
                        print(f"   Title: {info.get('title', 'Unknown')}")
                        return True
            except Exception as android_e:
                print(f"📱 Android strategy failed: {android_e}")
            
            # Try iOS client
            ios_opts = ydl_opts.copy()
            ios_opts['http_headers'] = {
                'User-Agent': 'com.google.ios.youtube/17.31.4 (iPhone; CPU iPhone OS 14_6 like Mac OS X; en_US)',
            }
            ios_opts['extractor_args'] = {
                'youtube': {
                    'player_client': ['ios'],
                }
            }
            
            try:
                with yt_dlp.YoutubeDL(ios_opts) as ydl:
                    print("📱 Trying iOS client strategy...")
                    info = ydl.extract_info(test_url, download=False)
                    if info:
                        print("✅ SUCCESS: Video info extracted with iOS strategy")
                        print(f"   Title: {info.get('title', 'Unknown')}")
                        return True
            except Exception as ios_e:
                print(f"📱 iOS strategy failed: {ios_e}")
            
            print("❌ All strategies failed - YouTube is blocking this environment")
            print("💡 Recommendations:")
            print("   - Try using a proxy or VPN")
            print("   - Deploy to a different region")
            print("   - Use residential IP addresses")
            print("   - Try again later (bot detection can be temporary)")
            
            return False
        else:
            print(f"❌ Unexpected error: {e}")
            return False

def main():
    """Main test function."""
    print("YouTube Bot Detection Test (Simple)")
    print("=" * 50)
    print("Testing basic YouTube access and bot detection handling...")
    print()
    
    try:
        success = test_simple_youtube_access()
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 Test completed successfully!")
            print("Your bot detection enhancements are working.")
        else:
            print("⚠️  Test encountered issues.")
            print("This may be expected in cloud environments.")
            print("Check docs/BOT_DETECTION_SOLUTIONS.md for solutions.")
        
        return 0 if success else 1
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install yt-dlp")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
