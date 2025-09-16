#!/usr/bin/env python3
"""
Test the reverted YouTube downloader without proxy
"""

import os
import sys
from pathlib import Path

# Clear any proxy environment variables
proxy_env_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
for var in proxy_env_vars:
    if var in os.environ:
        del os.environ[var]

# Set basic environment variables
os.environ["LOG_LEVEL"] = "INFO"
os.environ["PROXY_USERNAME"] = ""
os.environ["PROXY_PASSWORD"] = ""

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_youtube_downloader():
    """Test YouTube downloader without proxy."""
    try:
        print("🎥 Testing reverted YouTube downloader...")
        
        # Import the downloader using importlib to avoid circular imports
        import importlib.util
        downloader_path = project_root / 'src' / 'downloader' / 'youtube_downloader.py'
        spec = importlib.util.spec_from_file_location("youtube_downloader", downloader_path)
        yd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(yd_module)
        
        downloader = yd_module.YouTubeDownloader()
        
        # Test with a simple video
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print(f"Testing URL: {test_url}")
        
        info = downloader.get_video_info(test_url)
        
        if info:
            print(f"✅ YouTube extraction successful!")
            print(f"   Title: {info.get('title', 'Unknown')}")
            print(f"   Duration: {info.get('duration', 'Unknown')}s")
            print(f"   Uploader: {info.get('uploader', 'Unknown')}")
            return True
        else:
            print("❌ No video info returned")
            return False
            
    except Exception as e:
        print(f"❌ YouTube extraction failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Reverted YouTube Downloader (No Proxy)")
    print("=" * 50)
    
    success = test_youtube_downloader()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ YouTube downloader works without proxy!")
        print("Ready for deployment!")
    else:
        print("❌ YouTube downloader has issues - need to investigate")
