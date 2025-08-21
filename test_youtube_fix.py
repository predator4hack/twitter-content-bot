#!/usr/bin/env python3
"""
Test script for YouTube downloader fix
Tests the improved YouTube downloader with the problematic URL
"""

import sys
from pathlib import Path
from src.downloader.youtube_downloader import YouTubeDownloader, YouTubeURLValidator
from src.downloader.thumbnail_extractor import ThumbnailExtractor

def test_url_validation():
    """Test URL validation"""
    test_url = "https://www.youtube.com/watch?v=I1_iXwa-7dA"
    
    print("🔍 Testing URL Validation")
    print(f"URL: {test_url}")
    
    is_valid = YouTubeURLValidator.is_valid_youtube_url(test_url)
    print(f"✅ Valid: {is_valid}")
    
    video_id = YouTubeURLValidator.extract_video_id(test_url)
    print(f"📺 Video ID: {video_id}")
    
    normalized = YouTubeURLValidator.normalize_url(test_url)
    print(f"🔗 Normalized: {normalized}")
    
    return is_valid

def test_video_info():
    """Test video info extraction"""
    test_url = "https://www.youtube.com/watch?v=I1_iXwa-7dA"
    
    print("\n📊 Testing Video Info Extraction")
    print(f"URL: {test_url}")
    
    try:
        downloader = YouTubeDownloader()
        info = downloader.get_video_info(test_url)
        
        print("✅ Video info extracted successfully!")
        print(f"📺 Title: {info.get('title', 'N/A')}")
        print(f"⏱️ Duration: {info.get('duration', 0)} seconds")
        print(f"👤 Uploader: {info.get('uploader', 'N/A')}")
        print(f"📊 Formats available: {info.get('formats', 0)}")
        print(f"🌐 Available: {info.get('availability', 'N/A')}")
        
        return True, info
        
    except Exception as e:
        print(f"❌ Failed to extract video info: {e}")
        return False, None

def test_thumbnail_extraction():
    """Test thumbnail extraction"""
    test_url = "https://www.youtube.com/watch?v=I1_iXwa-7dA"
    
    print("\n🖼️ Testing Thumbnail Extraction")
    print(f"URL: {test_url}")
    
    try:
        extractor = ThumbnailExtractor()
        thumbnails = extractor.get_thumbnail_urls(test_url)
        
        print("✅ Thumbnail URLs extracted successfully!")
        print(f"🔗 Found {len(thumbnails)} thumbnail URLs:")
        
        for quality, url in list(thumbnails.items())[:5]:  # Show first 5
            print(f"  📸 {quality}: {url[:80]}...")
        
        if len(thumbnails) > 5:
            print(f"  ... and {len(thumbnails) - 5} more")
        
        return True, thumbnails
        
    except Exception as e:
        print(f"❌ Failed to extract thumbnails: {e}")
        return False, None

def test_format_selection():
    """Test format selection"""
    print("\n🎥 Testing Format Selection")
    
    try:
        downloader = YouTubeDownloader()
        format_selector = downloader._get_format_selector()
        
        print(f"✅ Format selector: {format_selector}")
        print("📋 This selector should handle various quality preferences with fallbacks")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test format selection: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing YouTube Downloader Fix")
    print("=" * 50)
    
    results = {
        'url_validation': False,
        'video_info': False,
        'thumbnail_extraction': False,
        'format_selection': False
    }
    
    # Test URL validation
    results['url_validation'] = test_url_validation()
    
    # Test video info extraction  
    results['video_info'], video_info = test_video_info()
    
    # Test thumbnail extraction
    results['thumbnail_extraction'], thumbnails = test_thumbnail_extraction()
    
    # Test format selection
    results['format_selection'] = test_format_selection()
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 30)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    
    print(f"\n🎯 Overall Result: {overall_status}")
    
    if all_passed:
        print("\n🚀 YouTube downloader is working correctly!")
        print("   The problematic URL should now work in the main application.")
    else:
        print("\n🔧 Some issues remain. Check the error messages above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
