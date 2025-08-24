#!/usr/bin/env python3
"""
Deployment readiness checker for Twitter Content Bot
This script verifies that all dependencies and configurations are correct for deployment.
"""

import sys
import subprocess
from pathlib import Path

def check_system_dependencies():
    """Check if system dependencies are available."""
    print("🔍 Checking system dependencies...")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed and working")
            return True
        else:
            print("❌ FFmpeg is not working properly")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg is not installed")
        print("💡 Install with: sudo apt install ffmpeg")
        return False

def check_python_dependencies():
    """Check if Python dependencies are available."""
    print("\n🐍 Checking Python dependencies...")
    
    required_packages = [
        ('streamlit', 'streamlit'),
        ('av', 'PyAV'),
        ('ffmpeg', 'ffmpeg-python'),
        ('yt_dlp', 'yt-dlp'),
        ('whisper', 'openai-whisper'),
        ('faster_whisper', 'faster-whisper'),
    ]
    
    success = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} is available")
        except ImportError:
            print(f"❌ {name} is missing")
            success = False
    
    return success

def check_deployment_files():
    """Check if deployment files are present."""
    print("\n📁 Checking deployment files...")
    
    required_files = [
        ('packages.txt', 'System dependencies for Streamlit Cloud'),
        ('requirements.txt', 'Python dependencies'),
        ('.env.example', 'Environment variables template'),
        ('src/ui/streamlit_app.py', 'Main Streamlit application'),
        ('DEPLOYMENT.md', 'Deployment documentation'),
    ]
    
    success = True
    for file_path, description in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - {description}")
        else:
            print(f"❌ {file_path} - {description} (MISSING)")
            success = False
    
    return success

def check_app_imports():
    """Check if the app can import correctly."""
    print("\n📦 Checking application imports...")
    
    try:
        # Add project root to path
        sys.path.insert(0, str(Path.cwd()))
        
        from src.core.config import config
        print("✅ Core config import successful")
        
        from src.ui.components import initialize_session_state
        print("✅ UI components import successful")
        
        from src.downloader import YouTubeURLValidator
        print("✅ Downloader components import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Application imports failed: {e}")
        return False

def main():
    """Run all deployment readiness checks."""
    print("🚀 Twitter Content Bot - Deployment Readiness Check")
    print("=" * 60)
    
    checks = [
        check_system_dependencies,
        check_python_dependencies,
        check_deployment_files,
        check_app_imports,
    ]
    
    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT READINESS SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("🎉 ALL CHECKS PASSED! Your app is ready for deployment.")
        print("\n📋 Next steps:")
        print("1. Push your code to GitHub")
        print("2. Connect to Streamlit Cloud")
        print("3. Add API keys as secrets")
        print("4. Deploy!")
    else:
        print("⚠️  Some checks failed. Please fix the issues above before deploying.")
        print("\n🔧 Common fixes:")
        print("- Install missing system dependencies")
        print("- Install missing Python packages: pip install -r requirements.txt")
        print("- Ensure all files are present in the repository")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
