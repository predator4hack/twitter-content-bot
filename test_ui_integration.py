#!/usr/bin/env python3
"""
Simple test to verify UI integration has the Twitter thread functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_ui_has_thread_functions():
    """Test that the UI has the required thread functions."""
    
    try:
        # Test importing the UI module
        from src.ui import streamlit_app
        
        # Check that the required functions exist
        required_functions = [
            'render_twitter_thread_section',
            'generate_twitter_thread', 
            'render_thread_preview',
            'format_thread_for_export',
            'format_thread_as_json'
        ]
        
        print("🧵 Testing Twitter Thread UI Integration")
        print("=" * 50)
        
        for func_name in required_functions:
            if hasattr(streamlit_app, func_name):
                print(f"✅ {func_name} - Found")
            else:
                print(f"❌ {func_name} - Missing")
                return False
        
        # Test that the main function includes thread section
        with open(Path(__file__).parent / "src/ui/streamlit_app.py", 'r') as f:
            content = f.read()
            
        if "render_twitter_thread_section()" in content:
            print("✅ Thread section integrated into main UI")
        else:
            print("❌ Thread section not integrated into main UI")
            return False
            
        if "🧵 Twitter Thread Generator" in content:
            print("✅ Thread generator UI elements present")
        else:
            print("❌ Thread generator UI elements missing")
            return False
            
        print("\n🎉 All Twitter thread UI components are integrated!")
        print("\nTo use the Twitter thread feature:")
        print("1. Run: streamlit run src/ui/streamlit_app.py")
        print("2. Process a YouTube video")
        print("3. Scroll down to '🧵 Twitter Thread Generator'")
        print("4. Choose your settings and click 'Generate Twitter Thread'")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_ui_has_thread_functions()
    if success:
        print("\n✅ Twitter thread integration test PASSED")
    else:
        print("\n❌ Twitter thread integration test FAILED")
        sys.exit(1)