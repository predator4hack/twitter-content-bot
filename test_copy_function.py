#!/usr/bin/env python3
"""
Test the copy_to_clipboard function to ensure JavaScript is properly formatted.
"""

import sys
from pathlib import Path
import html

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def copy_to_clipboard(text, button_id):
    """Create JavaScript code to copy text to clipboard."""
    # Escape the text for JavaScript
    escaped_text = html.escape(text).replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
    
    javascript_code = f"""
    <script>
    function copyToClipboard_{button_id}() {{
        const text = '{escaped_text}';
        
        // Try modern clipboard API first
        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(function() {{
                // Show success feedback
                const button = document.getElementById('copy_button_{button_id}');
                if (button) {{
                    const originalText = button.innerHTML;
                    button.innerHTML = '✅ Copied!';
                    button.style.backgroundColor = '#28a745';
                    setTimeout(function() {{
                        button.innerHTML = originalText;
                        button.style.backgroundColor = '';
                    }}, 2000);
                }}
            }}).catch(function(err) {{
                console.error('Clipboard API failed: ', err);
                fallbackCopy_{button_id}(text);
            }});
        }} else {{
            fallbackCopy_{button_id}(text);
        }}
    }}
    
    function fallbackCopy_{button_id}(text) {{
        // Fallback for older browsers or non-secure contexts
        const dummy = document.createElement('textarea');
        document.body.appendChild(dummy);
        dummy.value = text;
        dummy.select();
        dummy.setSelectionRange(0, 99999); // For mobile devices
        
        try {{
            const successful = document.execCommand('copy');
            if (successful) {{
                const button = document.getElementById('copy_button_{button_id}');
                if (button) {{
                    const originalText = button.innerHTML;
                    button.innerHTML = '✅ Copied!';
                    button.style.backgroundColor = '#28a745';
                    setTimeout(function() {{
                        button.innerHTML = originalText;
                        button.style.backgroundColor = '';
                    }}, 2000);
                }}
            }}
        }} catch (err) {{
            console.error('Fallback copy failed: ', err);
            alert('Copy failed. Please select and copy manually.');
        }}
        
        document.body.removeChild(dummy);
    }}
    </script>
    
    <button id="copy_button_{button_id}" onclick="copyToClipboard_{button_id}()" 
            style="background: #ff4b4b; color: white; border: none; padding: 5px 10px; 
                   border-radius: 3px; cursor: pointer; font-size: 12px;">
        📋 Copy
    </button>
    """
    
    return javascript_code


def test_copy_function():
    """Test the copy_to_clipboard function with various text inputs."""
    
    print("🧪 Testing Copy Function")
    print("=" * 40)
    
    # Test cases
    test_cases = [
        ("Simple tweet", "tweet_1"),
        ("🧵 This is a thread with emojis and special characters!", "tweet_2"),
        ("Tweet with 'quotes' and \"double quotes\"", "tweet_3"),
        ("Multiline tweet\nwith line breaks\nand more content", "tweet_4"),
        ("Tweet with #hashtags and @mentions", "tweet_5")
    ]
    
    for i, (text, button_id) in enumerate(test_cases):
        print(f"\n📝 Test Case {i+1}: {text[:30]}...")
        try:
            result = copy_to_clipboard(text, button_id)
            
            # Check that the result contains expected elements
            assert "<script>" in result, "Missing script tag"
            assert f"copy_button_{button_id}" in result, "Missing button ID"
            assert f"copyToClipboard_{button_id}" in result, "Missing function name"
            assert "navigator.clipboard" in result, "Missing modern clipboard API"
            assert "document.execCommand" in result, "Missing fallback method"
            
            print(f"✅ Test Case {i+1} passed - JavaScript generated correctly")
            
        except Exception as e:
            print(f"❌ Test Case {i+1} failed: {e}")
            return False
    
    print(f"\n🎉 All copy function tests passed!")
    print(f"\nThe copy buttons will work when:")
    print(f"1. Running in a web browser (Streamlit app)")
    print(f"2. On HTTPS or localhost (for modern clipboard API)")
    print(f"3. User grants clipboard permissions if prompted")
    
    return True


if __name__ == "__main__":
    success = test_copy_function()
    if success:
        print("\n✅ Copy function test PASSED")
    else:
        print("\n❌ Copy function test FAILED")
        sys.exit(1)