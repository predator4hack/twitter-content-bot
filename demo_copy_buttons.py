#!/usr/bin/env python3
"""
Demo Streamlit app to test the copy button functionality.

Run with: streamlit run demo_copy_buttons.py
"""

import streamlit as st
import streamlit.components.v1 as components
import html

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

def main():
    st.title("📋 Copy Button Test Demo")
    st.write("This demo tests the JavaScript clipboard copying functionality.")
    
    # Sample tweets
    sample_tweets = [
        "🧵 This is a sample tweet about AI and machine learning that you can copy to test the functionality!",
        "2/ Here's another tweet with 'quotes' and special characters: @mention #hashtag",
        "3/ Final tweet with emojis 🚀🤖 and a link: https://example.com",
        "Full thread:\n\nTweet 1 content\n\nTweet 2 content\n\nTweet 3 content"
    ]
    
    st.write("**Individual Tweet Copy Buttons:**")
    
    for i, tweet in enumerate(sample_tweets[:3]):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.text_area(f"Tweet {i+1}", value=tweet, height=80, key=f"tweet_{i}")
        
        with col2:
            st.write("") # Spacing
            copy_button_html = copy_to_clipboard(tweet, f"tweet_{i}")
            components.html(copy_button_html, height=50)
    
    st.write("---")
    st.write("**Full Thread Copy Button:**")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.text_area("Full Thread", value=sample_tweets[3], height=120)
    
    with col2:
        st.write("") # Spacing
        copy_full_html = copy_to_clipboard(sample_tweets[3], "full_thread")
        components.html(copy_full_html, height=50)
    
    st.write("---")
    st.info("""
    **How to test:**
    1. Click any 📋 Copy button
    2. The button should turn green and show "✅ Copied!" for 2 seconds
    3. Paste (Ctrl+V) in any text editor to verify the content was copied
    4. This works on localhost and HTTPS sites with modern browsers
    """)

if __name__ == "__main__":
    main()