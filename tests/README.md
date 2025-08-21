# Tests

Comprehensive test suite for the YouTube to Twitter clip extraction application.

## Structure

### Core Tests
- `test_setup.py` - Project setup and dependency validation
- `test_downloader.py` - YouTube downloader and thumbnail extraction (556 lines)
- `test_whisper_transcriber.py` - Whisper transcription functionality (267 lines)
- `test_llm_analysis.py` - LLM analysis with Gemini/Groq (557 lines)
- `test_clip_extraction.py` - Video clip extraction (576 lines)
- `test_twitter_optimization.py` - Twitter format optimization (634 lines)
- `test_streamlit_app.py` - Streamlit UI functionality (300 lines)
- `test_ui_components.py` - UI component testing (603 lines)

### Examples & Demos
- `examples/` - Example scripts and integration demonstrations
  - `example_clip_extraction.py` - Clip extraction demonstration
  - `example_llm_analysis.py` - LLM analysis demonstration
  - `example_transcription_integration.py` - Transcription demonstration
  - `test_enhanced_ui.py` - Enhanced UI testing/demo

## Running Tests

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test file
uv run python -m pytest tests/test_downloader.py -v

# Run with coverage
uv run python -m pytest tests/ --cov=src --cov-report=html
```

## Test Status

- **Total Tests**: 169 tests
- **Passing**: 148 tests (87.5%)
- **Failing**: 21 tests (mostly dependency-related)

Core functionality is well-tested and stable.