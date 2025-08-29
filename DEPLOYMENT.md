# Deployment Guide

## Streamlit Cloud Deployment

### Prerequisites

1. GitHub repository with your code
2. Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io/))
3. API keys for LLM services (Groq or Google Gemini)

### Files Required for Deployment

-   `packages.txt` - System dependencies (FFmpeg) ✅
-   `requirements.txt` - Python dependencies ✅
-   `.env.example` - Environment variable template ✅
-   `src/ui/streamlit_app.py` - Main application ✅

### Deployment Steps

1. **Push to GitHub**

    ```bash
    git add .
    git commit -m "Add deployment files"
    git push origin main
    ```

2. **Connect to Streamlit Cloud**

    - Go to [share.streamlit.io](https://share.streamlit.io/)
    - Click "New app"
    - Connect your GitHub repository
    - Set main file path: `src/ui/streamlit_app.py`

3. **Configure Environment Variables**
   In Streamlit Cloud settings, add these secrets:

    ```
    GROQ_API_KEY = "your_groq_api_key_here"
    # OR
    GOOGLE_API_KEY = "your_google_api_key_here"
    ```

4. **Deploy**
    - Click "Deploy"
    - Streamlit will automatically:
        - Install system packages from `packages.txt`
        - Install Python packages from `requirements.txt`
        - Start your application

### Troubleshooting

#### Common Issues:

1. **FFmpeg not found**: Ensure `packages.txt` contains `ffmpeg`
2. **Missing API keys**: Add them as secrets in Streamlit Cloud settings
3. **Import errors**: Check that all packages are in `requirements.txt`
4. **"No module named 'av'" error**:
    - Ensure `av>=15.0.0` is in `requirements.txt`
    - For local development: `pip install av`
    - This package (PyAV) is required for audio/video processing
5. **"No module named 'faster_whisper'" error**:
    - Ensure `faster-whisper>=1.2.0` is in `requirements.txt`
    - For local development: `pip install faster-whisper`
    - This package is required for speech-to-text transcription
6. **CUDA/cuDNN errors** (`libcudnn_ops.so` not found, `cudnnCreateTensorDescriptor` errors):
    - These occur when faster-whisper tries to use GPU but CUDA libraries are missing
    - **Solution**: Force CPU-only mode by setting environment variables:
        ```
        CUDA_VISIBLE_DEVICES=""
        WHISPER_DEVICE=cpu
        WHISPER_COMPUTE_TYPE=int8
        ```
    - For Streamlit Cloud: Add these as secrets in your app settings

#### Logs and Debugging:

-   Check the deployment logs in Streamlit Cloud
-   Use the "Manage app" option to view real-time logs
-   Test locally first with `streamlit run src/ui/streamlit_app.py`

## YouTube Bot Detection Issues

### Common Error
If you encounter this error in cloud deployments:
```
ERROR: [youtube] Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies for the authentication.
```

### Solutions Implemented
The application automatically handles bot detection with:

1. **Multiple extraction strategies**: Android, iOS, TV, and minimal clients
2. **Enhanced headers and user agents**: Realistic browser/mobile headers
3. **Exponential backoff retries**: Automatic retry with increasing delays
4. **Rate limiting**: Configurable delays between requests

### Environment Variables for Bot Detection
Add these to your deployment environment:

```bash
# Rate limiting and retries
RATE_LIMIT_DELAY=2.0
MAX_RETRIES=5
REQUEST_TIMEOUT=30

# Bot detection avoidance
USE_MOBILE_CLIENTS=true
RANDOMIZE_USER_AGENTS=true

# Optional: Proxy support
# HTTP_PROXY=http://your-proxy:8080
# HTTPS_PROXY=https://your-proxy:8080
```

### Cloud Provider Specific
- **GCP Cloud Run**: Use regional deployments, custom VPC with NAT
- **AWS Lambda**: Consider using Elastic IPs or VPN
- **Heroku**: May require proxy add-ons for consistent IPs

For detailed troubleshooting, see [docs/BOT_DETECTION_SOLUTIONS.md](docs/BOT_DETECTION_SOLUTIONS.md).

### Alternative Deployment Options

#### Docker Deployment

If you prefer Docker, create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Run the app
CMD ["streamlit", "run", "src/ui/streamlit_app.py", "--server.port=8501"]
```

#### Local Development

For local development:

1. Install FFmpeg system-wide
2. Create virtual environment with UV:

    ```bash
    uv venv --python 3.11
    source .venv/bin/activate
    uv sync
    ```

3. Set environment variables in `.env`
4. Run with virtual environment Python:

    ```bash
    .venv/bin/python -m streamlit run src/ui/streamlit_app.py
    ```

Alternatively, with UV directly:

```bash
uv run streamlit run src/ui/streamlit_app.py
```
