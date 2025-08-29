# ==================================
# STAGE 1: The Build Stage
# ==================================
FROM python:3.11-slim AS build

# Set a working directory
WORKDIR /app

# Install minimal system dependencies and clean up in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Create a Python virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements file
COPY requirements.txt .

# Create a constraint file to block triton
RUN echo "triton==0" > /tmp/constraints.txt

# Install CPU-only PyTorch and other requirements in one layer
RUN /opt/venv/bin/pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchaudio==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt -c /tmp/constraints.txt && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + && \
    rm -rf /root/.cache/pip/*

# Copy application source code
COPY . .

# Download and cache the 'small' Whisper model (CPU-only)
RUN /opt/venv/bin/python -c "from faster_whisper import WhisperModel; print('Loading Whisper model...'); model = WhisperModel('small', device='cpu'); print('Whisper model loaded successfully')"

# ==================================
# STAGE 2: The Final Runtime Image
# ==================================
FROM python:3.11-slim AS final

# Set a working directory
WORKDIR /app

# Install minimal runtime system dependencies and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Copy the virtual environment and app code
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app /app

# Set PATH for virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables to force CPU usage and configure bot detection avoidance
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
ENV WHISPER_DEVICE=cpu
ENV WHISPER_COMPUTE_TYPE=int8
ENV USE_MOBILE_CLIENTS=true
ENV RANDOMIZE_USER_AGENTS=true
ENV RATE_LIMIT_DELAY=1.0
ENV MAX_RETRIES=5
ENV REQUEST_TIMEOUT=30

# Expose Streamlit port
EXPOSE 8080

# Start the Streamlit service
CMD ["streamlit", "run", "src/ui/streamlit_app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]