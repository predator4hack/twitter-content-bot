#!/bin/bash

# YouTube to Twitter Content Extraction App
# Simple script to run the Streamlit application

echo "Starting YouTube to Twitter Content Extraction App..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with UV..."
    uv venv --python 3.11
    echo "Syncing dependencies..."
    uv sync
fi

# Run the app
echo "Starting Streamlit app on http://localhost:8501"
.venv/bin/python -m streamlit run src/ui/streamlit_app.py --server.port 8501
