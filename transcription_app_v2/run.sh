#!/bin/bash
# Startup script for Transcription App with GPU support

# Activate virtual environment
source venv/bin/activate

# Set library path for cuDNN
export LD_LIBRARY_PATH=$(pwd)/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Run the app
python app.py
